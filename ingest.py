"""
Optimized Ingestion & Deletion Module with Enhanced Metadata and Record Manager Sync

This module handles reading documents from S3 (supporting PDFs, images, audio, and text),
processing them (including extracting images from PDFs and generating image descriptions),
and indexing documents into a vector store and a key–value docstore.
It uses record managers (SQLRecordManager) to track indexed documents.
During deletion, it ensures that the vector store, docstore, and record manager records (keyed by filename)
are all updated.
Redundant metadata fields (e.g. image_path, audio_path) have been removed.
"""

import os
import uuid
import base64
import tempfile
from io import BytesIO
from typing import Tuple, List
from datetime import datetime

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv
from fastapi import HTTPException
from PIL import Image
import fitz  # PyMuPDF
import PIL

# LangChain imports
from langchain.storage._lc_store import create_kv_docstore
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.indexes import SQLRecordManager, index
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from langchain_community.document_loaders.parsers import OpenAIWhisperParser
from langchain_community.document_loaders.blob_loaders import Blob

# Local module imports
from .kvstore import SQLStore
from .constants import (
    TEXT_RECORD_MANAGER_DB_URL,
    VECTOR_CHROMA_DB_PATH,
    CHROMA_DOCS_INDEX_NAME,
    PARENT_DOC_DB_PATH,
    IMAGE_RECORD_MANAGER_DB_URL,
)
from backend_lcro.embeddings import get_embeddings_model

# Load environment variables
load_dotenv(".config")

def parse_s3_uri(s3_uri: str) -> Tuple[str, str]:
    """Parse an S3 URI and return (bucket_name, prefix)."""
    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI format. It should start with 's3://'.")
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket_name, prefix


def initialize_s3_client():
    """Initialize and return an S3 client, optionally using an AWS profile."""
    profile_name = os.getenv("AWS_PROFILE")
    if profile_name:
        print(f"Using AWS profile: {profile_name}")
        session = boto3.Session(profile_name=profile_name)
        return session.client("s3")
    else:
        print("Using direct AWS credentials")
        return boto3.client(
            "s3",
            aws_access_key_id=os.getenv("aws_access_key_id"),
            aws_secret_access_key=os.getenv("aws_secret_access_key"),
            region_name=os.getenv("region_name"),
        )


def read_pdf_content(content: bytes, source_file_name: str, s3_client) -> Tuple[str, List[dict]]:
    """
    Extract text and images from a PDF file.
    Upload extracted images to S3 and return a tuple (text, images).
    Each image is represented as a dict containing the PIL Image and its metadata.
    """
    images = []
    text = ""
    with fitz.open(stream=content, filetype="pdf") as doc:
        for page_num, page in enumerate(doc, start=1):
            text += page.get_text()
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list, start=1):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]
                image_obj = Image.open(BytesIO(image_bytes))

                # Prepare image for upload
                image_buffer = BytesIO()
                image_obj.save(image_buffer, format=image_ext.upper())
                image_buffer.seek(0)

                # Hard-coded S3 bucket and key for extracted images
                extracted_images_bucket = "lcro-ml-trainings"
                extracted_images_prefix = "extracted_images_from_pdf/"
                image_file_name = f"{source_file_name}_page_{page_num}_image_{img_index}.{image_ext}"
                s3_key = f"{extracted_images_prefix}{image_file_name}"

                try:
                    s3_client.upload_fileobj(
                        image_buffer,
                        Bucket=extracted_images_bucket,
                        Key=s3_key,
                        ExtraArgs={'ContentType': f'image/{image_ext}'}
                    )
                    image_s3_uri = f"s3://{extracted_images_bucket}/{s3_key}"
                except ClientError as e:
                    raise HTTPException(
                        status_code=500,
                        detail=f"Error uploading image to S3: {e.response['Error']['Message']}"
                    )

                images.append({
                    "image": image_obj,
                    "page_num": page_num,
                    "image_index": img_index,
                    "image_ext": image_ext,
                    "s3_uri": image_s3_uri,
                })
    return text, images


def read_object_content(s3_client, bucket_name: str, key: str, s3_uri: str) -> List[Document]:
    """
    Read an S3 object and return a list of Document objects.
    Supports images, PDFs, audio files, and text files.
    Enhanced metadata is used for consistency.
    """
    ingest_time = datetime.utcnow().isoformat() + "Z"
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        content = response["Body"].read()
        file_name = os.path.basename(key)

        if key.lower().endswith((".png", ".jpg", ".jpeg", ".gif", ".webp")):
            image_obj = Image.open(BytesIO(content))
            return [Document(
                page_content="",
                metadata={
                    "filename": file_name,
                    "image": image_obj,
                    "s3_uri": s3_uri,
                    "doc_type": "image",
                    "ingest_timestamp": ingest_time,
                }
            )]

        elif key.lower().endswith(".pdf"):
            text_content, images = read_pdf_content(content, file_name, s3_client)
            docs = [Document(
                page_content=text_content,
                metadata={
                    "filename": file_name,
                    "s3_uri": s3_uri,
                    "doc_type": "pdf_text",
                    "ingest_timestamp": ingest_time,
                }
            )]
            for img_data in images:
                docs.append(Document(
                    page_content="",
                    metadata={
                        "filename": file_name,
                        "image": img_data["image"],
                        "s3_uri": s3_uri,
                        "doc_type": "pdf_image",
                        "page_num": img_data["page_num"],
                        "extracted_image_s3_uri": img_data["s3_uri"],
                        "ingest_timestamp": ingest_time,
                    }
                ))
            return docs

        elif key.lower().endswith((".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm")):
            with tempfile.NamedTemporaryFile(suffix=os.path.splitext(key)[1], delete=False) as temp_audio_file:
                temp_audio_file.write(content)
                temp_audio_path = temp_audio_file.name
            try:
                audio_blob = Blob.from_path(temp_audio_path)
                parser = OpenAIWhisperParser(language='en')
                documents = list(parser.lazy_parse(blob=audio_blob))
                audio_text = "".join(doc.page_content for doc in documents)
                return [Document(
                    page_content=audio_text,
                    metadata={
                        "filename": file_name,
                        "s3_uri": s3_uri,
                        "doc_type": "audio",
                        "ingest_timestamp": ingest_time,
                    }
                )]
            finally:
                os.remove(temp_audio_path)
        else:
            text_content = content.decode("utf-8")
            return [Document(
                page_content=text_content,
                metadata={
                    "filename": file_name,
                    "s3_uri": s3_uri,
                    "doc_type": "text",
                    "ingest_timestamp": ingest_time,
                }
            )]
    except ClientError as e:
        raise HTTPException(
            status_code=403,
            detail=f"Error accessing S3 object {key}: {e.response['Error']['Message']}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error processing {key}: {str(e)}"
        )


def list_directory_contents(s3_client, bucket_name: str, prefix: str, s3_uri: str) -> List[Document]:
    """
    List objects in an S3 directory and read their contents.
    Returns a list of Document objects.
    """
    try:
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        documents = []
        for obj in response.get("Contents", []):
            full_object_uri = f"s3://{bucket_name}/{obj['Key']}"
            documents.extend(read_object_content(s3_client, bucket_name, obj["Key"], full_object_uri))
        return documents
    except ClientError as e:
        raise HTTPException(
            status_code=403,
            detail=f"Error listing S3 directory {prefix}: {e.response['Error']['Message']}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error while listing contents in {prefix}: {str(e)}"
        )


def read_from_s3(s3_uri: str) -> List[Document]:
    """
    Read files from an S3 URI.
    If the prefix does not end with '/', it is treated as a single object; otherwise, as a directory.
    Returns a list of Document objects.
    """
    bucket_name, prefix = parse_s3_uri(s3_uri)
    s3_client = initialize_s3_client()
    if prefix and not prefix.endswith("/"):
        return read_object_content(s3_client, bucket_name, prefix, s3_uri)
    else:
        return list_directory_contents(s3_client, bucket_name, prefix, s3_uri)


def describe_image(image: Image.Image) -> Tuple[str, str]:
    """
    Convert a PIL Image to a base64-encoded JPEG and generate a description via GPT-4.
    Returns a tuple: (description, base64_string).
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input is not a valid PIL Image object")
    if image.mode != "RGB":
        image = image.convert("RGB")
    desired_size = (1300, 600)
    image.thumbnail(desired_size, PIL.Image.Resampling.LANCZOS)
    buffered = BytesIO()
    image.save(buffered, format="JPEG", quality=85, optimize=True)
    base64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

    chat = ChatOpenAI(model="gpt-4o")
    prompt = (
        "You are an assistant tasked with summarizing images for retrieval. "
        "These summaries will be embedded and used to retrieve the raw image. "
        "Give a concise summary of the image that is well optimized for retrieval."
    )
    response = chat.invoke([HumanMessage(content=[
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_string}"}}
    ])])
    return response.content, base64_string


def ingest_docs(s3_uri: str) -> int:
    """
    Ingest documents from an S3 URI, splitting text files into chunks,
    generating image descriptions, and indexing documents into the vector store and docstore.
    Returns the number of raw documents processed.
    """
    try:
        # Initialize vector store, docstore, and record managers.
        vectorstore = Chroma(
            collection_name=CHROMA_DOCS_INDEX_NAME,
            embedding_function=get_embeddings_model(),
            persist_directory=VECTOR_CHROMA_DB_PATH,
        )
        cs = SQLStore(PARENT_DOC_DB_PATH, "docstore")
        docstore = create_kv_docstore(cs)

        text_record_manager = SQLRecordManager(
            f"chroma/{CHROMA_DOCS_INDEX_NAME}", db_url=TEXT_RECORD_MANAGER_DB_URL
        )
        text_record_manager.create_schema()

        image_record_manager = SQLRecordManager(
            f"chroma/{CHROMA_DOCS_INDEX_NAME}", db_url=IMAGE_RECORD_MANAGER_DB_URL
        )
        image_record_manager.create_schema()

        result = read_from_s3(s3_uri)
        if not result:
            raise HTTPException(
                status_code=404,
                detail="No documents found in the specified S3 path."
            )

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        documents_to_index = []
        images_to_index = []

        for doc in result:
            if "image" in doc.metadata:
                # Process image documents (both direct images and PDF-extracted images)
                doc_id = str(uuid.uuid4())
                image_description, base64_image = describe_image(doc.metadata["image"])
                doc_type = doc.metadata.get("doc_type", "image")
                vectorstore_metadata = {
                    "doc_id": doc_id,
                    "filename": doc.metadata["filename"],
                    "s3_uri": doc.metadata["s3_uri"],
                    "doc_type": doc_type,
                    "ingest_timestamp": doc.metadata.get("ingest_timestamp"),
                }
                if "page_num" in doc.metadata:
                    vectorstore_metadata["page_num"] = doc.metadata["page_num"]
                if "extracted_image_s3_uri" in doc.metadata:
                    vectorstore_metadata["extracted_image_s3_uri"] = doc.metadata["extracted_image_s3_uri"]

                vectorstore_doc = Document(
                    page_content=image_description,
                    metadata=vectorstore_metadata,
                )
                docstore_metadata = {
                    "doc_id": doc_id,
                    "filename": doc.metadata["filename"],
                    "s3_uri": doc.metadata["s3_uri"],
                    "doc_type": doc_type,
                    "ingest_timestamp": doc.metadata.get("ingest_timestamp"),
                }
                if "page_num" in doc.metadata:
                    docstore_metadata["page_num"] = doc.metadata["page_num"]
                if "extracted_image_s3_uri" in doc.metadata:
                    docstore_metadata["extracted_image_s3_uri"] = doc.metadata["extracted_image_s3_uri"]

                docstore_doc = Document(
                    page_content=base64_image,
                    metadata=docstore_metadata,
                )
                images_to_index.append(vectorstore_doc)
                docstore.mset([(doc_id, docstore_doc)])
            else:
                # Process text documents: split into chunks and add additional metadata.
                for idx, chunk in enumerate(text_splitter.split_documents([doc])):
                    chunk.metadata["doc_type"] = doc.metadata.get("doc_type", "text")
                    chunk.metadata["filename"] = doc.metadata.get("filename")
                    chunk.metadata["s3_uri"] = doc.metadata.get("s3_uri")
                    chunk.metadata["ingest_timestamp"] = doc.metadata.get("ingest_timestamp")
                    chunk.metadata["chunk_index"] = idx
                    documents_to_index.append(chunk)

        doc_indexing_stats = index(
            documents_to_index,
            text_record_manager,
            vectorstore,
            cleanup=None,
            source_id_key="filename",
        )
        print("Text docs indexing stats:", doc_indexing_stats)

        image_indexing_stats = index(
            images_to_index,
            image_record_manager,
            vectorstore,
            cleanup="incremental",
            source_id_key="filename",
        )
        print("Image docs indexing stats:", image_indexing_stats)

        return len(result)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


def delete_docs(s3_uri: str):
    """
    Delete documents from the vector store, docstore, and update record managers based on an S3 URI.
    Uses stored metadata (the 's3_uri' field) to locate and remove corresponding documents.
    Also, for each S3 object, it computes the unique key (filename) used as the source ID,
    and calls delete_keys on the record managers.
    """
    try:
        # Initialize vector store, docstore, and record managers.
        vectorstore = Chroma(
            collection_name=CHROMA_DOCS_INDEX_NAME,
            embedding_function=get_embeddings_model(),
            persist_directory=VECTOR_CHROMA_DB_PATH,
        )
        cs = SQLStore(PARENT_DOC_DB_PATH, "docstore")
        docstore = create_kv_docstore(cs)
        text_record_manager = SQLRecordManager(
            f"chroma/{CHROMA_DOCS_INDEX_NAME}", db_url=TEXT_RECORD_MANAGER_DB_URL
        )
        image_record_manager = SQLRecordManager(
            f"chroma/{CHROMA_DOCS_INDEX_NAME}", db_url=IMAGE_RECORD_MANAGER_DB_URL
        )

        bucket_name, prefix = parse_s3_uri(s3_uri)
        s3_client = initialize_s3_client()
        response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        if "Contents" not in response:
            return {"message": "No documents found for deletion."}

        for obj in response.get("Contents", []):
            full_object_uri = f"s3://{bucket_name}/{obj['Key']}"
            # For vectorstore: delete documents matching the s3_uri.
            results = vectorstore._collection.get(where={"s3_uri": full_object_uri})
            ids = results.get("ids", [])
            if ids:
                vectorstore.delete(ids=ids)

            # For docstore: attempt deletion if the stored document's s3_uri matches.
            # (Assuming docstore.delete(key) is available.)
            # Here, we compute the unique key (filename) as used during ingestion.
            filename = os.path.basename(obj["Key"])
            try:
                docstore.delete(filename)
            except Exception:
                # If deletion fails (e.g., key not found), continue.
                pass

            # Update record managers: delete records with the same filename.
            text_record_manager.delete_keys([filename])
            image_record_manager.delete_keys([filename])

        return {"message": "Documents and record manager entries deleted successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Example usage:
    # ingest_docs("s3://lcro-ml-trainings/sources/30october.mp3")
    # ingest_docs("s3://lcro-ml-trainings/20131231103232738561744.pdf")
    pass
