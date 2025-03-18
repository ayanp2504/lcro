import os
import boto3
import PyPDF2
import uuid
import base64
from langchain.storage._lc_store import create_kv_docstore
from langchain_openai import ChatOpenAI
from fastapi import HTTPException
from langchain_core.messages import HumanMessage
from langchain.indexes import SQLRecordManager, index
from botocore.exceptions import ClientError
from io import BytesIO
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.retrievers import MultiVectorRetriever
from .kvstore import SQLStore
from .constants import (TEXT_RECORD_MANAGER_DB_URL, VECTOR_CHROMA_DB_PATH, CHROMA_DOCS_INDEX_NAME, PARENT_DOC_DB_PATH, IMAGE_RECORD_MANAGER_DB_URL)
from backend_lcro.embeddings import get_embeddings_model
from PIL import Image
import fitz 


# Load environment variables
load_dotenv('.config')

def parse_s3_uri(s3_uri):
    """Parse the S3 URI to extract the bucket name and prefix."""
    if not s3_uri.startswith("s3://"):
        raise ValueError("Invalid S3 URI format. It should start with 's3://'.")
    parts = s3_uri.replace("s3://", "").split("/", 1)
    bucket_name = parts[0]
    prefix = parts[1] if len(parts) > 1 else ""
    return bucket_name, prefix

def initialize_s3_client():
    """Initialize and return an S3 client, using profile if specified."""
    profile_name = os.getenv("AWS_PROFILE")
    if profile_name:
        print("Using Profile name")
        session = boto3.Session(profile_name=profile_name)
        return session.client('s3')
    else:
        print("Using object")
        return boto3.client(
    's3',
    aws_access_key_id=os.getenv("aws_access_key_id"), 
    aws_secret_access_key=os.getenv("aws_secret_access_key"), 
    region_name=os.getenv("region_name"),
    aws_session_token=os.getenv("aws_session_token")
)


# def read_pdf_content(content):
#     """Extract text from a PDF file."""
#     pdf_file = PyPDF2.PdfReader(BytesIO(content))
#     text = ""
#     for page in pdf_file.pages:
#         text += page.extract_text()
#     return text

def read_pdf_content(content):
    """Extract text and images from a PDF file."""
    pdf_stream = BytesIO(content)  # Create a BytesIO stream from content
    pdf_document = fitz.open(stream=pdf_stream, filetype="pdf")  # Specify filetype as "pdf"
    
    text = ""
    images = []

    # Extract text using PyPDF2 (if needed)
    pdf_reader = PyPDF2.PdfReader(pdf_stream)
    for page in pdf_reader.pages:
        text += page.extract_text() or ""

    # Extract images using PyMuPDF
    for page_number in range(len(pdf_document)):
        page = pdf_document[page_number]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_document.extract_image(xref)
            image_bytes = base_image["image"]
            image_filename = f"image_page{page_number + 1}_{img_index + 1}.png"

            # Save the image to a file or handle it as needed
            with open(image_filename, "wb") as img_file:
                img_file.write(image_bytes)

            images.append(image_filename)  # Store the filename of the extracted image

    pdf_document.close()
    
    return text, images  # Return both text and list of image filenames

def read_object_content(s3_client, bucket_name, key, s3_uri):
    """Read the content of an object from S3 and return a Document object."""
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=key)
        content = response['Body'].read()
        file_name = os.path.basename(key)

        if key.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', ".webp")):
            # For images, return a Document with image content and metadata
            image = Image.open(BytesIO(content))
            return Document(page_content="", metadata={
                'source': file_name,
                'image': image,
                'image_path': f"{bucket_name}/{key}",
                "s3_uri": s3_uri
            })
        
        elif key.lower().endswith('.pdf'):
            text_content = read_pdf_content(content)
        else:
            text_content = content.decode('utf-8')

        
        return Document(page_content=text_content, metadata={'source': file_name, 's3_uri': s3_uri})
    except ClientError as e:
        raise HTTPException(status_code=403, detail=f"Error accessing S3 object {key}: {e.response['Error']['Message']}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error processing {key}: {str(e)}")

def list_directory_contents(s3_client, bucket_name, prefix, s3_uri):
    """List all objects in a directory on S3 and read their contents, returning Document objects."""
    try:
        objects = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
        contents = []
        for obj in objects.get('Contents', []):
            full_object_uri = f"s3://{bucket_name}/{obj['Key']}"  # Construct full object URI
            document = read_object_content(s3_client, bucket_name, obj['Key'],  full_object_uri)
            if document is not None:
                contents.append(document)
        return contents
    except ClientError as e:
        raise HTTPException(status_code=403, detail=f"Error listing S3 directory {prefix}: {e.response['Error']['Message']}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error while listing contents in {prefix}: {str(e)}")

def read_from_s3(s3_uri):
    """Read files from an S3 URI, whether it's a directory or a specific object."""
    bucket_name, prefix = parse_s3_uri(s3_uri)
    s3_client = initialize_s3_client()

    if not prefix.endswith('/'):
        return [read_object_content(s3_client, bucket_name, prefix, s3_uri)]
    else:
        return list_directory_contents(s3_client, bucket_name, prefix, s3_uri)

def describe_image(image):
    """
    Takes a PIL Image object, encodes it to base64, and generates a summary using GPT-4.
    
    Args:
    image: PIL Image object
    
    Returns:
    string: Summary of the image content
    """
    if not isinstance(image, Image.Image):
        raise ValueError("Input is not a valid PIL Image object")
    
    # Convert image to RGB mode if it's not
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Encode image to base64
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_string = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Initialize ChatOpenAI
    chat = ChatOpenAI(model="gpt-4o")

    # Tailored prompt for MultiVector retrieval
    prompt = """
    Analyze this image and provide a detailed description optimized for a MultiVector retrieval system. Your description should:

    1. Start with a brief overall summary of the image (1-2 sentences).
    2. Describe key visual elements, including:
       - Main subjects or objects
       - Colors, shapes, and textures
       - Spatial relationships and composition
       - Any text or numbers visible in the image
    3. Mention any notable actions, emotions, or atmosphere conveyed.
    4. Include relevant contextual information (e.g., setting, time period, style).
    5. Use specific, descriptive language that could match potential search queries.
    6. Aim for a length of 100-150 words to balance detail and conciseness.

    Format the description as a continuous paragraph without bullet points or numbering.
    """

    # Generate summary
    response = chat.invoke(
        [
            HumanMessage(
                content=[
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{base64_string}"},
                    },
                ]
            )
        ]
    )
    
    return response.content, base64_string


def ingest_docs(s3_uri):
    try:

        # Vector store for text chunks and image descriptions
        vectorstore = Chroma(
            collection_name=CHROMA_DOCS_INDEX_NAME, 
            embedding_function=get_embeddings_model(),  
            persist_directory=VECTOR_CHROMA_DB_PATH
        )

        # Parent document store for images
        cs = SQLStore(PARENT_DOC_DB_PATH, "docstore")
        docstore = create_kv_docstore(cs)
        
        # Initialize MultiVectorRetriever
        # retriever = MultiVectorRetriever(
        #     vectorstore=vectorstore,
        #     docstore=docstore,
        #     id_key="doc_id",
        # )

        # Initialize record manager for indexing texts
        text_record_manager = SQLRecordManager(
            f"chroma/{CHROMA_DOCS_INDEX_NAME}", db_url=TEXT_RECORD_MANAGER_DB_URL
        )
        text_record_manager.create_schema()

        # Initialize record manager for indexing images
        image_record_manager = SQLRecordManager(
            f"chroma/{CHROMA_DOCS_INDEX_NAME}", db_url=IMAGE_RECORD_MANAGER_DB_URL
        )
        image_record_manager.create_schema()

        result = read_from_s3(s3_uri)
        if not result:
            raise HTTPException(status_code=404, detail="No documents found in the specified S3 path.")
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

        documents_to_index = []
        images_to_index = []

        for doc in result:
            if 'image' in doc.metadata:
                # For images
                doc_id = str(uuid.uuid4())

                 # Create a detailed description of the image
                image_description, base64_image = describe_image(doc.metadata['image'])
                 # Prepare document for vectorstore
                vectorstore_doc = Document(
                    page_content=image_description,
                    metadata={
                        "doc_id": doc_id,
                        "source": doc.metadata['source'],
                        "is_image": True,
                        "s3_uri":doc.metadata["s3_uri"] 
                    }
                )

                # Prepare document for docstore
                docstore_doc = Document(
                    page_content=base64_image,  # Empty content as we're storing the image in metadata
                    metadata={
                        "doc_id": doc_id,
                        "source": doc.metadata['source'],
                        "image_path": doc.metadata['image_path'],
                        "is_image": True
                    }
                )
                images_to_index.append(vectorstore_doc)
                docstore.mset(list(zip(doc_id, [docstore_doc])))
                # Add image to docstore

                pass
            else:
                # For text documents
                chunked_docs = text_splitter.split_documents([doc])
                for chunk in chunked_docs:
                    chunk.metadata["is_image"] = False
                    documents_to_index.append(chunk)

        # Perform indexing for text docs
        doc_indexing_stats = index(
            documents_to_index,
            text_record_manager,
            vectorstore,
            cleanup=None,
            source_id_key="source"
        )
        print("Docs:",doc_indexing_stats)

        # Perform indexing for text docs
        image_indexing_stats = index(
            images_to_index,
            image_record_manager,
            vectorstore,
            cleanup="incremental",
            source_id_key="source"
        )
        print("Images:",image_indexing_stats)
        
        return len(result)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException as he:
        raise he  # Re-raise HTTPExceptions directly
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    ingest_docs(r"s3://lcro-ml-trainings/sources/Sample Business Report.pdf")
    # ingest_docs(r's3://lcro-ml-trainings/sources/cpap-flowchart.png')
    # ingest_docs(r's3://lcro-ml-trainings/sources/Clinical-Trial-Phases-Landscape.jpg')