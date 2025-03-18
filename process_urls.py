from langchain_community.document_loaders import UnstructuredURLLoader
from .constants import (TEXT_RECORD_MANAGER_DB_URL, VECTOR_CHROMA_DB_PATH, CHROMA_DOCS_INDEX_NAME, PARENT_DOC_DB_PATH, IMAGE_RECORD_MANAGER_DB_URL)
from langchain_chroma import Chroma
from backend_lcro.embeddings import get_embeddings_model
from .kvstore import SQLStore
from langchain.storage._lc_store import create_kv_docstore
from langchain.indexes import SQLRecordManager, index
from fastapi import HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
import logging, time
from markdownify import markdownify as md
from langchain.docstore.document import Document
import datetime 
def create_undetected_chrome_driver(driver_path):
    """
    Create an undetected Chrome driver with specified options.

    Parameters:
    - driver_path (str): The path to the Chrome driver executable.
    - user_agent (str): The user agent to use.

    Returns:
    - driver: Undetected Chrome driver instance.
    """
    try:
        # Configure Chrome options
        options = webdriver.ChromeOptions()
        # Disable the pop-up blocker
        # options.add_argument("--disable-popup-blocking")
        # options.add_argument('--headless')  # Run headless mode
        options.add_argument('--log-level=3')  # Disable logging
        # Specify the service parameter if required
        service = webdriver.chrome.service.Service(driver_path)
        # Create the Chrome driver instance
        driver = webdriver.Chrome(options=options, service=service)
        # Maximize the browser window
        driver.maximize_window()
        # Log information about successful driver creation
        logging.info("Chrome driver created successfully")
        # Return the Chrome driver instance
        return driver
    except FileNotFoundError:
        logging.error(f"File not found: {driver_path}")
        raise FileNotFoundError(f"File not found: {driver_path}")
    except WebDriverException as e:
        logging.error(f"WebDriverException occurred: {str(e)}")
        raise
    except Exception as e:
        logging.error(f"Error creating undetected Chrome driver: {str(e)}")
        raise

def scroll_to_end(driver):
    """
    Scroll to the end of the page.

    Args:
        driver: Selenium WebDriver instance.

    Returns:
        None
    """
    try:
        # Get current page height
        page_height = driver.execute_script("return document.body.scrollHeight")

        while True:
            # Scroll to the bottom of the page
            driver.find_element(By.TAG_NAME, 'body').send_keys(Keys.END)
            time.sleep(3)  # Adjust the sleep duration as needed

            # Get new page height after scrolling
            new_page_height = driver.execute_script("return document.body.scrollHeight")

            # Break the loop if no more scrolling is possible
            if new_page_height == page_height:
                break

            # Update page height for the next iteration
            page_height = new_page_height

    except Exception as e:
        # Log an error message if an exception occurs during scrolling
        logging.error(f"Error in scroll_to_end function: {str(e)}", exc_info=True)

def ingest_urls_old(urls):
    print("urls Received:",urls)
    try:
        # Vector store for text chunks and image descriptions
        vectorstore = Chroma(
            collection_name=CHROMA_DOCS_INDEX_NAME,
            embedding_function=get_embeddings_model(),
            persist_directory=VECTOR_CHROMA_DB_PATH,
        )

        # Parent document store for images
        cs = SQLStore(PARENT_DOC_DB_PATH, "docstore")
        docstore = create_kv_docstore(cs)
        # Initialize record manager for indexing texts
        text_record_manager = SQLRecordManager(
            f"chroma/{CHROMA_DOCS_INDEX_NAME}", db_url=TEXT_RECORD_MANAGER_DB_URL
        )
        text_record_manager.create_schema()

        loader = UnstructuredURLLoader(urls=urls)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        chunked_docs = text_splitter.split_documents(data)
        # Update metadata for each document
        for i, doc in enumerate(chunked_docs):
            # Set the source and s3_uri metadata
            doc.metadata['s3_uri'] = doc.metadata['source']  # Set s3_uri to the same value as source
        # Perform indexing for text docs
        doc_indexing_stats = index(
            chunked_docs,
            text_record_manager,
            vectorstore,
            cleanup=None,
            source_id_key="source",
        )
        print("Docs:", doc_indexing_stats)
        return doc_indexing_stats
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException as he:
        raise he  # Re-raise HTTPExceptions directly
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Internal server error: {str(e)}"
        )
    
def ingest_urls(urls):
    """
    Ingest a list of URLs, scrape their content, split it into chunks, 
    and index the content into a vector store. Continues processing 
    even if some URLs fail and returns the unsuccessful URLs.

    Args:
        urls (list): List of URLs to be ingested.

    Returns:
        dict: A dictionary containing:
              - `status`: Overall status message.
              - `successful_urls`: List of successfully processed URLs.
              - `failed_urls`: List of URLs that failed during processing.

    Raises:
        HTTPException: For server-side errors.
    """
    logging.info("Received URLs for ingestion: %s", urls)
    driver = None
    successful_urls = []
    failed_urls = []

    try:
        # Initialize the vector store for text chunks and image descriptions
        vectorstore = Chroma(
            collection_name=CHROMA_DOCS_INDEX_NAME,
            embedding_function=get_embeddings_model(),
            persist_directory=VECTOR_CHROMA_DB_PATH,
        )
        logging.info("Initialized vector store.")

        # Initialize parent document store for images
        cs = SQLStore(PARENT_DOC_DB_PATH, "docstore")
        docstore = create_kv_docstore(cs)
        logging.info("Initialized parent document store.")

        # Initialize record manager for indexing texts
        text_record_manager = SQLRecordManager(
            f"chroma/{CHROMA_DOCS_INDEX_NAME}", db_url=TEXT_RECORD_MANAGER_DB_URL
        )
        text_record_manager.create_schema()
        logging.info("Text record manager schema created.")

        # Set up the web driver
        driver = create_undetected_chrome_driver(r"C:\Users\Administrator\Desktop\chromedriver.exe")

        # Process each URL
        for url in urls:
            try:
                logging.info("Processing URL: %s", url)
                driver.get(url)
                time.sleep(2)
                scroll_to_end(driver)

                # Extract and process page content
                page_source = driver.page_source
                markdown_content = md(page_source)  # Convert HTML to Markdown
                logging.info("Converted page content to Markdown.")
                metadata={
                    "filename": url,
                    "s3_uri": url,
                    "doc_type": "url_text",
                    "ingest_timestamp": datetime.utcnow().isoformat() + "Z",
                }
                # Wrap the Markdown content in a Document object
                document = Document(page_content=markdown_content, metadata=metadata)

                # Split content into manageable chunks
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunked_docs = text_splitter.split_documents([document])

                # # Update metadata and prepare for indexing
                # for doc in chunked_docs:
                #     doc.metadata["s3_uri"] = doc.metadata.get("source", "")  # Set s3_uri metadata
                # logging.info("Metadata updated for chunked documents.")

                # Index the chunked documents
                index(
                    chunked_docs,
                    text_record_manager,
                    vectorstore,
                    cleanup=None,
                    source_id_key="source",
                )
                logging.info("Indexing completed for URL: %s", url)

                # Mark the URL as successful
                successful_urls.append(url)

            except Exception as url_error:
                logging.error("Failed to process URL: %s. Error: %s", url, str(url_error))
                failed_urls.append(url)

        # Prepare the response
        if failed_urls:
            status = "Completed with errors. Some URLs failed to process."
        else:
            status = "All URLs processed successfully."

        return {
            "status": status,
            "successful_urls": successful_urls,
            "failed_urls": failed_urls,
        }

    except Exception as e:
        logging.error("Unexpected error occurred: %s", str(e), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    finally:
        if driver:
            driver.quit()
            logging.info("Web driver closed.")

    
if __name__ == "__main__":
    ingest_urls([])
