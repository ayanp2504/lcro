import os
from contextlib import contextmanager
from typing import Iterator

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnableConfig
from backend_lcro.configuration import BaseConfiguration
from backend_lcro.constants import CHROMA_DOCS_INDEX_NAME, VECTOR_CHROMA_DB_PATH


def make_text_encoder(model: str) -> Embeddings:
    """Connect to the configured text encoder."""
    provider, model = model.split("/", maxsplit=1)
    match provider:
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            return OpenAIEmbeddings(model=model)
        case _:
            raise ValueError(f"Unsupported embedding provider: {provider}")


@contextmanager
def make_chroma_retriever(
    configuration: BaseConfiguration, embedding_model: Embeddings
) -> Iterator[BaseRetriever]:
    
    # Initialize Chroma vector store
    vectorstore = Chroma(
        collection_name=CHROMA_DOCS_INDEX_NAME,  # Specify your collection name
        embedding_function=embedding_model,  # Use the provided embedding model
        persist_directory=VECTOR_CHROMA_DB_PATH  # Path to store the Chroma DB
    )

    # Create a retriever from the vector store
    search_kwargs = {**configuration.search_kwargs}
    retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)

    yield retriever  # Yield the retriever for use


@contextmanager
def make_retriever(
    config: RunnableConfig,
) -> Iterator[BaseRetriever]:
    """Create a retriever for the agent, based on the current configuration."""
    configuration = BaseConfiguration.from_runnable_config(config)
    embedding_model = make_text_encoder(configuration.embedding_model)
    
    match configuration.retriever_provider:
        case "chroma":
            with make_chroma_retriever(configuration, embedding_model) as retriever:
                yield retriever

        case _:
            raise ValueError(
                "Unrecognized retriever_provider in configuration. "
                f"Expected one of: {', '.join(BaseConfiguration.__annotations__['retriever_provider'].__args__)}\n"
                f"Got: {configuration.retriever_provider}"
            )