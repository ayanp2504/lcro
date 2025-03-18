from __future__ import annotations
from dataclasses import dataclass, field, fields
from typing import Annotated, Any, Literal, Optional, Type, TypeVar
from langchain_core.runnables import RunnableConfig, ensure_config

# Mapping model names to their respective identifiers
MODEL_NAME_TO_RESPONSE_MODEL = {
    "openai_gpt-4o": "openai/gpt-4o"
}

@dataclass(kw_only=True)
class BaseConfiguration:
    """Configuration class for indexing and retrieval operations."""

    embedding_model: Annotated[
        str,
        {"__template_metadata__": {"kind": "embeddings"}},
    ] = field(
        default="openai/text-embedding-3-small",
        metadata={
            "description": "Name of the embedding model to use."
        },
    )

    retriever_provider: Annotated[
        Literal["chroma"],
        {"__template_metadata__": {"kind": "retriever"}},
    ] = field(
        default="chroma",
        metadata={"description": "The vector store provider to use for retrieval."},
    )

    search_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "description": "Additional keyword arguments for the search function."
        },
    )

    @classmethod
    def from_runnable_config(
        cls: Type[T], config: Optional[RunnableConfig] = None
    ) -> T:
        """Create an IndexConfiguration instance from a RunnableConfig object."""
        config = ensure_config(config)
        configurable = config.get("configurable") or {}
        
        # Filter only valid fields for initialization
        _fields = {f.name for f in fields(cls) if f.init}
        # print("Fiels:", _fields)
        
        return cls(**{k: v for k, v in configurable.items() if k in _fields})

T = TypeVar("T", bound=BaseConfiguration)
