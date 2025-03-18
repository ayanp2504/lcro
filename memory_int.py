import sqlite3, json, os
import asyncio
from datetime import datetime, timezone
from typing import Any, Iterable, Optional
import asyncio
import concurrent.futures as cf
import functools
import logging
from collections import defaultdict
from datetime import datetime, timezone
from importlib import util
from typing import Any, Iterable, Optional
from langchain_core.embeddings import Embeddings
from urllib.parse import urlparse

from .constants import MEMORY_DB_PATH
from .embeddings import get_embeddings_model

db_path = MEMORY_DB_PATH

from langgraph.store.base import (
    BaseStore,
    GetOp,
    IndexConfig,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    ensure_embeddings,
    get_text_at_path,
    tokenize_path,
)

logger = logging.getLogger(__name__)

@functools.lru_cache(maxsize=1)
def _check_numpy() -> bool:
    if bool(util.find_spec("numpy")):
        return True
    logger.warning(
        "NumPy not found in the current Python environment. "
        "The InMemoryStore will use a pure Python implementation for vector operations, "
        "which may significantly impact performance, especially for large datasets or frequent searches. "
        "For optimal speed and efficiency, consider installing NumPy: "
        "pip install numpy"
    )
    return False

def _cosine_similarity(X: list[float], Y: list[list[float]]) -> list[float]:
    """
    Compute cosine similarity between a vector X and a matrix Y.
    Lazy import numpy for efficiency.
    """
    if not Y:
        return []
    if _check_numpy():
        import numpy as np  # type: ignore

        X_arr = np.array(X) if not isinstance(X, np.ndarray) else X
        Y_arr = np.array(Y) if not isinstance(Y, np.ndarray) else Y
        X_norm = np.linalg.norm(X_arr)
        Y_norm = np.linalg.norm(Y_arr, axis=1)

        # Avoid division by zero
        mask = Y_norm != 0
        similarities = np.zeros_like(Y_norm)
        similarities[mask] = np.dot(Y_arr[mask], X_arr) / (Y_norm[mask] * X_norm)
        return similarities.tolist()

    similarities = []
    for y in Y:
        dot_product = sum(a * b for a, b in zip(X, y))
        norm1 = sum(a * a for a in X) ** 0.5
        norm2 = sum(a * a for a in y) ** 0.5
        similarity = dot_product / (norm1 * norm2) if norm1 > 0 and norm2 > 0 else 0.0
        similarities.append(similarity)

    return similarities

class SQLiteStore(BaseStore):
    """
    SQLite-backed store with optional vector search.
    """

    def __init__(self, db_path: str, *, index: Optional[IndexConfig] = None) -> None:
        # Handle SQLAlchemy-style URI
        if db_path.startswith("sqlite:///"):
            parsed = urlparse(db_path)
            db_path = os.path.abspath(parsed.path.lstrip("/"))

        # Create directories if necessary
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

        # Initialize attributes
        self.db_path = db_path
        self.index_config = index
        self.embeddings = ensure_embeddings(index["embed"]) if index else None

        # Initialize database
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database with required tables."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS items (
                    namespace TEXT,
                    key TEXT,
                    value TEXT,
                    created_at TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (namespace, key)
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS vectors (
                    namespace TEXT,
                    key TEXT,
                    path TEXT,
                    vector BLOB,
                    PRIMARY KEY (namespace, key, path),
                    FOREIGN KEY (namespace, key) REFERENCES items (namespace, key)
                )
            """)
            conn.commit()

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute batch operations synchronously."""
        results = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            for op in ops:
                if isinstance(op, GetOp):
                    results.append(self._get(cursor, op.namespace, op.key))
                elif isinstance(op, PutOp):
                    self._put(cursor, op)
                    results.append(None)
                elif isinstance(op, SearchOp):
                    results.append(self._search(cursor, op))
                elif isinstance(op, ListNamespacesOp):
                    results.append(self._list_namespaces(cursor, op))
                else:
                    raise ValueError(f"Unknown operation type: {type(op)}")
            conn.commit()
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        """Execute batch operations asynchronously."""
        return await asyncio.to_thread(self.batch, ops)

    def _get(self, cursor, namespace: tuple[str, ...], key: str) -> Optional[Item]:
        cursor.execute(
            "SELECT value, created_at, updated_at FROM items WHERE namespace = ? AND key = ?",
            (".".join(namespace), key),
        )
        row = cursor.fetchone()
        if row:
            value, created_at, updated_at = row
            value_as_dict = json.loads(value)  # Deserialize JSON to a Python dictionary
            return Item(
                namespace=namespace,
                key=key,
                value=value_as_dict,
                created_at=datetime.fromisoformat(created_at) if created_at else None,
                updated_at=datetime.fromisoformat(updated_at) if updated_at else None,
            )
        return None


    def _put(self, cursor, op: PutOp):
        namespace = ".".join(op.namespace)
        if op.value is None:
            cursor.execute("DELETE FROM items WHERE namespace = ? AND key = ?", (namespace, op.key))
            cursor.execute("DELETE FROM vectors WHERE namespace = ? AND key = ?", (namespace, op.key))
        else:
            now = datetime.now(timezone.utc).isoformat()
            value_as_json = json.dumps(op.value)  # Serialize value to JSON
            cursor.execute(
                """
                INSERT INTO items (namespace, key, value, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(namespace, key) DO UPDATE SET
                value = excluded.value, updated_at = excluded.updated_at
                """,
                (namespace, op.key, value_as_json, now, now),
            )
            # Handle vector storage
            if op.index is not False and self.index_config and self.embeddings:
                paths = self.index_config.get("fields", ["$"])
                for path in paths:
                    texts = get_text_at_path(op.value, path)
                    embeddings = self.embeddings.embed_documents(texts)
                    for text, vector in zip(texts, embeddings):
                        cursor.execute(
                            """
                            INSERT INTO vectors (namespace, key, path, vector)
                            VALUES (?, ?, ?, ?)
                            ON CONFLICT(namespace, key, path) DO UPDATE SET
                            vector = excluded.vector
                            """,
                            (namespace, op.key, path, sqlite3.Binary(vector)),
                        )

    def _search(self, cursor, op: SearchOp) -> list[SearchItem]:
        namespace_prefix = ".".join(op.namespace_prefix)
        cursor.execute(
            "SELECT namespace, key, value FROM items WHERE namespace LIKE ?",
            (f"{namespace_prefix}%",),
        )
        items = cursor.fetchall()
        if not self.embeddings or not op.query:
            return [
                SearchItem(
                    namespace=tuple(row[0].split(".")),
                    key=row[1],
                    value=row[2],
                    created_at=None,
                    updated_at=None,
                )
                for row in items
            ]

        # Perform similarity search
        query_vector = self.embeddings.embed_query(op.query)
        candidates = []
        for row in items:
            namespace, key, value = row
            cursor.execute(
                "SELECT vector FROM vectors WHERE namespace = ? AND key = ?",
                (namespace, key),
            )
            vectors = [sqlite3.loads(v[0]) for v in cursor.fetchall()]
            candidates.append((row, vectors))

        results = self._perform_similarity_search(query_vector, candidates, op)
        return results

    def _perform_similarity_search(self, query_vector, candidates, op):
        flat_items, flat_vectors = [], []
        for item, vectors in candidates:
            for vector in vectors:
                flat_items.append(item)
                flat_vectors.append(vector)

        scores = _cosine_similarity(query_vector, flat_vectors)
        sorted_results = sorted(
            zip(scores, flat_items), key=lambda x: x[0], reverse=True
        )

        return [
            SearchItem(
                namespace=tuple(item[0].split(".")),
                key=item[1],
                value=item[2],
                score=float(score),
            )
            for score, item in sorted_results[: op.limit]
        ]

    def _list_namespaces(self, cursor, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        cursor.execute("SELECT DISTINCT namespace FROM items")
        all_namespaces = [tuple(row[0].split(".")) for row in cursor.fetchall()]
        return sorted(ns[: op.max_depth] for ns in all_namespaces)


async def main():
    from dotenv import load_dotenv
    load_dotenv()

    # Initialize the SQLite store
    store = SQLiteStore(
        db_path=MEMORY_DB_PATH,
        index=IndexConfig(
            dims=1536,
            embed=get_embeddings_model(),
            fields=["text"],
        ),
    )

    # Define a namespace and key
    namespace = ("user_12345", "user_states")
    key = "memory_events"

    print("Testing with aget, aput, and asearch...")

    # Step 1: Add an item using aput
    await store.aput(namespace, key, {"state": "active"})
    print("Added initial memory state.")

    # Step 2: Retrieve the item using aget
    item = await store.aget(namespace, key)
    print("Fetched memory state:", item.value if item else "None")

    # Step 3: Update the item using aput
    await store.aput(namespace, key, {"state": "inactive"})
    print("Updated memory state.")

    # Step 4: Retrieve the updated item
    updated_item = await store.aget(namespace, key)
    print("Fetched updated memory state:", updated_item.value if updated_item else "None")

    # Step 5: Perform a similarity search using asearch
    search_results = await store.asearch(
        ("user_12345",), query="active", filter=None, limit=2, offset=0
    )
    print("Search results for 'active':", [result.value for result in search_results])

# Run the test
if __name__ == "__main__":
    asyncio.run(main())
    