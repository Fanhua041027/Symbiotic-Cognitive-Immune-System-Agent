"""免疫记忆系统 - 基于 ChromaDB 的抗体存储与检索。"""

import os
import uuid
from typing import Dict, Optional

from core.logger import setup_logger

# Attempt local ONNX embedding before ChromaDB default
try:
    from core.embeddings import LocalOnnxEmbeddingFunction
    HAS_LOCAL_EMBEDDING = True
except ImportError:
    HAS_LOCAL_EMBEDDING = False

logger = setup_logger("immune_db")

DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".immune_db")

# Attempt to import chromadb, with graceful fallback
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False
    logger.warning(
        "chromadb not installed. Immune memory will use in-memory fallback."
        " Install with: pip install chromadb"
    )


class InMemoryStore:
    """In-memory fallback when chromadb is not available."""

    def __init__(self):
        self._antibodies: list[dict] = []

    def store_antibody(
        self, error_pattern: str, antibody_code: str, context: str
    ) -> bool:
        """Store antibody if no similar one exists. Returns True if stored."""
        # Dedup: check if a similar antibody already exists
        if self._find_similar(error_pattern, antibody_code):
            logger.debug(
                "Skipping duplicate antibody for pattern: %s...", error_pattern[:40]
            )
            return False
        self._antibodies.append({
            "error_pattern": error_pattern,
            "code": antibody_code,
            "context": context,
        })
        logger.info(
            "Stored antibody (in-memory) for pattern: %s...", error_pattern[:50]
        )
        return True

    def _find_similar(self, error_pattern: str, antibody_code: str) -> bool:
        """Check if a similar antibody already exists (Jaccard overlap >= 0.7)."""
        if not self._antibodies:
            return False
        combined = (error_pattern + " " + antibody_code).lower()
        for ab in self._antibodies:
            existing = (ab.get("error_pattern", "") + " " + ab.get("code", "")).lower()
            if ImmunologyMemory._token_similarity(combined, existing) >= 0.7:
                return True
        return False

    def search_antibody(self, query: str) -> Optional[Dict[str, str]]:
        if not self._antibodies:
            return None

        # Token overlap scoring across error_pattern, context, and code
        query_tokens = set(query.lower().split())
        if not query_tokens:
            return None

        scored = []
        for ab in self._antibodies:
            texts = [
                ab.get("error_pattern", "").lower(),
                ab.get("context", "").lower(),
                ab.get("code", "").lower(),
            ]
            # Max token overlap ratio across all fields
            score = max(
                sum(1 for t in query_tokens if t in text) / len(query_tokens)
                for text in texts
            ) if any(texts) else 0.0
            if score > 0:
                scored.append((score, ab))

        if not scored:
            return None

        # Return best match
        scored.sort(key=lambda x: -x[0])
        best = scored[0][1]
        logger.debug("In-memory search: best score=%.2f for pattern=%s",
                      scored[0][0], best.get("error_pattern", "?")[:40])
        return {"code": best["code"], "pattern": best["error_pattern"]}

    def list_antibodies(self, limit: int = 50) -> list[dict]:
        return [
            {
                "id": str(i),
                "error_pattern": ab.get("error_pattern", "unknown"),
                "code": ab.get("code", "")[:200],
                "context": ab.get("context", "")[:200],
            }
            for i, ab in enumerate(self._antibodies)
        ][:limit]

    def delete_antibody(self, antibody_id: str) -> bool:
        try:
            idx = int(antibody_id)
            if 0 <= idx < len(self._antibodies):
                del self._antibodies[idx]
                return True
        except (ValueError, IndexError):
            pass
        return False

    def clear_all(self) -> int:
        count = len(self._antibodies)
        self._antibodies.clear()
        return count

    def count(self) -> int:
        return len(self._antibodies)


class ImmunologyMemory:
    """管理免疫记忆：存储和检索历史抗体（补丁）。"""

    def __init__(self, collection_name: str = "antibodies"):
        if HAS_CHROMADB:
            try:
                embedding_fcn = None
                if HAS_LOCAL_EMBEDDING:
                    try:
                        embedding_fcn = LocalOnnxEmbeddingFunction()
                        logger.info("Using local ONNX embedding function")
                    except Exception as e:
                        logger.warning(
                            "Local ONNX embedding unavailable (%s), using chromadb default", e
                        )

                os.makedirs(DB_DIR, exist_ok=True)
                self._backend = "chromadb"
                self.client = chromadb.PersistentClient(
                    path=DB_DIR, settings=ChromaSettings(anonymized_telemetry=False)
                )
                coll_kwargs: dict = {}
                if embedding_fcn is not None:
                    coll_kwargs["embedding_function"] = embedding_fcn
                self.collection = self.client.get_or_create_collection(
                    collection_name, **coll_kwargs
                )
                logger.info("Immune memory initialized (chromadb, path=%s)", DB_DIR)
                return
            except Exception as e:
                logger.warning("chromadb init failed (%s), falling back to in-memory", e)

        self._backend = "memory"
        self._in_memory = InMemoryStore()
        logger.info("Immune memory initialized (in-memory fallback)")

    def store_antibody(
        self, error_pattern: str, antibody_code: str, context: str
    ) -> bool:
        """存储有效的抗体到向量数据库。Returns False if duplicate was skipped."""
        antibody_id = str(uuid.uuid4())
        if self._backend == "chromadb":
            # Dedup: skip if similar antibody already exists in collection
            try:
                results = self.collection.query(
                    query_texts=[context + " " + error_pattern],
                    n_results=1,
                )
                if results["ids"] and results["ids"][0]:
                    existing_meta = (results.get("metadatas") or [[{}]])[0][0]
                    existing_code = existing_meta.get("code", "")
                    similarity = self._token_similarity(antibody_code, existing_code)
                    if existing_code and similarity > 0.7:
                        logger.debug(
                            "Skipping duplicate antibody (chromadb): %s...",
                            error_pattern[:40],
                        )
                        return False
            except Exception:
                pass  # Proceed with store on query failure

            self.collection.add(
                documents=[context],
                ids=[antibody_id],
                metadatas=[{"error_pattern": error_pattern, "code": antibody_code}],
            )
        else:
            stored = self._in_memory.store_antibody(error_pattern, antibody_code, context)
            if not stored:
                return False

        logger.info(
            "Stored new antibody for pattern: %s...", error_pattern[:50]
        )
        return True

    @staticmethod
    def _token_similarity(a: str, b: str) -> float:
        """Jaccard-like token overlap between two strings."""
        tokens_a = set(a.lower().split())
        tokens_b = set(b.lower().split())
        if not tokens_a or not tokens_b:
            return 0.0
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        return len(intersection) / len(union)

    def list_antibodies(self, limit: int = 50) -> list[dict]:
        """List stored antibodies with metadata."""
        if self._backend == "chromadb":
            try:
                data = self.collection.get(limit=limit)
                results = []
                for i, doc_id in enumerate(data.get("ids", [])):
                    meta_list = data.get("metadatas") or []
                    meta = meta_list[i] if i < len(meta_list) else {}
                    doc_list = data.get("documents") or []
                    doc = doc_list[i] if i < len(doc_list) else None
                    results.append({
                        "id": doc_id,
                        "error_pattern": meta.get("error_pattern", "unknown"),
                        "code": meta.get("code", "")[:200],
                        "context": (doc or "")[:200],
                    })
                return results
            except Exception as e:
                logger.warning("Failed to list chromadb antibodies: %s", e)
                return []
        # In-memory fallback
        return self._in_memory.list_antibodies(limit=limit)

    def delete_antibody(self, antibody_id: str) -> bool:
        """Delete an antibody by ID. Returns True on success."""
        if self._backend == "chromadb":
            try:
                self.collection.delete(ids=[antibody_id])
                logger.info("Deleted antibody: %s", antibody_id)
                return True
            except Exception as e:
                logger.warning("Failed to delete antibody %s: %s", antibody_id, e)
                return False
        # In-memory: delete by index
        return self._in_memory.delete_antibody(antibody_id)

    def clear_all(self) -> int:
        """Delete all antibodies. Returns count of deleted items."""
        if self._backend == "chromadb":
            count = self.count()
            try:
                data = self.collection.get()
                if data and data.get("ids"):
                    self.collection.delete(ids=data["ids"])
            except Exception as e:
                logger.warning("Failed to clear chromadb: %s", e)
                count = 0
        else:
            count = self._in_memory.clear_all()
        logger.info("Cleared %d antibodies", count)
        return count

    def search_antibody(self, query: str) -> Optional[Dict[str, str]]:
        """检索相似的历史错误及对应抗体。"""
        if self._backend == "chromadb":
            try:
                results = self.collection.query(query_texts=[query], n_results=1)
                if results["ids"] and results["ids"][0]:
                    metas = results.get("metadatas", [])
                    if metas and metas[0]:
                        meta = metas[0][0]
                        return {
                            "code": meta.get("code", ""),
                            "pattern": meta.get("error_pattern", ""),
                        }
            except Exception as e:
                logger.debug("chromadb query failed: %s", e)
            return None
        return self._in_memory.search_antibody(query)

    def count(self) -> int:
        """Return the number of stored antibodies."""
        if self._backend == "chromadb":
            try:
                return self.collection.count()
            except Exception:
                return 0
        return self._in_memory.count()


# 全局单例
memory_db = ImmunologyMemory()
