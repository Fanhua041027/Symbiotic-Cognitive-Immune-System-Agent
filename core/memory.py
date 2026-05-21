"""免疫记忆系统 - 基于 ChromaDB 的抗体存储与检索。"""

import os
import uuid
from typing import Optional, Dict

from core.logger import setup_logger

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

    def store_antibody(self, error_pattern: str, antibody_code: str, context: str) -> None:
        self._antibodies.append({
            "error_pattern": error_pattern,
            "code": antibody_code,
            "context": context,
        })
        logger.info(
            "Stored antibody (in-memory) for pattern: %s...", error_pattern[:50]
        )

    def search_antibody(self, query: str) -> Optional[Dict[str, str]]:
        if not self._antibodies:
            return None
        # Simple keyword matching fallback
        query_lower = query.lower()
        best = None
        for ab in self._antibodies:
            if query_lower in ab["context"].lower() or \
               query_lower in ab["error_pattern"].lower():
                best = ab
        if best:
            return {"code": best["code"], "pattern": best["error_pattern"]}
        return None

    def count(self) -> int:
        return len(self._antibodies)


class ImmunologyMemory:
    """管理免疫记忆：存储和检索历史抗体（补丁）。"""

    def __init__(self, collection_name: str = "antibodies"):
        if HAS_CHROMADB:
            try:
                os.makedirs(DB_DIR, exist_ok=True)
                self._backend = "chromadb"
                self.client = chromadb.PersistentClient(
                    path=DB_DIR, settings=ChromaSettings(anonymized_telemetry=False)
                )
                self.collection = self.client.get_or_create_collection(collection_name)
                logger.info("Immune memory initialized (chromadb, path=%s)", DB_DIR)
                return
            except Exception as e:
                logger.warning("chromadb init failed (%s), falling back to in-memory", e)

        self._backend = "memory"
        self._in_memory = InMemoryStore()
        logger.info("Immune memory initialized (in-memory fallback)")

    def store_antibody(
        self, error_pattern: str, antibody_code: str, context: str
    ) -> None:
        """存储有效的抗体到向量数据库。"""
        antibody_id = str(uuid.uuid4())
        if self._backend == "chromadb":
            self.collection.add(
                documents=[context],
                ids=[antibody_id],
                metadatas=[{"error_pattern": error_pattern, "code": antibody_code}],
            )
        else:
            self._in_memory.store_antibody(error_pattern, antibody_code, context)
        logger.info(
            "Stored new antibody for pattern: %s...", error_pattern[:50]
        )

    def list_antibodies(self, limit: int = 50) -> list[dict]:
        """List stored antibodies with metadata."""
        if self._backend == "chromadb":
            try:
                data = self.collection.get(limit=limit)
                results = []
                for i, doc_id in enumerate(data.get("ids", [])):
                    meta = (data.get("metadatas") or [{}])[i] if i < len(data.get("metadatas") or []) else {}
                    doc = (data.get("documents") or [None])[i] if i < len(data.get("documents") or []) else None
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
        return [
            {
                "id": str(i),
                "error_pattern": ab.get("error_pattern", "unknown"),
                "code": ab.get("code", "")[:200],
                "context": ab.get("context", "")[:200],
            }
            for i, ab in enumerate(self._in_memory._antibodies)
        ][:limit]

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
        try:
            idx = int(antibody_id)
            if 0 <= idx < len(self._in_memory._antibodies):
                del self._in_memory._antibodies[idx]
                return True
        except (ValueError, IndexError):
            pass
        return False

    def clear_all(self) -> int:
        """Delete all antibodies. Returns count of deleted items."""
        count = self.count()
        if self._backend == "chromadb":
            try:
                data = self.collection.get()
                if data and data.get("ids"):
                    self.collection.delete(ids=data["ids"])
            except Exception as e:
                logger.warning("Failed to clear chromadb: %s", e)
        else:
            self._in_memory._antibodies.clear()
        logger.info("Cleared %d antibodies", count)
        return count

    def search_antibody(self, query: str) -> Optional[Dict[str, str]]:
        """检索相似的历史错误及对应抗体。"""
        if self._backend == "chromadb":
            try:
                results = self.collection.query(query_texts=[query], n_results=1)
                if results["ids"] and results["ids"][0]:
                    return {
                        "code": results["metadatas"][0][0]["code"],
                        "pattern": results["metadatas"][0][0]["error_pattern"],
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
