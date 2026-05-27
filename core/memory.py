"""免疫记忆系统 - 基于 ChromaDB 的抗体存储与检索。"""

import json
import os
import time
import uuid
from datetime import datetime, timezone

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
        " Install with: pip install chromadb",
    )


class InMemoryStore:
    """In-memory fallback when chromadb is not available."""

    def __init__(self):
        self._antibodies: list[dict] = []
        self._embedding_fn = None
        # Try to load ONNX embedding for semantic dedup
        if HAS_LOCAL_EMBEDDING:
            try:
                from core.embeddings import LocalOnnxEmbeddingFunction
                self._embedding_fn = LocalOnnxEmbeddingFunction()
                logger.info("InMemoryStore: using ONNX embedding for dedup")
            except Exception:
                pass
        if self._embedding_fn is None:
            logger.info("InMemoryStore: using Jaccard token overlap for dedup")

    def _semantic_similarity(self, a: str, b: str) -> float:
        """Cosine similarity via ONNX embedding, with Jaccard fallback.
        Falls back to Jaccard for short texts (< 30 chars) where embeddings are unreliable."""
        if len(a) < 30 or len(b) < 30:
            return ImmunologyMemory._token_similarity(a, b)
        if self._embedding_fn:
            try:
                vecs = self._embedding_fn([a, b])
                return sum(  # L2 normed → dot = cosine
                    x * y for x, y in zip(vecs[0], vecs[1]))
            except Exception:
                pass
        return ImmunologyMemory._token_similarity(a, b)

    def store_antibody(
        self, error_pattern: str, antibody_code: str, context: str,
    ) -> bool:
        """Store antibody if no similar one exists. Returns True if stored."""
        # Dedup: check if a similar antibody already exists
        if self._find_similar(error_pattern, antibody_code):
            logger.debug(
                "Skipping duplicate antibody for pattern: %s...", error_pattern[:40],
            )
            return False
        now = time.time()
        self._antibodies.append({
            "error_pattern": error_pattern,
            "code": antibody_code,
            "context": context,
            "created_at": now,
            "last_matched": now,
        })
        logger.info(
            "Stored antibody (in-memory) for pattern: %s...", error_pattern[:50],
        )
        return True

    def _find_similar(self, error_pattern: str, antibody_code: str) -> bool:
        """Check if a similar antibody already exists (cosine >= 0.85 or Jaccard >= 0.7)."""  # noqa: E501
        if not self._antibodies:
            return False
        combined = error_pattern + " " + antibody_code
        for ab in self._antibodies:
            existing = ab.get("error_pattern", "") + " " + ab.get("code", "")
            if self._semantic_similarity(combined, existing) >= 0.85:
                return True
        return False

    def search_antibody(self, query: str, max_scan: int = 200) -> dict[str, str] | None:
        """Search antibodies by token overlap. Only scans up to *max_scan* entries."""
        if not self._antibodies:
            return None

        query_tokens = set(query.lower().split())
        if not query_tokens:
            return None

        best_score = 0.0
        best_ab = None
        scanned = 0
        for ab in self._antibodies:
            scanned += 1
            if scanned > max_scan:
                break
            texts = [
                ab.get("error_pattern", "").lower(),
                ab.get("context", "").lower(),
                ab.get("code", "").lower(),
            ]
            score = max(
                sum(1 for t in query_tokens if t in text) / len(query_tokens)
                for text in texts
            ) if any(texts) else 0.0
            if score > best_score:
                best_score = score
                best_ab = ab
                if score >= 0.9:  # high confidence match, stop early
                    break

        if best_ab is None or best_score <= 0:
            return None

        best_ab["last_matched"] = time.time()
        logger.debug("In-memory search: best score=%.2f for pattern=%s",
                      best_score, best_ab.get("error_pattern", "?")[:40])
        return {"code": best_ab["code"], "pattern": best_ab["error_pattern"]}

    @staticmethod
    def _fmt_time(ts: float) -> str:
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds")

    def list_antibodies(self, limit: int = 50) -> list[dict]:
        return [
            {
                "id": str(i),
                "error_pattern": ab.get("error_pattern", "unknown"),
                "code": ab.get("code", "")[:200],
                "context": ab.get("context", "")[:200],
                "created_at": self._fmt_time(ab.get("created_at", 0)),
                "last_matched": self._fmt_time(ab.get("last_matched", 0)),
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

    def decay(self, ttl_days: int = 30, prune_unmatched_days: int = 90) -> int:
        """Remove antibodies past TTL or last_matched threshold. Returns count removed."""
        now = time.time()
        ttl_sec = ttl_days * 86400
        unmatched_sec = prune_unmatched_days * 86400
        before = len(self._antibodies)
        self._antibodies = [
            ab for ab in self._antibodies
            if (now - ab.get("created_at", now)) < ttl_sec
            and (now - ab.get("last_matched", now)) < unmatched_sec
        ]
        removed = before - len(self._antibodies)
        if removed:
            logger.info("InMemoryStore: decay removed %d antibodies", removed)
        return removed


class ImmunologyMemory:
    """管理免疫记忆：存储和检索历史抗体（补丁）。"""

    def __init__(self, collection_name: str = "antibodies"):
        # Always create in-memory fallback for use when chromadb.add fails
        self._in_memory = InMemoryStore()

        if HAS_CHROMADB:
            try:
                embedding_fcn = None
                if HAS_LOCAL_EMBEDDING:
                    try:
                        from core.embeddings import MODEL_PATH, TOKENIZER_PATH
                        if os.path.exists(MODEL_PATH) and os.path.exists(TOKENIZER_PATH):
                            embedding_fcn = LocalOnnxEmbeddingFunction()
                            logger.info("Using local ONNX embedding function")
                        else:
                            logger.info(
                                "ONNX model not found, using chromadb default embedding",
                            )
                    except Exception as e:
                        logger.warning(
                            "Local ONNX unavailable (%s), using chromadb default", e,
                        )

                os.makedirs(DB_DIR, exist_ok=True)
                self._backend = "chromadb"
                self.client = chromadb.PersistentClient(
                    path=DB_DIR, settings=ChromaSettings(anonymized_telemetry=False),
                )
                coll_kwargs: dict = {}
                if embedding_fcn is not None:
                    coll_kwargs["embedding_function"] = embedding_fcn
                self.collection = self.client.get_or_create_collection(
                    collection_name, **coll_kwargs,
                )
                logger.info("Immune memory initialized (chromadb, path=%s)", DB_DIR)
                self._auto_decay()
                return
            except Exception as e:
                logger.warning("chromadb init failed (%s), falling back to in-memory", e)

        self._backend = "memory"
        self._auto_decay()
        logger.info("Immune memory initialized (in-memory fallback)")

    def store_antibody(
        self, error_pattern: str, antibody_code: str, context: str,
    ) -> bool:
        """存储有效的抗体到向量数据库。Returns False if duplicate was skipped."""
        antibody_id = str(uuid.uuid4())
        if self._backend == "chromadb":
            # Dedup: single query for exact match + semantic distance check
            query_text = (context + " " + error_pattern).strip()
            try:
                n = 5 if len(query_text) < 40 else 1
                q = query_text if query_text else error_pattern
                results = self.collection.query(
                    query_texts=[q], n_results=n,
                    include=["distances", "metadatas"],
                )
                if results["ids"] and results["ids"][0]:
                    for i, meta in enumerate(results["metadatas"][0]):
                        if (meta and meta.get("code") == antibody_code
                                and meta.get("error_pattern") == error_pattern):
                            logger.debug("Skipping exact duplicate antibody (chromadb)")
                            return False
                    # Semantic dedup only for sufficiently long texts
                    if len(query_text) >= 40:
                        distance = (results.get("distances") or [[1.0]])[0][0]
                        if distance < 0.5:
                            logger.debug(
                                "Skipping duplicate antibody (chromadb dist=%.3f): %s...",
                                distance, error_pattern[:40])
                            return False
            except Exception:
                pass

            try:
                self.collection.add(
                    documents=[context],
                    ids=[antibody_id],
                    metadatas=[{
                        "error_pattern": error_pattern,
                        "code": antibody_code,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "last_matched": datetime.now(timezone.utc).isoformat(),
                    }],
                )
            except Exception as e:
                logger.warning(
                    "chromadb add failed (%s), falling back to in-memory", e,
                )
                stored = self._in_memory.store_antibody(
                    error_pattern, antibody_code, context)
                if not stored:
                    return False
        else:
            stored = self._in_memory.store_antibody(error_pattern, antibody_code, context)
            if not stored:
                return False

        logger.info(
            "Stored new antibody for pattern: %s...", error_pattern[:50],
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
        """List stored antibodies with metadata including timestamps."""
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
                        "created_at": meta.get("created_at", ""),
                        "last_matched": meta.get("last_matched", ""),
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

    def search_antibody(self, query: str) -> dict[str, str] | None:
        """检索相似的历史错误及对应抗体。Updates last_matched on hit."""
        if self._backend == "chromadb":
            try:
                results = self.collection.query(query_texts=[query], n_results=1)
                if results["ids"] and results["ids"][0]:
                    metas = results.get("metadatas", [])
                    if metas and metas[0]:
                        meta = metas[0][0]
                        # Update last_matched timestamp
                        try:
                            updated_meta = dict(meta)
                            updated_meta["last_matched"] = (
                                datetime.now(timezone.utc).isoformat())
                            self.collection.update(
                                ids=[results["ids"][0][0]],
                                metadatas=[updated_meta],
                            )
                        except Exception as e:
                            logger.debug("Failed to update last_matched: %s", e)
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

    # ------------------------------------------------------------------
    # Antibody export / import (cross-instance immune memory sharing)
    # ------------------------------------------------------------------
    def export_antibodies(self, path: str | None = None) -> str:
        """Export all antibodies to a JSON file. Returns the path."""
        if path is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            export_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "exports",
            )
            os.makedirs(export_dir, exist_ok=True)
            path = os.path.join(export_dir, f"antibodies_{ts}.json")

        antibodies = self.list_antibodies(limit=10000)
        data = {
            "exported_at": datetime.now(timezone.utc).isoformat(),
            "backend": self._backend,
            "count": len(antibodies),
            "antibodies": antibodies,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info("Exported %d antibodies to %s", len(antibodies), path)
        return path

    def import_antibodies(self, path: str) -> int:
        """Import antibodies from a JSON export file. Returns count imported."""
        if not os.path.exists(path):
            logger.warning("Import file not found: %s", path)
            return 0
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Failed to read import file: %s", e)
            return 0

        imported = 0
        for ab in data.get("antibodies", []):
            stored = self.store_antibody(
                error_pattern=ab.get("error_pattern", "imported"),
                antibody_code=ab.get("code", ""),
                context=ab.get("context", ""),
            )
            if stored:
                imported += 1

        logger.info("Imported %d/%d antibodies from %s", imported,
                     len(data.get("antibodies", [])), path)
        return imported

    def _auto_decay(self) -> None:
        """Run decay at startup to prune old antibodies. Silently handles empty stores."""
        try:
            removed = self.decay(ttl_days=30, prune_unmatched_days=90)
            if removed:
                logger.info("Auto-decay removed %d old antibodies at startup", removed)
        except Exception:
            pass

    def decay(self, ttl_days: int = 30, prune_unmatched_days: int = 90) -> int:
        """Remove antibodies past TTL or not matched recently. Returns count removed."""
        if self._backend == "chromadb":
            try:
                data = self.collection.get()
                if not data or not data.get("ids"):
                    return 0
                now = time.time()
                ttl_sec = ttl_days * 86400
                unmatched_sec = prune_unmatched_days * 86400
                to_delete = []
                for i, doc_id in enumerate(data["ids"]):
                    metadatas = data.get("metadatas") or [{}]
                    meta = metadatas[i] if i < len(metadatas) else {}
                    created_str = meta.get("created_at", "")
                    last_str = meta.get("last_matched", "")
                    created = (
                        datetime.fromisoformat(created_str).timestamp()
                        if created_str else now)
                    last = (
                        datetime.fromisoformat(last_str).timestamp()
                        if last_str else now)
                    if (now - created) >= ttl_sec or (now - last) >= unmatched_sec:
                        to_delete.append(doc_id)
                if to_delete:
                    self.collection.delete(ids=to_delete)
                    logger.info("ChromaDB decay: removed %d antibodies", len(to_delete))
                return len(to_delete)
            except Exception as e:
                logger.warning("Failed to decay chromadb: %s", e)
                return 0
        return self._in_memory.decay(ttl_days, prune_unmatched_days)


# 全局单例
memory_db = ImmunologyMemory()
