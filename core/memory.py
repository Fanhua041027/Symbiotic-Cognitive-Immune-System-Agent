"""免疫记忆系统 - 基于 ChromaDB 的抗体存储与检索。"""

import os
import uuid
from typing import Optional, Dict

import chromadb
from chromadb.config import Settings

from core.logger import setup_logger

# 持久化存储路径
DB_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".immune_db")
logger = setup_logger("immune_db")


class ImmunologyMemory:
    """管理免疫记忆：存储和检索历史抗体（补丁）。"""

    def __init__(self, collection_name: str = "antibodies"):
        os.makedirs(DB_DIR, exist_ok=True)
        self.client = chromadb.PersistentClient(
            path=DB_DIR, settings=Settings(anonymized_telemetry=False)
        )
        self.collection = self.client.get_or_create_collection(collection_name)

    def store_antibody(
        self, error_pattern: str, antibody_code: str, context: str
    ) -> None:
        """存储有效的抗体到向量数据库。"""
        self.collection.add(
            documents=[context],
            ids=[str(uuid.uuid4())],
            metadatas=[{"error_pattern": error_pattern, "code": antibody_code}],
        )
        logger.info(
            "Stored new antibody for pattern: %s...", error_pattern[:50]
        )

    def search_antibody(self, query: str) -> Optional[Dict[str, str]]:
        """检索相似的历史错误及对应抗体。"""
        results = self.collection.query(query_texts=[query], n_results=1)
        if results["ids"] and results["ids"][0]:
            return {
                "code": results["metadatas"][0][0]["code"],
                "pattern": results["metadatas"][0][0]["error_pattern"],
            }
        return None


# 全局单例
memory_db = ImmunologyMemory()
