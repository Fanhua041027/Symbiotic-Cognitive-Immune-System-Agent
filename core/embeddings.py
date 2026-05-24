"""自定义 ONNX embedding 函数 — 从本地文件加载，避免 ChromaDB S3 下载。"""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import onnxruntime
from tokenizers import Tokenizer

from core.logger import setup_logger

logger = setup_logger("embeddings")

MODEL_DIR = os.path.join(
    os.path.expanduser("~"),
    ".cache/chroma/onnx_models/all-MiniLM-L6-v2",
)

MODEL_PATH = os.path.join(MODEL_DIR, "onnx_model.onnx")
TOKENIZER_PATH = os.path.join(MODEL_DIR, "tokenizer.json")
CONFIG_PATH = os.path.join(MODEL_DIR, "config.json")


class LocalOnnxEmbeddingFunction:
    """Embedding function using local ONNX model + tokenizer files.

    Replaces ChromaDB's built-in ONNXMiniLM_L6_V2 which downloads from S3.
    Loads all-MiniLM-L6-v2 ONNX model from HF mirror-downloaded files.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self._model: onnxruntime.InferenceSession | None = None
        self._tokenizer: Tokenizer | None = None
        self._dimension: int = 384  # all-MiniLM-L6-v2 output dimension
        self._model_path = MODEL_PATH
        self._tokenizer_path = TOKENIZER_PATH

    @staticmethod
    def name() -> str:
        return "local_all_MiniLM_L6_v2"

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "LocalOnnxEmbeddingFunction":
        return LocalOnnxEmbeddingFunction()

    def get_config(self) -> dict[str, Any]:
        return {"model_name": self.MODEL_NAME}

    def is_legacy(self) -> bool:
        return False

    def embed_query(self, input: list[str]) -> list[list[float]]:
        return self.__call__(input)

    def default_space(self) -> str:
        return "cosine"

    def supported_spaces(self) -> list[str]:
        return ["cosine", "l2", "ip"]

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._tokenizer is not None:
            return

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"ONNX model not found at {self._model_path}. "
                "Download from HF mirror first.",
            )
        if not os.path.exists(self._tokenizer_path):
            raise FileNotFoundError(
                f"Tokenizer not found at {self._tokenizer_path}.",
            )

        logger.info("Loading local ONNX model from %s", self._model_path)
        self._model = onnxruntime.InferenceSession(
            self._model_path,
            providers=["CPUExecutionProvider"],
        )
        self._tokenizer = Tokenizer.from_file(self._tokenizer_path)

        # Configure tokenizer for sentence-transformers style
        # Set safe defaults FIRST, then override max_len from config if available
        self._tokenizer.enable_truncation(max_length=256)
        self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        if os.path.exists(CONFIG_PATH):
            try:
                with open(CONFIG_PATH) as f:
                    cfg = json.load(f)
                max_len = cfg.get("max_position_embeddings", 256)
                self._tokenizer.enable_truncation(max_length=max_len)
                self._tokenizer.enable_padding(
                    pad_id=0, pad_token="[PAD]", length=max_len,
                )
            except Exception as e:
                logger.warning("Failed to read config.json: %s", e)

    def __call__(self, input: list[str]) -> list[list[float]]:
        self._ensure_loaded()
        assert self._model is not None
        assert self._tokenizer is not None

        # Tokenize
        encoded = [self._tokenizer.encode(text) for text in input]
        input_ids = np.array([e.ids for e in encoded], dtype=np.int64)
        attention_mask = np.array([e.attention_mask for e in encoded], dtype=np.int64)
        token_type_ids = np.zeros_like(input_ids, dtype=np.int64)

        # Run ONNX model
        outputs = self._model.run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            },
        )

        # Mean pooling
        token_embeddings = outputs[0]
        mask = attention_mask.astype(np.float32)
        mask_expanded = mask[..., np.newaxis]
        sum_embeddings = np.sum(token_embeddings * mask_expanded, axis=1)
        sum_mask = np.maximum(np.sum(mask, axis=1)[..., np.newaxis], 1e-9)
        pooled = sum_embeddings / sum_mask

        # L2 normalize
        norm = np.linalg.norm(pooled, axis=1, keepdims=True)
        normalized = pooled / np.maximum(norm, 1e-9)

        return normalized.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension
