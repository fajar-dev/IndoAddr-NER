from transformers import pipeline, AutoTokenizer, AutoModelForTokenClassification
from .config import settings
from typing import List, Dict, Any
import logging
import os

logger = logging.getLogger(__name__)

class NerModel:
    """Wrapper around transformers token-classification pipeline for NER."""
    def __init__(self, model_dir: str = None, model_name: str = None, device: str = "cpu", aggregation_strategy: str = "simple"):
        self.model_dir = model_dir or model_name
        self.model_name = model_name
        self.device = 0 if device.startswith('cuda') else -1  # pipeline expects device int
        self.aggregation_strategy = aggregation_strategy
        self._pipeline = None
        self._load()

    def _load(self):
        # prefer model_dir if exists, else model_name
        source = None
        if self.model_dir and os.path.isdir(self.model_dir):
            source = self.model_dir
        else:
            source = self.model_name
        logger.info(f"Loading model from {source} on device {self.device}")
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForTokenClassification.from_pretrained(source)
        self._pipeline = pipeline(
            task="token-classification",
            model=model,
            tokenizer=tokenizer,
            aggregation_strategy=self.aggregation_strategy,
            device=self.device
        )

    def predict(self, text: str) -> List[Dict[str, Any]]:
        if not self._pipeline:
            self._load()
        return self._pipeline(text)
