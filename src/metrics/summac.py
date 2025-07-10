import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# src/metrics/summac.py

"""
Wrapper for the SummaC metric, an independent library for evaluating
factual consistency in summaries.
"""

from typing import List, Dict
from summac.model_summac import SummaCConv
import torch

from .base_metric import EvaluationMetric


class SummaCWrapper:
    """
    A wrapper class for the SummaC-Conv model that conforms to the
    EvaluationMetric protocol.

    This class initializes the SummaCConv model and uses it to compute
    consistency scores.
    """

    def __init__(self, **kwargs):
        """
        Initializes the SummaCWrapper.

        Args:
            **kwargs: Keyword arguments for initializing the SummaCConv model.
                      Common arguments include 'model_name', 'granularity',
                      and 'device'.
        """
        # Set default device if not provided
        if "device" not in kwargs:
            kwargs["device"] = "cuda" if torch.cuda.is_available() else "cpu"

        self.model = SummaCConv(**kwargs)

    def compute(
            self,
            predictions: List[str],
            references: List[str],
            **kwargs
    ) -> Dict[str, List[float]]:
        """
        Computes the SummaC-Conv score for factual consistency.

        Args:
            predictions (List[str]): The list of generated summaries.
            references (List[str]): The list of original source documents.
            **kwargs: Ignored for this metric, but kept for protocol compatibility.

        Returns:
            Dict[str, List[float]]: A dictionary with a single key 'summac_conv_score'
                                    containing the consistency scores.
        """
        # 修正：根据报错信息，将参数名从 documents, summaries 改为 originals, generateds
        # 'references' 对应 'originals' (原始文本)
        # 'predictions' 对应 'generateds' (生成文本)
        results = self.model.score(originals=references, generateds=predictions)

        # The result object contains a 'scores' key with a list of floats.
        # We format it to match our protocol's return type.
        return {"summac_conv_score": results["scores"]}