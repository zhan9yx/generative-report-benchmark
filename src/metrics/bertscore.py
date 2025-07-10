# src/metrics/bertscore.py
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
"""
Wrapper for the BERTScore metric using the Hugging Face evaluate library.
"""

from typing import List, Dict
import evaluate
import torch


from .base_metric import EvaluationMetric


class BERTScoreWrapper:
    """
    A wrapper class for the BERTScore metric that conforms to the
    EvaluationMetric protocol.

    This class loads the metric using `evaluate.load('bertscore')` and handles
    the computation and result formatting.
    """

    def __init__(self, **kwargs):
        """
        Initializes the BERTScoreWrapper.

        Args:
            **kwargs: Keyword arguments passed to the metric. The 'device'
                      argument is specifically used to set the computation device.
        """
        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        # Pop device from kwargs if it exists, as evaluate.load doesn't use it.
        # It's used in the compute method instead.
        # 从 src / metrics / bertscore.py 找到 main / local_metrics / bertscore
        current_dir = os.path.dirname(__file__)
        # 从 'D:\...\src\metrics' 回到 'D:\...\main' 再进入 'local_metrics\bertscore'
        metric_path = os.path.join(current_dir, '..', '..', 'local_metrics', 'bertscore')
        kwargs.pop("device", None)
        #self.metric = evaluate.load('bertscore', **kwargs)
        self.metric = evaluate.load(metric_path, **kwargs)

    def compute(
            self,
            predictions: List[str],
            references: List[str],
            **kwargs
    ) -> Dict[str, List[float]]:
        """
        Computes BERTScore for the given predictions and references.

        Args:
            predictions (List[str]): The list of generated texts.
            references (List[str]): The list of reference texts.
            **kwargs: Additional arguments for the metric's compute function.
                      'lang' can be passed here, defaults to 'en'.

        Returns:
            Dict[str, List[float]]: A dictionary containing precision, recall,
                                    and F1 scores from BERTScore, with keys
                                    'bertscore_precision', 'bertscore_recall',
                                    and 'bertscore_f1'.
        """
        lang = kwargs.get("lang", "en")  # Default to English for BERTScore

        scores = self.metric.compute(
            predictions=predictions,
            references=references,
            lang=lang,
            device=self.device
        )

        # The 'hashcode' key is not needed in the final output.
        scores.pop("hashcode", None)

        # Format the output to match the EvaluationMetric protocol.
        formatted_scores = {
            f"bertscore_{key}": value for key, value in scores.items()
        }

        return formatted_scores