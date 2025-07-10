# src/metrics/rouge.py
import os
from typing import List, Dict
import evaluate
from .base_metric import EvaluationMetric


class RougeWrapper(EvaluationMetric):
    """
    A wrapper class for the ROUGE metric that conforms to the EvaluationMetric protocol.
    Loads the metric from a local path.
    """

    def __init__(self, **kwargs):
        """
        Initializes the RougeWrapper from a local checkpoint.
        """
        current_dir = os.path.dirname(__file__)
        metric_path = os.path.join(current_dir, '..', '..', 'local_metrics', 'rouge')

        print(f"Loading ROUGE from local path: {metric_path}")
        self.metric = evaluate.load(metric_path, **kwargs)

    def compute(
            self,
            predictions: List[str],
            references: List[str],
            **kwargs
    ) -> Dict[str, List[float]]:
        """
        Computes ROUGE scores.
        """
        results = self.metric.compute(predictions=predictions, references=references, **kwargs)

        num_predictions = len(predictions)
        formatted_scores = {}


        for key, value in results.items():
            # 將單一的 corpus-level 分數複製 N 份，N 為樣本數
            formatted_scores[f"rouge_{key}"] = [value] * num_predictions

        return formatted_scores