# src/metrics/bleu.py
import os
import sys
from typing import List, Dict

current_dir = os.path.dirname(__file__)
local_metrics_path = os.path.join(current_dir, '..', '..', 'local_metrics')
sys.path.append(local_metrics_path)

from bleu.bleu import Bleu
from .base_metric import EvaluationMetric


class BleuWrapper(EvaluationMetric):
    """
    A wrapper class for the BLEU metric that conforms to the EvaluationMetric protocol.
    Loads the metric from a local path.
    """

    def __init__(self, **kwargs):
        """
        Initializes the BleuWrapper from a local checkpoint.
        """
        print("Initializing BLEU by direct import from local_metrics...")
        self.metric = Bleu(**kwargs)
        # current_dir = os.path.dirname(__file__)
        #
        # metric_path = os.path.join(current_dir, '..', '..', 'local_metrics', 'bleu')
        #
        # print(f"Loading BLEU from local path: {metric_path}")
        # self.metric = evaluate.load(metric_path, **kwargs)

    def compute(
            self,
            predictions: List[str],
            references: List[List[str]],
            **kwargs
    ) -> Dict[str, List[float]]:
        """
        Computes BLEU score.
        """
        # Note: The 'evaluate' library's BLEU metric expects references to be a list of lists.
        # We will wrap each reference in a list to match this format.
        formatted_references = [[ref] for ref in references]

        results = self.metric.compute(predictions=predictions, references=formatted_references, **kwargs)

        bleu_score = results.get('bleu', 0.0)

        # Duplicate the corpus-level score for each prediction to maintain consistency.
        num_predictions = len(predictions)
        return {"bleu_score": [bleu_score] * num_predictions}