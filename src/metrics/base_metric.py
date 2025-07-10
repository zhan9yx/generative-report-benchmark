import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# src/metrics/base_metric.py

"""
Defines the base interface (Protocol) for all evaluation metrics.

This module ensures that every metric implemented in the project adheres to a
common structure, making them interchangeable and modular.
"""

from typing import Protocol, List, Dict

class EvaluationMetric(Protocol):
    """
    A protocol that defines the standard interface for an evaluation metric.

    All specific metric classes (e.g., BLEU, BERTScore) should conform to this
    protocol to ensure they can be used interchangeably by the evaluation runner.
    """

    def __init__(self, **kwargs) -> None:
        """
        Initializes the metric.

        This method allows for passing any metric-specific configuration
        parameters, such as model checkpoints, device settings (e.g., 'cuda'
        or 'cpu'), or batch sizes.

        Args:
            **kwargs: A dictionary of arbitrary keyword arguments for
                      metric-specific configuration.
        """
        ...

    def compute(
        self,
        predictions: List[str],
        references: List[str],
        **kwargs
    ) -> Dict[str, List[float]]:
        """
        Computes the evaluation score(s) for a batch of predictions and references.

        Args:
            predictions (List[str]): A list of generated texts to be evaluated
                                     (e.g., degraded texts).
            references (List[str]): A list of corresponding reference texts
                                    (e.g., original texts).
            **kwargs: A dictionary for metric-specific arguments. For example,
                      some metrics might require 'sources' for consistency checks.

        Returns:
            Dict[str, List[float]]: A dictionary where keys are the names of the
                                    scores computed by the metric (e.g., 'bertscore_f1',
                                    'bleu') and values are lists of floats. Each list
                                    contains the scores for the corresponding
                                    prediction-reference pair and must have the
                                    same length as the input `predictions` list.
        """
