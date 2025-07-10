# src/metrics/bleurt.py
import os
from typing import List, Dict
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .base_metric import EvaluationMetric


class BLEURTWrapper:
    """
    对 BLEURT 指标的封装，使其符合 EvaluationMetric 协议。

    使用 transformers 库来加载 Hugging Face Hub 格式的本地模型。
    """

    def __init__(self, **kwargs):
        """
        初始化 BLEURTWrapper。

        Args:
            **kwargs: 关键字参数。
                      `checkpoint` (str): Hugging Face 模型 checkpoint 的名称，
                                         用于构建本地文件路径。
        """
        checkpoint_name = kwargs.get("checkpoint", "bleurt-base-128")
        current_dir = os.path.dirname(__file__)

        checkpoint_path = os.path.join(current_dir, '..', '..', 'local_metrics', 'bleurt', checkpoint_name)

        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"BLEURT checkpoint not found at path: {checkpoint_path}. "
                f"Please make sure you have downloaded the files from Hugging Face Hub correctly."
            )

        print(f"Loading BLEURT from local Hugging Face checkpoint: {checkpoint_path}")


        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path)

        self.device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def compute(
            self,
            predictions: List[str],
            references: List[str],
            **kwargs
    ) -> Dict[str, List[float]]:
        """
        使用加载的 BLEURT 模型计算分數。
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                references,
                predictions,
                return_tensors='pt',
                truncation=True,
                padding=True
            )

            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 模型推理
            scores_tensor = self.model(**inputs)[0].squeeze()

            scores = scores_tensor.cpu().tolist()

        return {"bleurt_score": scores}