# src/metrics/meteor.py
import os
from typing import List, Dict
import evaluate
import nltk
from .base_metric import EvaluationMetric


class MeteorWrapper(EvaluationMetric):
    """
    A wrapper class for the METEOR metric that conforms to the EvaluationMetric protocol.
    Loads the metric from a local path.
    """

    def __init__(self, **kwargs):
        """
        Initializes the MeteorWrapper from a local checkpoint.
        Ensures the 'wordnet' resource from NLTK is available.
        """
        try:
            # 檢查 'wordnet' 是否存在
            nltk.data.find('corpora/wordnet.zip')
        except LookupError:
            # --- 修正：捕捉正確的 LookupError 異常 ---
            print("=" * 80)
            print("!! NLTK 'wordnet' 資源未找到 !!")
            print("由於您的環境可能無法連線網路，自動下載可能失敗。")
            print("請手動下載 'wordnet.zip' 並將其放置在 NLTK 的資料目錄中。")
            print("1. 下載連結: https://github.com/nltk/nltk_data/blob/gh-pages/packages/corpora/wordnet.zip?raw=true")
            print(
                f"2. 將 wordnet.zip 文件放入以下任一資料夾的 'corpora' 子目錄中: \n   (例如: C:/Users/YourUser/nltk_data/corpora/wordnet.zip)")
            print("\n   NLTK 搜索路徑:")
            for path in nltk.data.path:
                print(f"   - {path}")
            print("=" * 80)
            # 即使手動指導後，仍然嘗試下載，以防萬一網路是通的
            try:
                nltk.download('wordnet')
            except Exception as e:
                # 如果下載仍然失敗，則拋出更明確的錯誤
                raise LookupError("自動下載 'wordnet' 失敗。請遵循上面的手動說明。") from e

        current_dir = os.path.dirname(__file__)
        metric_path = os.path.join(current_dir, '..', '..', 'local_metrics', 'meteor')

        print(f"Loading METEOR from local path: {metric_path}")
        self.metric = evaluate.load(metric_path, **kwargs)

    def compute(
            self,
            predictions: List[str],
            references: List[str],
            **kwargs
    ) -> Dict[str, List[float]]:
        """
        Computes METEOR score.
        """
        results = self.metric.compute(predictions=predictions, references=references, **kwargs)
        meteor_score = results.get('meteor', 0.0)

        # Duplicate the corpus-level score for each prediction.
        num_predictions = len(predictions)
        return {"meteor_score": [meteor_score] * num_predictions}