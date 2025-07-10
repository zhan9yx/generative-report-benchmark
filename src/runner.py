# src/runner.py

"""
定义中央评估运行器 (EvaluationRunner)，负责协调和执行整个评估流程。
"""

import pandas as pd
from tqdm import tqdm
from typing import List
import os

# 从我们自己的模块中导入基础协议，用于类型提示
# 这确保了传递给运行器的任何指标对象都符合我们设计的接口
from .metrics.base_metric import EvaluationMetric


class EvaluationRunner:
    """
    一个中央控制器类，用于管理和执行文本评估指标。

    这个类会接收一个指标列表，并在给定的数据集上依次运行它们，
    然后将所有结果汇总到一个pandas DataFrame中。
    """

    def __init__(self, metrics: List[EvaluationMetric]):
        """
        初始化评估运行器。

        Args:
            metrics (List[EvaluationMetric]): 一个包含评估指标实例的列表。
                                              每个实例都必须符合在 base_metric.py
                                              中定义的 EvaluationMetric 协议。
        """
        if not metrics:
            raise ValueError("指标列表 (metrics) 不能为空。")
        self.metrics = metrics

    def run(self, data: pd.DataFrame, original_col: str, degraded_col: str) -> pd.DataFrame:
        """
        对给定的数据运行所有已注册的评估指标。

        此方法会遍历所有在初始化时注册的指标，使用DataFrame中的指定列
        作为输入来计算分数，并将这些分数作为新列添加回DataFrame。

        Args:
            data (pd.DataFrame): 包含待评估文本的DataFrame。
            original_col (str): DataFrame中包含原始参考文本的列名。
            degraded_col (str): DataFrame中包含待评估生成文本的列名。

        Returns:
            pd.DataFrame: 一个新的DataFrame，包含了原始数据以及所有新计算出的指标分数。
        """
        # 创建数据副本以避免修改传入的原始DataFrame，这是一个良好的编程实践
        results_df = data.copy()

        # 从DataFrame中一次性提取所有文本到列表，这样可以避免在循环中重复操作，提高效率
        predictions = results_df[degraded_col].tolist()
        references = results_df[original_col].tolist()

        print(f"开始对 {len(predictions)} 条数据进行评估...")
        print(f"参考文本列: '{original_col}', 待评估文本列: '{degraded_col}'")

        # 使用tqdm包装指标列表的迭代，以显示一个清晰的进度条
        for metric in tqdm(self.metrics, desc="正在计算评估指标"):
            # 获取当前指标的类名，用于日志输出
            metric_name = metric.__class__.__name__
            print(f"  -> 正在运行: {metric_name}...")

            # 调用每个指标实例的compute方法，这是我们协议的核心
            scores_dict = metric.compute(predictions, references)

            # 将返回的每个分数列表（例如 'bertscore_f1', 'bertscore_precision'）
            # 作为新列添加到结果DataFrame中
            for score_name, score_values in scores_dict.items():
                if len(score_values) != len(results_df):
                     print(f"    [警告] 指标 '{score_name}' 返回了 "
                           f"{len(score_values)} 个结果, 但输入有 {len(results_df)} 行。可能存在问题。")
                results_df[score_name] = score_values
                print(f"    -- 已添加新列: '{score_name}'")

        print("所有评估指标计算完成。")
        return results_df

    def save_results(self, results_df: pd.DataFrame, output_path: str):
        """
        将包含评估结果的DataFrame保存到指定的CSV文件路径。

        此方法会先确保目标目录存在，然后再保存文件。

        Args:
            results_df (pd.DataFrame): 由 run() 方法返回的、包含结果的DataFrame。
            output_path (str): 保存CSV文件的完整路径 (例如 'results/evaluation_scores.csv')。
        """
        # 从完整路径中提取目录部分
        output_dir = os.path.dirname(output_path)

        # 如果路径中包含目录（即不只是一个文件名），则创建该目录
        # exist_ok=True 参数可以防止在目录已存在时抛出错误
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        # 将DataFrame保存为CSV文件，index=False表示不将DataFrame的行索引写入文件
        # 使用 'utf-8-sig' 编码可以确保在Excel中打开时能正确显示中文等非英文字符
        results_df.to_csv(output_path, index=False, encoding='utf-8-sig')
        print(f"结果已成功保存至: {output_path}")