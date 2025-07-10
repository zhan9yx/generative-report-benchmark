# main.py
import os
import pandas as pd
import warnings


from src.runner import EvaluationRunner

from src.metrics.bertscore import BERTScoreWrapper
from src.metrics.summac import SummaCWrapper
from src.metrics.bleurt import BLEURTWrapper
from src.metrics.bleu import BleuWrapper
from src.metrics.rouge import RougeWrapper
from src.metrics.meteor import MeteorWrapper

warnings.filterwarnings("ignore", category=UserWarning, module='torch.utils.data')


def load_data_from_files(data_dir: str = 'data') -> pd.DataFrame:
    """
    从data文件夹中读取原始文本和劣化文本。

    结构:
    data/
    ├── original.txt
    └── degraded/
        ├── missing_info.txt
        ├── simplification.txt
        └── ... (其他劣化文本)

    Args:
        data_dir (str): 包含 `original.txt` 和 `degraded/` 子目录的根目录。

    Returns:
        pd.DataFrame: 包含 `original_text`, `degraded_text`, `degradation_type` 的 DataFrame。
    """
    print(f"读取文本： '{data_dir}' ")
    records = []
    original_file_path = os.path.join(data_dir, 'original.txt')
    degraded_dir_path = os.path.join(data_dir, 'degraded')

    try:
        with open(original_file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        records.append({
            'original_text': original_content,
            'degraded_text': original_content,
            'degradation_type': 'Original (Self-comparison)'
        })

        # 读取劣化文本
        if os.path.isdir(degraded_dir_path):
            for filename in os.listdir(degraded_dir_path):
                if filename.endswith('.txt'):
                    degradation_type = os.path.splitext(filename)[0]
                    file_path = os.path.join(degraded_dir_path, filename)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        degraded_content = f.read()

                    records.append({
                        'original_text': original_content,
                        'degraded_text': degraded_content,
                        'degradation_type': degradation_type
                    })

        if len(records) <= 1:  #
            print(f"警告: 在 '{degraded_dir_path}' 中没有找到txt格式的劣化文本。")

    except FileNotFoundError:
        print(f"错误: 找不到 '{original_file_path}'。")
        return pd.DataFrame(columns=['original_text', 'degraded_text', 'degradation_type'])

    return pd.DataFrame(records)


if __name__ == "__main__":
    # ===================================================================
    # 1. 实例化评估指标
    # ===================================================================
    print("正在初始化评估指标")

    # bertscore_metric = BERTScoreWrapper(device='cuda')
    # summac_metric = SummaCWrapper(device='cuda')
    # bleurt_metric = BLEURTWrapper(device='cuda')
    bleu_metric = BleuWrapper()
    rouge_metric = RougeWrapper()
    meteor_metric = MeteorWrapper()

    metrics_to_run = [
        # bertscore_metric,
        # summac_metric,
        # bleurt_metric,
        bleu_metric,
        rouge_metric,
        meteor_metric
    ]

    # ===================================================================
    # 2. 初始化冰进行评估
    # ===================================================================
    runner = EvaluationRunner(metrics=metrics_to_run)

    # 载入数据
    sample_data = load_data_from_files()

    if not sample_data.empty:
        results_df = runner.run(
            data=sample_data,
            original_col='original_text',
            degraded_col='degraded_text'
        )

        # ===================================================================
        # 3. 保存结果
        # ===================================================================
        print("\n" + "=" * 20 + " 最終評估結果 " + "=" * 20)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        # 排序
        results_df = results_df.sort_values(by='degradation_type')
        print(results_df)
        print("=" * 55)

        output_file_path = 'results/evaluation_scores.csv'
        runner.save_results(results_df, output_file_path)
    else:
        print("数据为空。")