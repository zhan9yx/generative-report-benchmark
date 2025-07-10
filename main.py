# main.py
import os
import pandas as pd
import warnings


from src.runner import EvaluationRunner

from src.metrics.bertscore import BERTScoreWrapper
from src.metrics.summac import SummaCWrapper
from src.metrics.bleurt import BLEURTWrapper


warnings.filterwarnings("ignore", category=UserWarning, module='torch.utils.data')


def load_data_from_files(data_dir: str = 'data') -> pd.DataFrame:
    """
    從指定的資料夾結構中讀取原始文本和多個劣化文本。

    預期的文件結構:
    data/
    ├── original.txt
    └── degraded/
        ├── missing_info.txt
        ├── simplification.txt
        └── ... (其他劣化文本)

    Args:
        data_dir (str): 包含 `original.txt` 和 `degraded/` 子目錄的根目錄。

    Returns:
        pd.DataFrame: 一個包含 `original_text`, `degraded_text`, `degradation_type` 的 DataFrame。
    """
    print(f"正在從 '{data_dir}' 目錄載入文件...")
    records = []
    original_file_path = os.path.join(data_dir, 'original.txt')
    degraded_dir_path = os.path.join(data_dir, 'degraded')

    try:
        # 讀取唯一的源文本
        with open(original_file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()

        # **** 核心改動 1: 添加源文本自身的評估 ****
        # 將源文本與自身比較，作為評估的基準線 (baseline)
        records.append({
            'original_text': original_content,
            'degraded_text': original_content,
            'degradation_type': 'Original (Self-comparison)'
        })

        # 讀取所有劣化文本
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

        if len(records) <= 1:  # 只有源文本自己
            print(f"警告: 在 '{degraded_dir_path}' 中沒有找到任何 .txt 格式的劣化文本。")

    except FileNotFoundError:
        print(f"錯誤: 找不到 '{original_file_path}'。請確保文件存在並路徑正確。")
        # 創建一個空的 DataFrame 返回，以避免後續崩潰
        return pd.DataFrame(columns=['original_text', 'degraded_text', 'degradation_type'])

    return pd.DataFrame(records)


if __name__ == "__main__":
    # ===================================================================
    # 1. 實例化並註冊您想要運行的評估指標
    # ===================================================================
    print("正在初始化評估指標...")

    bertscore_metric = BERTScoreWrapper(device='cuda')
    summac_metric = SummaCWrapper(device='cuda')
    bleurt_metric = BLEURTWrapper(device='cuda')

    metrics_to_run = [
        bertscore_metric,
        summac_metric,
        bleurt_metric,
    ]

    # ===================================================================
    # 2. 初始化並運行評估器
    # ===================================================================
    runner = EvaluationRunner(metrics=metrics_to_run)

    # **** 核心改動 2: 從文件載入資料 ****
    # 從 `data` 目錄載入，而不是使用硬編碼的範例
    sample_data = load_data_from_files()

    if not sample_data.empty:
        # 執行評估流程
        results_df = runner.run(
            data=sample_data,
            original_col='original_text',
            degraded_col='degraded_text'
        )

        # ===================================================================
        # 3. 顯示並保存結果
        # ===================================================================
        print("\n" + "=" * 20 + " 最終評估結果 " + "=" * 20)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 200)
        # 根據劣化類型排序，讓源文本的結果顯示在第一行
        results_df = results_df.sort_values(by='degradation_type')
        print(results_df)
        print("=" * 55)

        output_file_path = 'results/evaluation_scores.csv'
        runner.save_results(results_df, output_file_path)
    else:
        print("資料為空，已跳過評估流程。")