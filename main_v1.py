import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
# main.py

"""
项目主入口脚本 (Main Entry Point)。

本脚本演示了如何使用 EvaluationRunner 来运行一系列文本评估指标。
它遵循以下步骤：
1. 准备示例数据（模拟用户输入）。
2. 初始化并注册所有需要运行的评估指标。
3. 实例化并运行中央评估器 EvaluationRunner。
4. 打印并保存最终的评估结果。

同时，本脚本也包含了清晰的指南，说明如何轻松添加新的评估指标。
"""

import pandas as pd
import warnings

# 导入我们的核心评估运行器
from src.runner import EvaluationRunner

# 导入我们已经实现的具体指标封装类
from src.metrics.bertscore import BERTScoreWrapper
from src.metrics.summac import SummaCWrapper
from src.metrics.bleurt import BLEURTWrapper
# from src.metrics.new_metric import NewMetricWrapper # <- 未来添加新指标时在此处取消注释

# 忽略一些第三方库可能产生的特定警告，保持输出整洁
warnings.filterwarnings("ignore", category=UserWarning, module='torch.utils.data')


def get_data_from_user_input() -> pd.DataFrame:
    """
    创建一个包含示例数据的 pandas DataFrame。

    重要: 此函数用于模拟从用户界面（如文件对话框、文本输入框）获取
    原始文本和劣化文本的过程。在实际应用中，您需要在此处替换为您的
    真实数据加载逻辑（例如，从CSV、Excel或数据库加载数据）。
    """
    print("正在加载数据... (当前为模拟数据)")
    data = {
        'original_text': [
            "Our company is committed to reducing its carbon footprint by 20% by 2030, investing in renewable energy sources like solar and wind power.",
            "Employee wellness programs have been expanded to include mental health support, flexible working hours, and subsidized gym memberships.",
            "The board has approved a new supplier code of conduct that mandates fair labor practices and environmental sustainability checks for all partners.",
            "In 2023, we achieved a 50% representation of women in management positions, reaching our diversity goal two years ahead of schedule."
        ],
        'degraded_text': [
            "Our company aims to lower carbon emissions by 20% before 2030 by using solar energy.", # 劣化类型：信息丢失
            "Employee wellness programs now include mental health support.", # 劣化类型：简化摘要
            "The board has approved a new supplier code of conduct. It's about how partners should act.", # 劣化类型：模糊化
            "In 2023, we achieved a 30% representation of men in management positions." # 劣化类型：事实错误
        ],
        'degradation_type': [
            'Missing Information',
            'Simplification',
            'Vague Paraphrasing',
            'Factual Error'
        ]
    }
    return pd.DataFrame(data)


if __name__ == "__main__":
    # ===================================================================
    # 1. 实例化并注册您想要运行的评估指标
    # ===================================================================
    print("正在初始化评估指标...")

    # 为每个指标创建一个实例。您可以传递特定参数，如 device。
    # 建议默认使用 'cpu' 以保证在没有GPU的机器上也能运行。
    # 如果您有可用的CUDA环境，可以改为 'cuda' 以获得更快的速度。
    bertscore_metric = BERTScoreWrapper(device='cpu')
    summac_metric = SummaCWrapper(device='cpu')
    bleurt_metric = BLEURTWrapper(device='cpu')

    # ===================================================================
    # 扩展指南: 要添加一个新的评估指标 (例如 'NewMetric'), 请按以下步骤操作:
    # 1. 在 `src/metrics/` 目录下创建一个 `new_metric.py` 文件。
    # 2. 在该文件中创建一个 `NewMetricWrapper` 类, 并实现 `EvaluationMetric` 协议。
    # 3. 在此处的 `main.py` 中导入它: `from src.metrics.new_metric import NewMetricWrapper`
    # 4. 在下面的 `metrics_to_run` 列表中实例化并添加它: `NewMetricWrapper()`
    #
    # 就是这样！无需修改 `EvaluationRunner` 的任何代码。
    # ===================================================================
    metrics_to_run = [
        bertscore_metric,
        summac_metric,
        bleurt_metric,
        # NewMetricWrapper() # <- 未来添加新指标时在此处取消注释
    ]

    # ===================================================================
    # 2. 初始化并运行评估器
    # ===================================================================
    # 创建 EvaluationRunner 的实例，并将指标列表传入
    runner = EvaluationRunner(metrics=metrics_to_run)

    # 从模拟函数或您的真实数据源加载数据
    sample_data = get_data_from_user_input()

    # 执行评估流程
    # .run() 方法会返回一个包含所有原始数据和新计算出的分数的新DataFrame
    results_df = runner.run(
        data=sample_data,
        original_col='original_text',
        degraded_col='degraded_text'
    )

    # ===================================================================
    # 3. 显示并保存结果
    # ===================================================================
    print("\n" + "="*20 + " 最终评估结果 " + "="*20)
    # 使用pd.set_option可以更好地在控制台显示宽表格
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 200)
    print(results_df)
    print("="*55)

    # 定义结果保存路径
    output_file_path = 'results/evaluation_scores.csv'

    # 调用save_results方法将DataFrame保存到CSV文件
    runner.save_results(results_df, output_file_path)