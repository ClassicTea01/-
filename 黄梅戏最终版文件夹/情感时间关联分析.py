import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import numpy as np

# 1. 数据加载与时间戳转换
def load_and_preprocess_data(input_file):
    print("加载数据并处理时间戳...")
    df = pd.read_json(input_file)

    # 时间戳转换为标准时间
    df['create_time'] = pd.to_datetime(df['create_time'], unit='s')
    df['day'] = df['create_time'].dt.date
    df['hour'] = df['create_time'].dt.hour
    df['week'] = df['create_time'].dt.isocalendar().week

    # 映射情感标签
    sentiment_map = {'LABEL_0': '负向', 'LABEL_1': '中性', 'LABEL_2': '正向'}
    df['sentiment_category'] = df['sentiment_label'].map(sentiment_map)

    # 添加模拟互动数据
    df['like_count'] = np.random.randint(0, 100, size=len(df))  # 点赞数
    df['share_count'] = np.random.randint(0, 50, size=len(df))  # 分享数

    print("时间戳处理完成！")
    return df

# 2. 按时间段统计情感分布和平均情感强度
def calculate_sentiment_summary(df):
    print("按时间段统计情感分布和平均强度...")
    sentiment_summary = df.groupby('day').agg({
        'sentiment_category': lambda x: x.value_counts(normalize=True).to_dict(),
        'sentiment_score': 'mean',
        'like_count': 'sum',
        'share_count': 'sum'
    }).reset_index()
    print("统计完成！")
    return sentiment_summary

# 3. 滑动窗口分析情感强度波动
def sliding_window_analysis(df, window='1D'):
    print(f"进行滑动窗口分析，窗口大小: {window}...")

    # 确保 create_time 是索引并且按时间升序排列
    df = df.sort_index()
    df = df.set_index('create_time')  # 确保时间列为索引

    # 使用 resample 按时间窗口聚合数据
    sliding_df = df['sentiment_score'].resample(window).mean().reset_index()
    sliding_df.rename(columns={'sentiment_score': 'sentiment_score_avg'}, inplace=True)

    print("滑动窗口分析完成！")
    return sliding_df

# 4. 可视化结果
def visualize_results(sentiment_summary, sliding_df):
    print("生成可视化图表...")
    plt.style.use('seaborn-v0_8-darkgrid')

    # 情感分布
    sentiment_df = pd.DataFrame(sentiment_summary['sentiment_category'].tolist(), index=sentiment_summary['day'])
    sentiment_df = sentiment_df.fillna(0)
    sentiment_df.plot(kind='bar', stacked=True, figsize=(12, 6))
    plt.title("情感分布随时间变化")
    plt.tight_layout()
    plt.show()

    # 滑动窗口
    plt.figure(figsize=(12, 6))
    plt.plot(sliding_df['create_time'], sliding_df['sentiment_score_avg'], marker='o')
    plt.title("滑动窗口内情感强度波动")
    plt.xlabel("Time")
    plt.ylabel("Average Sentiment Score")
    plt.tight_layout()
    plt.show()

# 5. 保存结果
def save_results_to_json(sentiment_summary, output_file):
    print(f"保存分析结果到: {output_file}")
    sentiment_summary.to_json(output_file, orient='records', force_ascii=False, indent=4)
    print("结果保存成功！")

# 6. 主函数
def main():
    input_file = r"D:/theguidetoculturaledconomic/数据/情感分析结果打分版本.json"
    output_file = r"D:/theguidetoculturaledconomic/数据/情感与时间关联分析.json"

    df = load_and_preprocess_data(input_file)

    print("处理中...")
    for _ in tqdm(range(100), desc="数据分析中", ncols=100):
        pass

    sentiment_summary = calculate_sentiment_summary(df)
    sliding_df = sliding_window_analysis(df.copy(), window='1D')

    visualize_results(sentiment_summary, sliding_df)
    save_results_to_json(sentiment_summary, output_file)

    print("情感与时间关联分析完成！")

if __name__ == "__main__":
    main()
