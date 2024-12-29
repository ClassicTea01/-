import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 配置 matplotlib 支持中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用 SimHei 字体显示中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号'-'显示问题

# --- Step 1: 数据加载与整合 ---
def load_and_merge_data(sentiment_file, video_file):
    """
    加载情感分析结果和视频互动数据，并整合为一个数据框。
    """
    if not os.path.exists(sentiment_file):
        raise FileNotFoundError(f"情感分析结果文件不存在: {sentiment_file}")
    if not os.path.exists(video_file):
        raise FileNotFoundError(f"视频数据文件不存在: {video_file}")

    print("\n加载情感分析结果...")
    sentiment_df = pd.read_json(sentiment_file)

    print("\n加载视频互动数据...")
    video_df = pd.read_json(video_file)

    print("\n处理视频互动数据：去重...")
    # 确保 user_id 唯一并处理 video_id
    video_df_unique = video_df.drop_duplicates(subset='user_id', keep='first').copy()
    video_df_unique['video_id'] = video_df_unique['video_id'].astype(str)

    print("\n映射评论到视频...")
    # 通过 user_id 将评论数据映射到视频
    sentiment_df['video_id'] = sentiment_df['user_id'].map(
        video_df_unique.set_index('user_id')['video_id']
    )

    # 过滤出有效的映射结果
    print(f"映射成功的评论数: {sentiment_df['video_id'].notnull().sum()} / {len(sentiment_df)}")
    sentiment_df = sentiment_df.dropna(subset=['video_id']).copy()
    sentiment_df['video_id'] = sentiment_df['video_id'].astype(str)

    print("\n数据整合中...")
    # 合并评论数据和视频数据
    merged_df = pd.merge(sentiment_df, video_df_unique, on='video_id', how='inner')

    # 确保数据列类型正确
    numeric_columns = ['sub_comment_count', 'liked_count', 'sentiment_score', 'video_play_count']
    for col in numeric_columns:
        merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

    merged_df = merged_df.dropna(subset=numeric_columns)
    print(f"\n数据整合完成！有效数据条数: {len(merged_df)}")
    print(merged_df.head())
    return merged_df

# --- Step 2: 回归分析 ---
def perform_regression_analysis(df):
    print("\n开始回归分析...")
    label_encoder = LabelEncoder()
    df['sentiment_label_encoded'] = label_encoder.fit_transform(df['sentiment_label'])

    X = df[['sentiment_score', 'sentiment_label_encoded', 'sub_comment_count', 'video_play_count']]
    y = df['liked_count']

    if len(X) == 0 or len(y) == 0:
        raise ValueError("回归分析输入数据为空，请检查数据处理步骤。")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\n线性回归模型评估结果：\n均方误差 (MSE): {mse:.4f}\nR² 分数: {r2:.4f}")
    return model

# --- Step 3: 机器学习建模 (XGBoost) ---
def train_xgboost_model(df):
    print("\n训练 XGBoost 模型...")
    label_encoder = LabelEncoder()
    df['sentiment_label_encoded'] = label_encoder.fit_transform(df['sentiment_label'])

    X = df[['sentiment_score', 'sentiment_label_encoded', 'sub_comment_count', 'video_play_count']]
    y = df['liked_count']

    if len(X) == 0 or len(y) == 0:
        raise ValueError("XGBoost 模型输入数据为空，请检查数据处理步骤。")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = XGBRegressor(objective='reg:squarederror', n_estimators=100, learning_rate=0.1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"\nXGBoost 模型评估结果：\n均方误差 (MSE): {mse:.4f}\nR² 分数: {r2:.4f}")
    return model

# --- Step 4: 可视化分析 ---
def visualize_results(df):
    print("\n生成可视化图表...")
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='sentiment_score', y='liked_count', hue='sentiment_label', data=df, palette='viridis')
    plt.title("情感强度与点赞数的关系")
    plt.xlabel("情感强度 (Sentiment Score)")
    plt.ylabel("点赞数 (Liked Count)")
    plt.tight_layout()
    plt.show()

# --- Step 5: 主函数 ---
def main():
    sentiment_file = r"D:/theguidetoculturaledconomic/数据/情感分析结果打分版本.json"
    video_file = r"D:/theguidetoculturaledconomic/数据/cleaned_video_data.json"

    try:
        # Step 1: 加载与整合数据
        df = load_and_merge_data(sentiment_file, video_file)

        # Step 2: 执行回归分析
        perform_regression_analysis(df)

        # Step 3: 使用XGBoost进行建模
        train_xgboost_model(df)

        # Step 4: 生成可视化图表
        visualize_results(df)

        print("\n分析完成！")
    except Exception as e:
        print(f"\n运行时发生错误: {e}")

# --- 执行主程序 ---
if __name__ == "__main__":
    main()
