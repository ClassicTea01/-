import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from snownlp import SnowNLP

# 设置中文字体支持
import matplotlib

matplotlib.rc("font", family='SimHei')  # 支持中文显示
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# 文件路径
comments_path = r"D:/theguidetoculturaledconomic/数据/cleaned_comments.json"
video_path = r"D:/theguidetoculturaledconomic/数据/cleaned_video_data.json"
mapping_path = r"D:/theguidetoculturaledconomic/数据/comment_video_map.json"


# --- Step 1: 加载数据与预处理 ---
def load_and_merge_data(comments_path, video_path, mapping_path):
    """
    加载评论数据、视频数据和映射关系，进行关联合并。
    """
    print("加载评论数据...")
    comments_df = pd.read_json(comments_path)

    print("加载视频数据...")
    video_df = pd.read_json(video_path)

    print("加载 comment_id 与 video_id 的映射关系...")
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    mapping_df = pd.DataFrame(mapping)

    print("合并评论数据与视频数据...")
    comments_df = comments_df.merge(mapping_df, on='comment_id', how='left')
    merged_df = comments_df.merge(video_df, on='video_id', how='left')
    print(f"合并后的数据共 {merged_df.shape[0]} 条记录。")

    return merged_df


# --- Step 2: 添加情感分析 ---
def add_sentiment_analysis(comments_df):
    """
    使用 SnowNLP 为评论数据添加情感标签和分数。
    """
    print("开始情感分析...")
    sentiment_labels = []
    sentiment_scores = []

    for content in comments_df['content']:
        # 过滤空值和无效内容
        if not isinstance(content, str) or len(content.strip()) == 0:
            sentiment_scores.append(0.5)  # 默认中性情感分数
            sentiment_labels.append('LABEL_0')  # 标记为中性
            continue

        s = SnowNLP(content)
        score = s.sentiments
        sentiment_scores.append(score)
        sentiment_labels.append('LABEL_1' if score >= 0.5 else 'LABEL_0')

    # 添加新列到 DataFrame
    comments_df['sentiment_label'] = sentiment_labels
    comments_df['sentiment_score'] = sentiment_scores
    print("情感分析完成！")
    return comments_df


# --- Step 3: 数据可视化 ---
def visualize_user_type_distribution(merged_df):
    """
    可视化评论用户类型占比（头部用户 vs 普通用户）。
    """
    print("可视化用户类型占比...")
    merged_df['is_top_user'] = merged_df['liked_count'] > 10000  # 简单判断头部用户
    user_type_counts = merged_df['is_top_user'].value_counts()

    plt.figure(figsize=(8, 6))
    user_type_counts.plot(kind='pie', autopct='%1.1f%%', labels=['普通用户', '头部用户'],
                          colors=['lightblue', 'orange'])
    plt.title("用户类型占比（头部用户 vs 普通用户）")
    plt.ylabel('')
    plt.show()


def visualize_sentiment_vs_likes(merged_df):
    """
    可视化情感分析结果与点赞数的关系。
    """
    print("可视化情感分析与点赞数的关系...")
    sentiment_stats = merged_df.groupby('sentiment_label')['liked_count'].mean()
    print("情感标签与点赞数的平均值：")
    print(sentiment_stats)

    plt.figure(figsize=(8, 6))
    sentiment_stats.plot(kind='bar', color=['lightgreen', 'lightcoral'])
    plt.title("情感分析结果与点赞数的关系")
    plt.xlabel("情感标签")
    plt.ylabel("平均点赞数")
    plt.xticks(rotation=0)
    plt.show()


def visualize_comments_vs_playcount(merged_df):
    """
    可视化视频播放量与评论数量的分布关系。
    """
    print("可视化视频播放量与评论数量的关系...")
    video_stats = merged_df.groupby('video_id').agg({
        'comment_id': 'count',
        'video_play_count': 'first'
    }).rename(columns={'comment_id': 'comment_count'})

    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='comment_count', y='video_play_count', data=video_stats, color='purple')
    plt.title("视频播放量与评论数量的关系")
    plt.xlabel("评论数量")
    plt.ylabel("视频播放量")
    plt.show()


# --- Step 4: 主程序 ---
def main():
    # 加载并合并数据
    merged_df = load_and_merge_data(comments_path, video_path, mapping_path)

    # 检查并添加情感分析
    if 'sentiment_label' not in merged_df.columns:
        merged_df = add_sentiment_analysis(merged_df)
    else:
        print("情感分析列已存在，跳过情感分析步骤。")

    # 数据可视化
    visualize_user_type_distribution(merged_df)
    visualize_sentiment_vs_likes(merged_df)
    visualize_comments_vs_playcount(merged_df)


# --- 程序执行 ---
if __name__ == "__main__":
    main()
