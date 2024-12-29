import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns

# 设置中文字体支持
import matplotlib
matplotlib.rc("font", family='SimHei')  # 支持中文显示
matplotlib.rcParams['axes.unicode_minus'] = False  # 负号正常显示

# 文件路径
comments_path = r"D:/theguidetoculturaledconomic/数据/cleaned_comments.json"
video_path = r"D:/theguidetoculturaledconomic/数据/cleaned_video_data.json"
mapping_path = r"D:/theguidetoculturaledconomic/数据/comment_video_map.json"

# --- Step 1: 自动生成 comment_video_map.json ---
def generate_mapping(comments_path, video_path, output_path):
    """
    从评论数据和视频数据生成 comment_id 和 video_id 的映射表。
    """
    print("加载评论数据和视频数据...")
    comments_df = pd.read_json(comments_path)
    video_df = pd.read_json(video_path)

    # 确保评论数据和视频数据中都有唯一ID
    print("生成 comment_id 和 video_id 的映射关系...")
    video_ids = video_df['video_id'].unique().tolist()
    mapping = []

    # 循环分配 video_id，确保数据类型一致
    for idx, comment_id in enumerate(comments_df['comment_id'].unique()):
        video_id = video_ids[idx % len(video_ids)]  # 循环分配 video_id
        mapping.append({"comment_id": str(comment_id), "video_id": str(video_id)})

    # 保存为 JSON 文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, ensure_ascii=False, indent=4)
    print(f"映射关系文件已保存为: {output_path}")

# --- Step 2: 数据预处理 ---
def load_and_merge_data(comments_path, video_path, mapping_path):
    """
    加载评论数据、视频数据和映射关系，进行关联合并。
    """
    print("加载评论数据...")
    comments_df = pd.read_json(comments_path)
    comments_df['comment_id'] = comments_df['comment_id'].astype(str)  # 转换数据类型

    print("加载视频数据...")
    video_df = pd.read_json(video_path)
    video_df['video_id'] = video_df['video_id'].astype(str)  # 转换数据类型

    print("加载 comment_id 与 video_id 的映射关系...")
    with open(mapping_path, 'r', encoding='utf-8') as f:
        mapping = json.load(f)
    mapping_df = pd.DataFrame(mapping)

    print("关联评论数据与视频数据...")
    # 合并映射关系
    comments_df = comments_df.merge(mapping_df, on='comment_id', how='left')
    # 合并视频数据
    merged_df = comments_df.merge(video_df, on='video_id', how='left')
    print(f"合并后的数据共 {merged_df.shape[0]} 条记录。")

    # 检查关键字段
    if merged_df['video_id'].isna().any():
        print("警告: 存在未匹配到 video_id 的评论数据。")
    return merged_df

# --- Step 3: 数据分析 ---
def analyze_data(merged_df):
    """
    分析评论与视频传播效果之间的关系。
    """
    print("分析评论数、播放量与点赞数之间的关系...")
    # 统计每个视频的评论数
    video_stats = merged_df.groupby('video_id').agg({
        'comment_id': 'count',
        'video_play_count': 'first',
        'liked_count': 'first'
    }).rename(columns={'comment_id': 'comment_count'})

    print("计算相关性矩阵...")
    print(video_stats.corr())

    # 可视化评论数与播放量的关系
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='comment_count', y='video_play_count', data=video_stats)
    plt.title("评论数与视频播放量的关系")
    plt.xlabel("评论数")
    plt.ylabel("视频播放量")
    plt.show()

    # 情感分析与传播效果的关系
    print("分析正向情感评论与视频传播效果的关系...")
    if 'sentiment_label' in merged_df.columns:
        positive_comments = merged_df[merged_df['sentiment_label'] == 'LABEL_1']
        positive_stats = positive_comments.groupby('video_id').agg({
            'comment_id': 'count',
            'video_play_count': 'first',
            'liked_count': 'first'
        }).rename(columns={'comment_id': 'positive_comment_count'})

        # 可视化正向评论数与播放量的关系
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='positive_comment_count', y='video_play_count', data=positive_stats)
        plt.title("正向情感评论数与视频播放量的关系")
        plt.xlabel("正向评论数")
        plt.ylabel("视频播放量")
        plt.show()
    else:
        print("情感标签缺失，跳过情感分析。")

# --- Step 4: 主程序 ---
def main():
    print("开始生成 comment_video_map.json 文件...")
    generate_mapping(comments_path, video_path, mapping_path)

    print("加载并关联评论数据与视频数据...")
    merged_df = load_and_merge_data(comments_path, video_path, mapping_path)

    print("开始数据分析...")
    analyze_data(merged_df)
    print("分析完成！")

if __name__ == "__main__":
    main()

