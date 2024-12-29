import pandas as pd
from snownlp import SnowNLP
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置字体为 SimHei，支持中文显示
rcParams['font.sans-serif'] = ['SimHei']
rcParams['axes.unicode_minus'] = False


def load_data(comments_path, creators_path):
    print("加载评论数据和创作者数据...")
    comments_df = pd.read_json(comments_path)
    creators_df = pd.read_json(creators_path)
    return comments_df, creators_df


def add_sentiment_label(comments_df):
    print("开始进行情感分析...")
    comments_df = comments_df[comments_df['content'].notna()]  # 确保 'content' 列非空

    sentiment_labels = []
    sentiment_scores = []

    for content in comments_df['content']:
        content = str(content).strip()
        if not content:  # 跳过空内容
            sentiment_labels.append(None)
            sentiment_scores.append(None)
            continue
        try:
            s = SnowNLP(content)
            score = s.sentiments
            label = "LABEL_1" if score > 0.5 else "LABEL_0"
            sentiment_labels.append(label)
            sentiment_scores.append(score)
        except Exception as e:
            print(f"情感分析出错，跳过该内容: {content[:30]}... 错误: {e}")
            sentiment_labels.append(None)
            sentiment_scores.append(None)

    comments_df['sentiment_label'] = sentiment_labels
    comments_df['sentiment_score'] = sentiment_scores
    comments_df = comments_df.dropna(subset=['sentiment_label', 'sentiment_score'])  # 删除无效行
    print(f"有效评论数: {len(comments_df)}")
    return comments_df


def merge_data(comments_df, creators_df):
    print("\n关联评论数据与创作者数据...")
    merged_df = pd.merge(comments_df, creators_df, on='user_id', how='left')
    return merged_df


def classify_top_users(merged_df, fans_threshold=10000):
    print(f"\n基于粉丝数阈值 ({fans_threshold}) 进行头部用户分类...")
    merged_df['is_top_user'] = merged_df['total_fans'].apply(lambda x: "头部用户" if x >= fans_threshold else "普通用户")
    return merged_df


def analyze_sentiment(merged_df):
    sentiment_dist = merged_df.groupby(['is_top_user', 'sentiment_label']).size().reset_index(name='count')
    print(sentiment_dist)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=sentiment_dist, x='is_top_user', y='count', hue='sentiment_label')
    plt.title("头部用户与普通用户的情感分布")
    plt.xlabel("用户类型")
    plt.ylabel("评论数量")
    plt.tight_layout()
    plt.show()


def main():
    comments_path = r"D:/theguidetoculturaledconomic/数据/cleaned_comments.json"
    creators_path = r"D:/theguidetoculturaledconomic/数据/cleaned_creators.json"

    comments_df, creators_df = load_data(comments_path, creators_path)
    comments_df = add_sentiment_label(comments_df)
    merged_df = merge_data(comments_df, creators_df)
    merged_df = classify_top_users(merged_df, fans_threshold=10000)
    analyze_sentiment(merged_df)
    print("分析完成！")


if __name__ == "__main__":
    main()
