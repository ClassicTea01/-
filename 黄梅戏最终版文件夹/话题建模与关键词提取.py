import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from sklearn.cluster import KMeans
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from transformers import BertTokenizer, BertModel
from sklearn.metrics.pairwise import cosine_similarity
import torch
import os

# --- Step 1: 数据加载 ---
def load_data(file_path):
    """
    加载评论数据文件。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件不存在: {file_path}")
    print("加载评论数据...")
    data = pd.read_json(file_path)
    # 提取文本列
    comments = data['content'].dropna().tolist()
    return comments, data

# --- Step 2: TF-IDF关键词提取 ---
def extract_keywords_tfidf(comments, top_n=10):
    """
    使用TF-IDF提取关键词。
    """
    print("\n使用TF-IDF提取关键词...")
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, stop_words='english', max_features=1000)
    tfidf_matrix = vectorizer.fit_transform(comments)
    feature_names = vectorizer.get_feature_names_out()

    # 提取关键词
    keywords = []
    for row in tfidf_matrix:
        row_data = row.toarray().flatten()
        top_indices = row_data.argsort()[-top_n:][::-1]
        top_keywords = [feature_names[i] for i in top_indices]
        keywords.append(top_keywords)
    return keywords

# --- Step 3: BERT嵌入与KMeans聚类 ---
def extract_topics_with_bert(comments, n_clusters=5):
    """
    使用BERT嵌入生成评论的向量表示，并使用KMeans进行聚类提取主题。
    """
    print("\n使用BERT嵌入和KMeans提取话题...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    model = BertModel.from_pretrained('bert-base-chinese')

    embeddings = []
    for comment in comments:
        inputs = tokenizer(comment, return_tensors='pt', truncation=True, max_length=128, padding='max_length')
        with torch.no_grad():
            outputs = model(**inputs)
        # 获取句向量（CLS token表示）
        embeddings.append(outputs.last_hidden_state[:, 0, :].numpy().flatten())

    # KMeans聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(embeddings)

    # 提取每个簇的主题关键词
    clustered_comments = {i: [] for i in range(n_clusters)}
    for i, cluster in enumerate(clusters):
        clustered_comments[cluster].append(comments[i])

    return clustered_comments

# --- Step 4: LDA主题建模 ---
def lda_topic_modeling(comments, num_topics=5):
    """
    使用LDA进行主题建模。
    """
    print("\n使用LDA进行话题建模...")
    # 文本预处理
    tokenized_comments = [comment.split() for comment in comments]  # 假设中文分词已完成
    dictionary = Dictionary(tokenized_comments)
    corpus = [dictionary.doc2bow(text) for text in tokenized_comments]

    # LDA建模
    lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)
    topics = lda.print_topics()
    return topics

# --- Step 5: NMF主题建模 ---
def nmf_topic_modeling(comments, num_topics=5, top_n=10):
    """
    使用NMF进行主题建模。
    """
    print("\n使用NMF进行话题建模...")
    vectorizer = TfidfVectorizer(max_df=0.85, min_df=2, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(comments)
    feature_names = vectorizer.get_feature_names_out()

    nmf_model = NMF(n_components=num_topics, random_state=42)
    W = nmf_model.fit_transform(tfidf_matrix)
    H = nmf_model.components_

    topics = {}
    for i, topic in enumerate(H):
        top_keywords = [feature_names[j] for j in topic.argsort()[-top_n:][::-1]]
        topics[f"Topic {i+1}"] = top_keywords
    return topics

# --- Step 6: 结果结合与输出 ---
def analyze_with_source_keywords(topics, source_keywords):
    """
    结合source_keywords和主题建模结果，分析常见话题。
    """
    print("\n结合source_keyword分析...")
    source_keywords_set = set(source_keywords.split())
    matched_topics = {}
    for topic, keywords in topics.items():
        matched = source_keywords_set.intersection(set(keywords))
        matched_topics[topic] = matched
    return matched_topics

# --- Step 7: 主函数 ---
def main():
    # 文件路径
    comment_file = r"D:/theguidetoculturaledconomic/数据/情感分析结果打分版本.json"
    source_keywords = "黄梅戏 音频分开录制 虚拟背景 数字化创新"

    # 加载评论数据
    comments, data = load_data(comment_file)

    # Step 2: TF-IDF关键词提取
    tfidf_keywords = extract_keywords_tfidf(comments, top_n=5)
    print("\nTF-IDF提取的关键词示例:")
    for i, kw in enumerate(tfidf_keywords[:5]):
        print(f"评论 {i+1}: {kw}")

    # Step 3: BERT嵌入与KMeans聚类
    clustered_comments = extract_topics_with_bert(comments, n_clusters=3)
    print("\nBERT聚类话题示例:")
    for cluster, texts in clustered_comments.items():
        print(f"\nCluster {cluster}:")
        print("\n".join(texts[:2]))  # 显示每个簇的2条示例

    # Step 4: LDA话题建模
    lda_topics = lda_topic_modeling(comments, num_topics=3)
    print("\nLDA 话题建模结果:")
    for topic in lda_topics:
        print(topic)

    # Step 5: NMF话题建模
    nmf_topics = nmf_topic_modeling(comments, num_topics=3)
    print("\nNMF 话题建模结果:")
    for topic, keywords in nmf_topics.items():
        print(f"{topic}: {keywords}")

    # Step 6: 结合source_keywords
    matched_topics = analyze_with_source_keywords(nmf_topics, source_keywords)
    print("\n与source_keywords匹配的关键词:")
    for topic, matched in matched_topics.items():
        print(f"{topic}: {matched}")

if __name__ == "__main__":
    main()
