import pandas as pd
from gensim import corpora, models
import pyLDAvis
import pyLDAvis.gensim
import jieba
from tqdm import tqdm
import warnings
import os

warnings.filterwarnings("ignore")


# --- Step 1: 数据加载与预处理 ---
def load_and_preprocess_data(file_path):
    """
    加载 JSON 数据并提取评论内容，进行分词与预处理。
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"指定的文件路径不存在: {file_path}")

    print("\n加载评论数据...")
    df = pd.read_json(file_path)

    print("提取评论内容...")
    contents = df['content'].dropna().tolist()

    print("开始分词和停用词过滤...")
    stop_words = {'的', '了', '是', '呢', '啊', '吧', '都', '和', '着', '在', '也', '你', '我', '他', '她', '我们', '这', '那', '一个', '有', '说', '要', '到'}

    processed_texts = []
    for content in tqdm(contents, desc="处理文本"):
        try:
            tokens = jieba.cut(content)
            filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
            processed_texts.append(filtered_tokens)
        except Exception as e:
            print(f"处理文本时出错: {e}")
            continue

    print("分词与预处理完成！")
    return processed_texts


# --- Step 2: 构建词袋模型与LDA主题建模 ---
def lda_topic_modeling(processed_texts, num_topics=5, passes=10):
    """
    使用LDA对文本进行主题建模。
    """
    print("\n构建词袋模型...")
    dictionary = corpora.Dictionary(processed_texts)
    corpus = [dictionary.doc2bow(text) for text in processed_texts]

    print(f"训练LDA模型，主题数: {num_topics}, passes: {passes}...")
    lda_model = models.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=passes, random_state=42)

    print("LDA模型训练完成！")
    return lda_model, corpus, dictionary


# --- Step 3: 显示主题及关键词概率分布 ---
def display_topics(lda_model, num_topics=5, num_words=10):
    """
    输出每个主题及其关键词的概率分布。
    """
    print("\nLDA 主题提取结果：")
    for i, topic in lda_model.print_topics(num_topics=num_topics, num_words=num_words):
        print(f"主题 {i + 1}: {topic}")


# --- Step 4: 可视化主题分布 ---
def visualize_topics(lda_model, corpus, dictionary, output_path='lda_visualization.html'):
    """
    使用 pyLDAvis 可视化主题模型，并保存到本地 HTML 文件。
    """
    print("\n生成主题可视化，请稍候...")
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary)
    pyLDAvis.save_html(vis, output_path)  # 将可视化结果保存为 HTML 文件
    print(f"主题可视化结果已保存为: {output_path}")


# --- Step 5: 主函数 ---
def main():
    # 文件路径
    file_path = r"D:/theguidetoculturaledconomic/数据/情感分析结果打分版本.json"

    try:
        # Step 1: 加载与预处理数据
        processed_texts = load_and_preprocess_data(file_path)

        # Step 2: LDA建模
        num_topics = 8  # 设置提取的主题数
        lda_model, corpus, dictionary = lda_topic_modeling(processed_texts, num_topics=num_topics, passes=10)

        # Step 3: 显示主题及关键词概率分布
        display_topics(lda_model, num_topics=num_topics, num_words=10)

        # Step 4: 可视化主题并保存为 HTML
        output_file = r"D:/theguidetoculturaledconomic/数据/lda_visualization.html"
        visualize_topics(lda_model, corpus, dictionary, output_path=output_file)

        print("\nLDA主题建模分析完成！")

    except Exception as e:
        print(f"程序运行时发生错误: {e}")


# --- 执行主程序 ---
if __name__ == "__main__":
    main()
