import pandas as pd
from gensim import corpora, models
from nltk.corpus import stopwords
import jieba
import pyLDAvis.gensim_models
import pyLDAvis
import nltk
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# --- Step 1: 数据加载与预处理 ---
def load_and_preprocess_data(file_path):
    """
    加载 JSON 数据并提取评论内容，进行分词与预处理。
    """
    print("加载评论数据...")
    df = pd.read_json(file_path)

    print("提取评论内容...")
    contents = df['content'].dropna().tolist()

    print("开始分词和停用词过滤...")
    # 下载NLTK停用词（仅首次需要）
    nltk.download('stopwords')
    stop_words = set(stopwords.words('chinese')) if 'chinese' in stopwords.fileids() else set()
    custom_stop_words = set(["的", "了", "是", "啊", "吧", "都", "和", "着", "就", "呢", "还有",
                             "这个", "那个", "什么", "这么", "这样", "真的", "哈哈", "喜欢",
                             "就是", "可以", "不会", "不是", "没有", "这样", "已经","doge", "吃瓜", "知道"])
    stop_words.update(custom_stop_words)

    processed_texts = []
    for content in tqdm(contents, desc="处理文本"):
        tokens = jieba.cut(content)  # 使用 jieba 进行中文分词
        filtered_tokens = [word for word in tokens if word not in stop_words and len(word) > 1]
        processed_texts.append(filtered_tokens)

    print("分词与预处理完成！")
    return processed_texts


# --- Step 2: 构建LDA主题模型 ---
def lda_topic_modeling(processed_texts, num_topics=5, passes=15):
    """
    使用LDA对文本进行主题建模。
    """
    print(f"构建词袋模型和训练LDA，主题数: {num_topics}, passes: {passes}...")
    dictionary = corpora.Dictionary(processed_texts)  # 构建字典
    corpus = [dictionary.doc2bow(text) for text in processed_texts]  # 构建词袋模型

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
        print(f"主题 {i+1}: {topic}")


# --- Step 4: 生成主题可视化 ---
def visualize_topics(lda_model, corpus, dictionary):
    """
    使用 pyLDAvis 进行主题模型可视化。
    """
    print("\n生成主题可视化，请稍候...")
    vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, dictionary)
    output_file = 'LDA_Visualization.html'
    pyLDAvis.save_html(vis, output_file)
    print(f"主题可视化已保存为 {output_file}，请使用浏览器打开查看。")


# --- Step 5: 主函数 ---
def main():
    # 文件路径
    file_path = r"D:/theguidetoculturaledconomic/数据/情感分析结果打分版本.json"

    # 加载与预处理数据
    processed_texts = load_and_preprocess_data(file_path)

    # LDA建模
    num_topics = 10  # 设置提取的主题数
    lda_model, corpus, dictionary = lda_topic_modeling(processed_texts, num_topics=num_topics, passes=15)

    # 显示主题及其关键词概率分布
    display_topics(lda_model, num_topics=num_topics, num_words=10)

    # 可视化主题
    visualize_topics(lda_model, corpus, dictionary)


# --- 执行主程序 ---
if __name__ == "__main__":
    main()
