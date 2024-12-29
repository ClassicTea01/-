import json
from transformers import pipeline
from tqdm import tqdm
import os

# 1. 禁用符号链接警告（可选）
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# 2. 定义输入和输出文件路径
input_file = r"D:\theguidetoculturaledconomic\数据\cleaned_comments.json"
output_file = r"D:\theguidetoculturaledconomic\数据\情感分析结果2.json"

# 3. 加载 Hugging Face 中文情感分析模型
def load_classifier():
    try:
        print("Loading BERT-based sentiment analysis model...")
        classifier = pipeline("sentiment-analysis", model="bert-base-chinese")
        print("Model loaded successfully.")
        return classifier
    except Exception as e:
        print(f"Error loading model: {e}")
        exit(1)

# 4. 加载 JSON 数据
def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        exit(1)

# 5. 处理评论文本（确保不超过 512 tokens）
def process_comments(comments, classifier):
    results = []
    for comment in tqdm(comments, desc="Processing comments", ncols=100):
        text = comment.get('content', '')  # 使用 'content' 键
        if not text.strip():  # 跳过空文本
            continue

        # 截断文本，避免超过 BERT 最大长度限制
        text = text[:512]
        try:
            # 执行情感分类
            result = classifier(text)
            sentiment_label = result[0]['label']
            sentiment_score = result[0]['score']
        except Exception as e:
            print(f"Error processing comment: {e}")
            sentiment_label = 'unknown'
            sentiment_score = 0.0

        # 将分析结果保存到原始评论数据中
        comment['sentiment_label'] = sentiment_label
        comment['sentiment_score'] = sentiment_score
        results.append(comment)
    return results

# 6. 保存分析结果到 JSON 文件
def save_json(data, file_path):
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"Sentiment analysis results saved to: {file_path}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")
        exit(1)

# 7. 主函数（使用 Windows 多进程安全保护）
if __name__ == '__main__':
    # 加载模型
    classifier = load_classifier()

    # 加载输入数据
    print("Loading input data...")
    comments_data = load_json(input_file)

    # 执行情感分析
    print("Starting sentiment analysis...")
    analyzed_comments = process_comments(comments_data, classifier)

    # 保存结果
    save_json(analyzed_comments, output_file)
    print("Sentiment analysis completed successfully!")
