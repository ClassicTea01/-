import json
import pandas as pd
import requests
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import os

# 0. 禁用符号链接警告（可选）
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

# 1. API设置
API_URL = "https://api-inference.huggingface.co/models/google-bert/bert-base-chinese"
headers = {"Authorization": "Bearer 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"}  # 更新为你的有效 token


def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()


# 2. 初始化VADER情感分析器
vader_analyzer = SentimentIntensityAnalyzer()

# 3. 读取评论数据
input_file = r"D:\theguidetoculturaledconomic\数据\cleaned_comments.json"
output_file = r"D:\theguidetoculturaledconomic\数据\情感分析结果.json"


# 读取原始JSON文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 4. 对长文本进行切分，确保每个文本片段的token数量不超过512
def split_text(text, tokenizer, max_length=512):
    # 使用tokenizer将文本转换为token列表
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = [tokens[i:i + max_length] for i in range(0, len(tokens), max_length)]
    chunk_texts = [tokenizer.decode(chunk, clean_up_tokenization_spaces=True) for chunk in chunks]
    return chunk_texts


# 5. 使用Hugging Face API进行情感分析
def analyze_sentiment_with_api(text, tokenizer):
    # 如果评论内容为空，直接返回默认值
    if not text.strip():
        return '未知', 0.0

    # 将长文本分割为多个小块
    chunks = split_text(text, tokenizer, max_length=512)

    sentiment_categories = []
    sentiment_scores = []

    for chunk in chunks:
        if not chunk.strip():
            continue
        try:
            # 使用API进行推理
            api_result = query({"inputs": chunk})
            sentiment_category = api_result[0].get('label', '未知')
            sentiment_score = api_result[0].get('score', 0.0)
        except Exception as e:
            print(f"Error processing chunk: {chunk}\nException: {e}")
            sentiment_category = '未知'
            sentiment_score = 0.0

        # 使用VADER进行情感强度评分
        vader_score = vader_analyzer.polarity_scores(chunk)
        sentiment_scores.append(vader_score['compound'])  # 添加情感分数
        sentiment_categories.append(sentiment_category)

    # 计算整体情感分数，避免多个小块导致的评分偏差
    if len(sentiment_scores) > 0:
        average_sentiment_score = sum(sentiment_scores) / len(sentiment_scores)
    else:
        average_sentiment_score = 0.0

    # 确定整体情感类别（多数投票）
    if sentiment_categories:
        sentiment_category = max(set(sentiment_categories), key=sentiment_categories.count)
    else:
        sentiment_category = '未知'

    return sentiment_category, average_sentiment_score


# 6. 处理每条评论并分析情感
comments_data = load_json(input_file)

# 7. 加载tokenizer
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-chinese")

# 清洗并分析评论数据
analyzed_comments = []
for comment in comments_data:
    content = comment.get('content', '')
    # 情感分析
    sentiment_category, sentiment_score = analyze_sentiment_with_api(content, tokenizer)

    # 将分析结果加入到评论字典
    analyzed_comment = {
        'comment_id': comment.get('comment_id'),
        'content': content,
        'create_time': comment.get('create_time'),
        'user_id': comment.get('user_id'),
        'nickname': comment.get('nickname'),
        'avatar': comment.get('avatar'),
        'sub_comment_count': comment.get('sub_comment_count'),
        'sentiment_category': sentiment_category,
        'sentiment_score': sentiment_score
    }
    analyzed_comments.append(analyzed_comment)

# 8. 保存分析结果到新的JSON文件
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(analyzed_comments, f, ensure_ascii=False, indent=4)

# 9. 打印分析结果的前几条数据
df = pd.DataFrame(analyzed_comments)
print(df.head())
