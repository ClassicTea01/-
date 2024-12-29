import json
import re
import pandas as pd

import jieba
from datetime import datetime, timezone

def convert_timestamp(timestamp):
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


# 读取 JSON 数据
input_file = r"D:\theguidetoculturaledconomic\MediaCrawler-main\MediaCrawler-main\data\bilibili\json\search_comments_2024-12-08.json"
output_file = r"D:\theguidetoculturaledconomic\数据\cleaned_comments.json"

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 数据清洗：去除网址、表情、HTML标签等
def clean_text(text):
    # 去除网址
    text = re.sub(r'http[s]?://\S+', '', text)
    # 去除表情符号
    text = re.sub(r'[\U00010000-\U0010FFFF]', '', text)  # 去除所有Unicode表情符号
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 去除多余的空格
    text = ' '.join(text.split())
    return text

# 加载原始数据
comments_data = load_json(input_file)

# 清洗数据
cleaned_comments = []
for comment in comments_data:
    cleaned_comment = {}
    cleaned_comment['comment_id'] = comment.get('comment_id')
    cleaned_comment['content'] = clean_text(comment.get('content', ''))
    cleaned_comment['create_time'] = convert_timestamp(comment.get('create_time'))
    cleaned_comment['user_id'] = comment.get('user_id')
    cleaned_comment['nickname'] = comment.get('nickname')
    cleaned_comment['avatar'] = comment.get('avatar')
    cleaned_comment['sub_comment_count'] = comment.get('sub_comment_count')
    cleaned_comment['last_modify_ts'] = comment.get('last_modify_ts')

    # 将处理后的评论加入列表
    cleaned_comments.append(cleaned_comment)

# 转换为 DataFrame 以便后续分析
df = pd.DataFrame(cleaned_comments)

# 添加时间信息（可以按天、小时等进行分析）
df['create_time'] = pd.to_datetime(df['create_time'])
df['date'] = df['create_time'].dt.date
df['hour'] = df['create_time'].dt.hour

# 保存清洗后的数据
df.to_json(output_file, orient='records', force_ascii=False, indent=4)

# 打印出部分数据以验证清洗效果
print(df.head())
