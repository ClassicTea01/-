import json
import re
import pandas as pd
from datetime import datetime, timezone

# 时间戳转日期函数
def convert_timestamp(timestamp):
    """
    将时间戳（毫秒级）转换为'年-月-日'格式，并返回
    """
    # 如果时间戳是毫秒级的，需要除以1000
    timestamp = timestamp / 1000  # 转换为秒级
    return datetime.fromtimestamp(timestamp, timezone.utc).strftime('%Y-%m-%d')  # 使用带时区支持的方式

# 读取 JSON 数据函数
def load_json(file_path):
    """
    加载 JSON 数据文件
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

# 数据清洗：去除网址、表情、HTML标签等
def clean_text(text):
    """
    清洗文本内容，去除网址、表情、HTML标签、多余空格等
    """
    # 去除网址
    text = re.sub(r'http[s]?://\S+', '', text)
    # 去除表情符号（Unicode范围）
    text = re.sub(r'[\U00010000-\U0010FFFF]', '', text)  # 去除所有Unicode表情符号
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 去除多余的空格
    text = ' '.join(text.split())
    return text

# 主函数：加载、清洗并保存数据
def clean_data(input_file, output_file):
    """
    主程序：读取原始数据，清洗数据并保存为JSON文件
    """
    # 加载原始数据
    comments_data = load_json(input_file)

    # 清洗数据并准备输出
    cleaned_comments = []
    for comment in comments_data:
        cleaned_comment = {}
        # 获取所需字段并清洗
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

    # 转换为 Pandas DataFrame 以便分析
    df = pd.DataFrame(cleaned_comments)

    # 时间处理：转换为日期格式，提取日期和小时
    df['create_time'] = pd.to_datetime(df['create_time'])
    df['date'] = df['create_time'].dt.date
    df['hour'] = df['create_time'].dt.hour

    # 保存清洗后的数据到文件
    df.to_json(output_file, orient='records', force_ascii=False, indent=4)

    # 打印出部分数据以验证清洗效果
    print(df.head())

# 输入和输出文件路径
input_file = r"D:\theguidetoculturaledconomic\MediaCrawler-main\MediaCrawler-main\data\bilibili\json\search_comments_2024-12-08.json"
output_file = r"D:\theguidetoculturaledconomic\数据\cleaned_comments_new2.json"

# 运行数据清洗程序
clean_data(input_file, output_file)
