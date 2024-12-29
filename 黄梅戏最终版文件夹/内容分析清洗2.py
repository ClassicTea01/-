import pandas as pd
import json
import re
from datetime import datetime

# 加载数据
input_file = 'D:/theguidetoculturaledconomic/MediaCrawler-main/MediaCrawler-main/data/bilibili/json/search_contents_2024-12-08.json'
output_file = 'D:/theguidetoculturaledconomic/数据/cleaned_video_data.json'

# 读取JSON文件
with open(input_file, 'r', encoding='utf-8') as file:
    video_data = json.load(file)

# 数据清洗和处理
def clean_text(text):
    """清洗文本，移除网址、HTML标签、表情等"""
    # 去除网址
    text = re.sub(r'http\S+|www\S+', '', text)
    # 去除HTML标签
    text = re.sub(r'<.*?>', '', text)
    # 去除表情（表情符号通常由 Unicode 编码表示）
    text = re.sub(r'[^\w\s,.-]', '', text)
    # 去除多余空格
    text = ' '.join(text.split())
    return text

def timestamp_to_date(timestamp):
    """将毫秒时间戳转换为'年-月-日'格式"""
    # 由于时间戳是毫秒级别，需除以1000转换为秒级别
    timestamp = timestamp / 1000
    # 转换为日期格式
    return datetime.utcfromtimestamp(timestamp).strftime('%Y-%m-%d')

# 清洗数据
cleaned_data = []
for video in video_data:
    video_info = {}
    video_info['video_id'] = video.get('video_id')
    video_info['title'] = clean_text(video.get('title', ''))
    video_info['desc'] = clean_text(video.get('desc', ''))
    video_info['create_time'] = timestamp_to_date(video.get('create_time', 0))
    video_info['user_id'] = video.get('user_id')
    video_info['nickname'] = video.get('nickname', '')
    video_info['liked_count'] = int(video.get('liked_count', 0))
    video_info['video_play_count'] = int(video.get('video_play_count', 0))
    video_info['video_danmaku'] = int(video.get('video_danmaku', 0))
    video_info['video_comment'] = int(video.get('video_comment', 0))
    video_info['video_url'] = video.get('video_url', '')
    video_info['video_cover_url'] = video.get('video_cover_url', '')
    video_info['source_keyword'] = video.get('source_keyword', '')
    video_info['date'] = timestamp_to_date(video.get('date', 0))
    video_info['hour'] = video.get('hour', 0)
    video_info['week'] = video.get('week', 0)

    # 将处理后的数据添加到列表中
    cleaned_data.append(video_info)

# 将清洗后的数据保存为JSON文件
with open(output_file, 'w', encoding='utf-8') as outfile:
    json.dump(cleaned_data, outfile, ensure_ascii=False, indent=4)

# 输出清洗后的数据示例
df = pd.DataFrame(cleaned_data)
print(df.head())
