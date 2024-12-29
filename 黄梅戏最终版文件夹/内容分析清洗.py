import json
import re
import pandas as pd
from datetime import datetime
import jieba







import datetime





# 读取 JSON 数据
input_file = r"D:\theguidetoculturaledconomic\MediaCrawler-main\MediaCrawler-main\data\bilibili\json\search_contents_2024-12-08.json"
output_file = r"D:\theguidetoculturaledconomic\数据\cleaned_video_data.json"


# 加载 JSON 文件
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


# 时间戳转为可读格式


# 时间戳转为可读格式，使用新的方式处理
def convert_timestamp(timestamp):
    # 使用datetime.fromtimestamp()并指定时区为UTC
    return datetime.datetime.fromtimestamp(timestamp, tz=datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S')


# 文本标准化：转小写，去除多余空格等
def normalize_text(text):
    text = text.lower()  # 转换为小写
    text = ' '.join(text.split())  # 去除多余的空格
    return text


# 加载原始数据
video_data = load_json(input_file)

# 清洗数据
cleaned_video_data = []
for video in video_data:
    cleaned_video = {}
    cleaned_video['video_id'] = video.get('video_id')
    cleaned_video['title'] = clean_text(video.get('title', ''))
    cleaned_video['desc'] = clean_text(video.get('desc', ''))
    cleaned_video['create_time'] = convert_timestamp(video.get('create_time'))
    cleaned_video['user_id'] = video.get('user_id')
    cleaned_video['nickname'] = video.get('nickname')
    cleaned_video['avatar'] = video.get('avatar')
    cleaned_video['liked_count'] = video.get('liked_count')
    cleaned_video['video_play_count'] = video.get('video_play_count')
    cleaned_video['video_danmaku'] = video.get('video_danmaku')
    cleaned_video['video_comment'] = video.get('video_comment')
    cleaned_video['last_modify_ts'] = video.get('last_modify_ts')
    cleaned_video['video_url'] = video.get('video_url')
    cleaned_video['video_cover_url'] = video.get('video_cover_url')
    cleaned_video['source_keyword'] = video.get('source_keyword')

    # 标准化文本（如果需要处理内容字段）
    cleaned_video['desc'] = normalize_text(cleaned_video['desc'])

    # 将处理后的视频数据加入列表
    cleaned_video_data.append(cleaned_video)

# 转换为 DataFrame 以便后续分析
df = pd.DataFrame(cleaned_video_data)

# 添加时间信息（可以按天、小时等进行分析）
df['create_time'] = pd.to_datetime(df['create_time'])
df['date'] = df['create_time'].dt.date
df['hour'] = df['create_time'].dt.hour
df['week'] = df['create_time'].dt.isocalendar().week  # 添加周信息

# 保存清洗后的数据
df.to_json(output_file, orient='records', force_ascii=False, indent=4)

# 打印出部分数据以验证清洗效果
print(df.head())
