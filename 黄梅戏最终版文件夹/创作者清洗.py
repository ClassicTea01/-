import json
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

# 数据清洗：去除无关数据和时间戳转换
def clean_data(input_file, output_file):
    """
    清洗数据：包括时间戳转换、去除无效字段等
    """
    # 加载原始数据
    creators_data = load_json(input_file)

    # 清洗数据并准备输出
    cleaned_creators = []
    for creator in creators_data:
        cleaned_creator = {}

        # 获取需要字段
        cleaned_creator['user_id'] = creator.get('user_id')
        cleaned_creator['nickname'] = creator.get('nickname')
        cleaned_creator['avatar'] = creator.get('avatar')
        cleaned_creator['total_fans'] = creator.get('total_fans')
        cleaned_creator['total_liked'] = creator.get('total_liked')
        cleaned_creator['user_rank'] = creator.get('user_rank')
        cleaned_creator['is_official'] = creator.get('is_official')

        # 处理时间戳
        cleaned_creator['last_modify_ts'] = convert_timestamp(creator.get('last_modify_ts'))

        # 将清洗后的数据加入列表
        cleaned_creators.append(cleaned_creator)

    # 转换为 Pandas DataFrame 以便分析
    df = pd.DataFrame(cleaned_creators)

    # 保存清洗后的数据到文件
    df.to_json(output_file, orient='records', force_ascii=False, indent=4)

    # 打印出部分数据以验证清洗效果
    print(df.head())

# 输入和输出文件路径
input_file = r"D:\theguidetoculturaledconomic\MediaCrawler-main\MediaCrawler-main\data\bilibili\json\search_creators_2024-12-08.json"
output_file = r"D:\theguidetoculturaledconomic\数据\cleaned_creators.json"

# 运行数据清洗程序
clean_data(input_file, output_file)
