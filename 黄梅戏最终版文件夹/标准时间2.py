import json
from datetime import datetime

# 定义输入和输出文件路径
input_file_path = r'D:\theguidetoculturaledonomic\数据\情感分析结果打分版本.json'
output_file_path = r'D:\theguidetoculturaledonomic\数据\更新后的情感分析结果.json'

def timestamp_to_date(ts, millisecond=False):
    """将时间戳转换为'年-月-日'格式的字符串"""
    if millisecond:
        ts = ts / 1000  # 将毫秒转换为秒
    dt = datetime.fromtimestamp(ts)
    return dt.strftime('%Y-%m-%d')

def process_comments(comments):
    """遍历评论列表，更新时间戳为日期格式"""
    for comment in comments:
        # 转换create_time
        comment['create_time'] = timestamp_to_date(comment['create_time'])
        # 转换last_modify_ts，假定这是毫秒时间戳
        comment['last_modify_ts'] = timestamp_to_date(comment['last_modify_ts'], millisecond=True)
        # 转换date（如果与create_time相同，则无需再次转换）
        if 'date' in comment and comment['date'] != comment['create_time']:
            comment['date'] = timestamp_to_date(comment['date'])
    return comments

def main():
    with open(input_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 假设data是一个包含多个评论对象的列表
    updated_data = process_comments(data)

    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(updated_data, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    main()