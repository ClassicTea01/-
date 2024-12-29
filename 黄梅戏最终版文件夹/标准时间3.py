import json
from datetime import datetime
from tqdm import tqdm  # 添加进度条显示

# 1. 输入和输出文件路径
input_file = r"D:\theguidetoculturaledconomic\数据\情感分析结果打分版本.json"
output_file = r"D:\theguidetoculturaledconomic\数据\情感分析结果_标准时间版.json"


# 2. 时间戳转换函数
def timestamp_to_date(timestamp, unit='s'):
    """
    将时间戳转换为 YYYY-MM-DD 格式。
    :param timestamp: 原始时间戳（秒级或毫秒级）
    :param unit: 单位，'s' 为秒，'ms' 为毫秒
    :return: 标准日期字符串 'YYYY-MM-DD'
    """
    try:
        if unit == 'ms':  # 毫秒级时间戳，先除以 1000
            timestamp = int(timestamp) / 1000
        else:  # 秒级时间戳
            timestamp = int(timestamp)
        return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error converting timestamp {timestamp}: {e}")
        return "1970-01-01"  # 错误时返回默认日期


# 3. 加载 JSON 文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 4. 处理 JSON 数据，转换时间戳
def process_data(data):
    for comment in tqdm(data, desc="Processing comments"):
        # 转换 "create_time" 和 "date"（秒级）
        if "create_time" in comment:
            comment["create_time"] = timestamp_to_date(comment["create_time"], unit='s')
        if "date" in comment:
            comment["date"] = timestamp_to_date(comment["date"], unit='s')

        # 转换 "last_modify_ts"（毫秒级）
        if "last_modify_ts" in comment:
            comment["last_modify_ts"] = timestamp_to_date(comment["last_modify_ts"], unit='ms')
    return data


# 5. 保存修改后的 JSON 数据
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Processed JSON saved to: {file_path}")


# 6. 主程序
if __name__ == '__main__':
    print("Loading input JSON file...")
    data = load_json(input_file)

    print("Converting timestamps to standard date format...")
    processed_data = process_data(data)

    print("Saving output JSON file...")
    save_json(processed_data, output_file)
    print("Processing completed successfully!")
