import json
from datetime import datetime

# 1. 输入和输出文件路径
input_file = r"D:\theguidetoculturaledconomic\数据\情感分析结果打分版本.json"
output_file = r"D:\theguidetoculturaledconomic\数据\情感分析结果_标准时间版.json"


# 2. 时间戳转换为 "YYYY-MM-DD" 格式的函数
def timestamp_to_date(timestamp):
    # 确保时间戳是合法的整数
    try:
        return datetime.fromtimestamp(int(timestamp)).strftime('%Y-%m-%d')
    except Exception as e:
        print(f"Error converting timestamp {timestamp}: {e}")
        return "1970-01-01"  # 如果出错，返回默认日期


# 3. 加载 JSON 文件
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 4. 处理数据并修改时间戳
def process_data(data):
    for comment in data:
        # 检查并转换 "create_time" 和 "date"
        if "create_time" in comment:
            comment["create_time"] = timestamp_to_date(comment["create_time"])
        if "date" in comment:
            comment["date"] = timestamp_to_date(comment["date"])
        # 处理其他时间戳字段（如 last_modify_ts），若有需要请解除注释
        # if "last_modify_ts" in comment:
        #     comment["last_modify_ts"] = timestamp_to_date(comment["last_modify_ts"] // 1000)  # 毫秒时间戳处理
    return data


# 5. 保存修改后的数据
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print(f"Converted JSON file saved to: {file_path}")


# 6. 主程序
if __name__ == '__main__':
    print("Loading input JSON file...")
    data = load_json(input_file)

    print("Converting timestamps to standard date format...")
    processed_data = process_data(data)

    print("Saving output JSON file...")
    save_json(processed_data, output_file)
    print("Processing completed successfully!")
