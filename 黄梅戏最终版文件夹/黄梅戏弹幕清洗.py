import pandas as pd

# 读取原始数据，并指定第一行为列名
file_path = r'D:\theguidetoculturaledconomic\数据\黄梅戏弹幕爬取.csv'
output_path = r'D:\theguidetoculturaledconomic\数据\清洗后的黄梅戏弹幕2.csv'

# 加载数据，指定第一行为列名
df = pd.read_csv(file_path, header=None, names=["弹幕ID", "弹幕内容", "时间", "显示位置", "用户ID"])

# 数据清洗
df["弹幕内容"] = df["弹幕内容"].str.strip().str.replace(r"['\"]", "", regex=True)  # 去除首尾的空格和引号

# 指定时间格式以避免警告
df["时间"] = pd.to_datetime(df["时间"], format="%Y-%m-%d %H:%M:%S", errors='coerce')  # 统一时间格式

# 处理显示位置，排除无法转换为float的非数值数据
df["显示位置"] = pd.to_numeric(df["显示位置"], errors='coerce')  # 将无法转换的值转为NaN

# 去除显示位置为NaN的行
df.dropna(subset=["显示位置"], inplace=True)

# 去除无效数据（如果时间为空的行）
df.dropna(subset=["时间"], inplace=True)

# 保存清洗后的数据
df.to_csv(output_path, index=False, encoding='utf-8')

print(f"清洗后的数据已保存到: {output_path}")
