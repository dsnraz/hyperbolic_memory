import json

# 打开文件
with open("hotpot_train_v1.1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 取第一个数据
first_item = data[0]  # 数组用这个
# first_item = next(iter(data.items()))  # 字典用这个

print(first_item)