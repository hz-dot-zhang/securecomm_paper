import csv

# 定义CSV文件路径
csv_file_path = '../a5/a100.csv'

# 用于存储整数的列表
all_integers = []

# 打开CSV文件
with open(csv_file_path, 'r') as file:
    # 创建CSV读取器
    csv_reader = csv.reader(file)

    # 读取前100行
    for row_number, row in enumerate(csv_reader):
        # 跳过标题行（如果有的话）
        if row_number == 0:
            continue

        # 将每个整数添加到列表中
        integers_in_row = [int(value) for value in row]
        all_integers.extend(integers_in_row)

        # 统计前100行
        if row_number == 9999:
            break

# 计算均值
average_value = sum(all_integers) / len(all_integers)

# 打印结果
print(f"前100行所有整数的均值为：{average_value}")
