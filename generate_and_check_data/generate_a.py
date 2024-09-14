import numpy as np
import csv

# 设置随机数生成参数
sigma_a = 100
num_cols = 1024
num_rows = 300000
batch_size = 10000  # 每次生成的行数

# 指定保存文件路径
csv_file_path = '../a5/a100.csv'

# 逐批次生成随机整数并保存为CSV文件
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    for iter in range(num_rows // batch_size):
        # 生成随机整数矩阵，服从均值为0、标准差为sigma_a的离散均匀分布，注意low和high的数学推导
        random_matrix = np.random.randint(low=int(-1*1.732*sigma_a), high=int(1*1.732*sigma_a)+1, size=(batch_size, num_cols))

        # 写入数据行
        for row in random_matrix:
            csv_writer.writerow(row)
        print("第{}/{}批写入完毕".format(iter+1, num_rows // batch_size))
