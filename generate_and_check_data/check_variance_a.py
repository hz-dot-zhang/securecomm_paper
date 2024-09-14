import pandas as pd
import numpy as np

# 读取CSV文件的前50行数据
data = pd.read_csv('../a5/a100.csv', nrows=10000)

# 将整数数据展平为一维数组
integers = data.values.flatten()

# 计算方差
variance = np.var(integers)

# 输出方差值
print("方差值：", variance)
