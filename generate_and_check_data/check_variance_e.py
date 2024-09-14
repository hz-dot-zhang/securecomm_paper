import pandas as pd
import numpy as np

# 读取CSV文件的前1万行数据
data = pd.read_csv('../e1/e3000-dg-gpt.csv', nrows=10000)

# 获取整数列数据
integers = data.iloc[:, 0]

# 计算方差
variance = np.var(integers)

# 输出方差值
print("方差值：", variance)
