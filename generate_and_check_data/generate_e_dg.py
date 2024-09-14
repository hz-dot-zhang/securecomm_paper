import numpy as np
import pandas as pd

# 生成服从离散高斯分布的随机整数
mean = 0
std_dev = 5000
num_samples = 300000

samples = np.random.normal(mean, std_dev, num_samples)
samples = np.round(samples).astype(int)

# 将整数写入CSV文件
df = pd.DataFrame(samples, columns=['Integer'])
df.to_csv('../e5/e5000-dg-gpt.csv', index=False, header=False)

print("生成的整数已写入文件 e5000-dg-gpt.csv")
