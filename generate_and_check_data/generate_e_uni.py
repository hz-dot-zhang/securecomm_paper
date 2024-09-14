import numpy as np
import pandas as pd

# 生成服从均匀分布的随机整数
num_samples = 300000
sigma_e = 5000

samples = np.random.randint(low=int(-1 * 1.732 * sigma_e), high=int(1 * 1.732 * sigma_e) + 1, size=num_samples)

# 将整数写入CSV文件
df = pd.DataFrame(samples, columns=['Integer'])
df.to_csv('../e5/e5000-uni-gpt.csv', index=False, header=False)

print("生成的整数已写入文件 e5000-uni-gpt.csv")
