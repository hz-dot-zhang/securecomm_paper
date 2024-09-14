# 使用最大似然估计
# 通过Batch Gradient Descent来计算参数，而且随机选择样本(不是用前num_sample个样本)
import numpy as np
import csv
import time

num_sample = 512000

def batch_gradient_descent(X, y, learning_rate=1, num_iterations=100, parameter_change_threshold=1e-2):
    m, n = X.shape
    theta = np.zeros(n)  # 初始化参数向量
    prev_theta = np.copy(theta)  # 初始化前一次迭代的参数向量

    for i in range(num_iterations):
        # 计算预测值
        y_pred = np.dot(X, theta)
        
        # 计算误差
        error = y_pred - y
        
        # 计算梯度
        gradient = (2/m) * np.dot(X.T, error)

        # 计算参数变化量
        parameter_change = learning_rate * gradient

        # 判断参数变化是否小于阈值
        if np.all(np.abs(parameter_change) < parameter_change_threshold):
            print(f"Converged after {i} iterations.")
            break

        # 更新参数
        theta = theta - parameter_change

        prev_theta = np.copy(theta)  # 更新前一次迭代的参数向量

    return theta

# 读取s.csv文件
print("读取s开始")
# 初始化向量s256
s256 = []

# 读取s.csv文件
with open('./s.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for i, row in enumerate(csv_reader):
        # 读取第一行
        if i == 0:
            # 获取前256个整数
            s256 = list(map(int, row[:256]))
            break
print(s256)
print("读取s完成")

# 读取A.csv文件
print("读取A开始")
X_all = []
with open('./A.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        row = list(map(int, row[:256]))  # 转换为整数并截取前256维
        X_all.append(row)

# 读取b.csv文件
print("读取b开始")
y_all = []
with open('./b.csv', 'r') as file:
    csv_reader = csv.reader(file)
    for row in csv_reader:
        y_all.append(int(row[0]))

# 随机选择num_sample个样本
indices = np.random.choice(len(X_all), num_sample, replace=False)

# 使用选择的样本索引从X_all和y_all中抽取样本
X = np.array([X_all[i] for i in indices])
y = np.array([y_all[i] for i in indices])

print("读取A和b完成")

s = s256

start_time = time.time()
theta = batch_gradient_descent(X, y)
end_time = time.time()
elapsed_time = end_time - start_time

print("求解时间：", elapsed_time, "秒")
# 输出模型参数
print("Coefficients:", theta)

# check whether recover successful
# coef = model.coef_ # s_tilde in R^n, before rounding
coef = theta
diff = coef - s
recovered = True
n = len(s)
for i in range(n):
    if(diff[i] >= 0.5 or diff[i] <= -0.5):
        recovered = False
        print(diff[i])
if(recovered):
    print("Success")
    s_tilde_rounded = [round(elem) for elem in coef]
    # print("The secret s is:", s_tilde_rounded)
    # set exit code to indicate whether the recover succeeded
    exit(0)
else:
    print("Failure")
    # set exit code to indicate whether the recover succeeded
    exit(1)
