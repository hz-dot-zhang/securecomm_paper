# normal equation和batch gradient descent求解内存占用对比

import numpy as np
import csv
import time

def normal_equation(X, y):
    # 计算 (X 转置乘以 X) 求逆
    X_transpose = np.transpose(X)
    X_transpose_X = np.dot(X_transpose, X)
    X_transpose_X_inv = np.linalg.inv(X_transpose_X)
    # 计算 X 转置乘以 y
    X_transpose_y = np.dot(X_transpose, y)
    # 计算参数
    return np.dot(X_transpose_X_inv, X_transpose_y)

def batch_gradient_descent(X, y, learning_rate=1e-5, num_iterations=1000, parameter_change_threshold=0.01):
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

@profile
def memory_compare():
    num_sample = 90000

    # 读取s.csv文件
    print("读取s开始")
    # 初始化向量s1024
    s1024 = []

    # 读取s.csv文件
    with open('./s5/s.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for i, row in enumerate(csv_reader):
            # 读取第一行
            if i == 0:
                # 获取前1024个整数
                s1024 = list(map(int, row[:1024]))
                break
    print(s1024)
    print("读取s完成")

    # 读取a100.csv文件
    print("读取A开始")
    read_start_time = time.time()
    A = []
    with open('./a5/a100.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for i, row in enumerate(csv_reader):
            if i < num_sample:
                row = list(map(int, row[:1024]))  # 转换为整数并截取前1024维
                A.append(row)
    # 转换为numpy数组
    A = np.array(A)
    read_end_time = time.time()
    read_elapsed_time = read_end_time - read_start_time
    print("读取A并转化为numpy数组完成")
    print("花费时间：", read_elapsed_time, "秒")

    # 读取b1024-5000-dg.csv文件
    print("读取b开始")
    read_start_time = time.time()
    b = []
    with open('./b5/b1024-5000-dg.csv', 'r') as file:
        csv_reader = csv.reader(file)
        for i, row in enumerate(csv_reader):
            if i < num_sample:
                b.append(int(row[0]))
    print("读取b完成")
    # 转换为numpy数组
    b = np.array(b)
    read_end_time = time.time()
    read_elapsed_time = read_end_time - read_start_time
    print("读取b并转化为numpy数组完成")
    print("花费时间：", read_elapsed_time, "秒")

    s = s1024
    X = A
    y = b

    # 使用我自己实现的最基本的Normal Equation运算求解参数
    start_time = time.time()
    theta_best = normal_equation(X, y)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(num_sample, "个sample的求解时间：", elapsed_time, "秒")

    # check whether recover successful
    coef = theta_best  # s_tilde in R^n, before rounding
    diff = coef - s
    recovered = True
    n = len(s)
    for i in range(n):
        if(diff[i] >= 0.5 or diff[i] <= -0.5):
            recovered = False
            print(diff[i])
    if(recovered):
        print("Normal Equation solve Success")
        s_tilde_rounded = [round(elem) for elem in coef]
        # print("The secret s is:", s_tilde_rounded)
        # set exit code to indicate whether the recover succeeded
    else:
        print("Normal Equation solve Failure")
        # set exit code to indicate whether the recover succeeded
        # exit(1)

    # 使用Batch Gradient Descent运算求解参数
    start_time = time.time()
    theta_best = batch_gradient_descent(X, y)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(num_sample, "个sample的求解时间：", elapsed_time, "秒")

    # check whether recover successful
    coef = theta_best  # s_tilde in R^n, before rounding
    diff = coef - s
    recovered = True
    n = len(s)
    for i in range(n):
        if(diff[i] >= 0.5 or diff[i] <= -0.5):
            recovered = False
            print(diff[i])
    if(recovered):
        print("Batch Gradient Descent solve Success")
        s_tilde_rounded = [round(elem) for elem in coef]
        # print("The secret s is:", s_tilde_rounded)
        # set exit code to indicate whether the recover succeeded
    else:
        print("Batch Gradient Descent solve Failure")
        # set exit code to indicate whether the recover succeeded
        # exit(1)

if __name__=='__main__':
    memory_compare()