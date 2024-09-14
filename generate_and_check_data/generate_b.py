import numpy as np

# 读取文件s.csv
print("读取s")
s = np.genfromtxt('../s2/s.csv', delimiter=',', dtype=int)
s128 = s[0, :128].reshape(-1, 1)
s256 = s[0, :256].reshape(-1, 1)
s512 = s[0, :512].reshape(-1, 1)
s1024 = s[0, :1024].reshape(-1, 1)
s2048 = s[0, :2048].reshape(-1, 1)
s4096 = s[0, :4096].reshape(-1, 1)

# 读取文件a100.csv
print("读取a")
a100 = np.genfromtxt('../a2/a100.csv', delimiter=',', dtype=int)

# 获取子矩阵
print("获取子矩阵")
A128 = a100[:, :128]
A256 = a100[:, :256]
A512 = a100[:, :512]
A1024 = a100[:, :1024]
# A2048 = a100[:, :2048]
# A4096 = a100[:, :4096]

# 读取列向量文件
print("读取e")
e1000dg = np.genfromtxt('../e2/e1000-dg-gpt.csv', delimiter=',', dtype=int, usecols=0)
e2000dg = np.genfromtxt('../e2/e2000-dg-gpt.csv', delimiter=',', dtype=int, usecols=0)
e3000dg = np.genfromtxt('../e2/e3000-dg-gpt.csv', delimiter=',', dtype=int, usecols=0)
e5000dg = np.genfromtxt('../e2/e5000-dg-gpt.csv', delimiter=',', dtype=int, usecols=0)

e1000uni = np.genfromtxt('../e2/e1000-uni-gpt.csv', delimiter=',', dtype=int, usecols=0)
e2000uni = np.genfromtxt('../e2/e2000-uni-gpt.csv', delimiter=',', dtype=int, usecols=0)
e3000uni = np.genfromtxt('../e2/e3000-uni-gpt.csv', delimiter=',', dtype=int, usecols=0)
e5000uni = np.genfromtxt('../e2/e5000-uni-gpt.csv', delimiter=',', dtype=int, usecols=0)

print("计算As128")
dot128 = A128.dot(s128)
print("计算As256")
dot256 = A256.dot(s256)
print("计算As512")
dot512 = A512.dot(s512)
print("计算As1024")
dot1024 = A1024.dot(s1024)
# print("计算As2048")
# dot2048 = A2048.dot(s2048)
# print("计算As4096")
# dot4096 = A4096.dot(s4096)

# 计算矩阵和向量的乘积以及加法
print("计算b")
b128_1000_dg = dot128 + e1000dg.reshape(-1, 1)
b128_2000_dg = dot128 + e2000dg.reshape(-1, 1)
b128_3000_dg = dot128 + e3000dg.reshape(-1, 1)
b128_5000_dg = dot128 + e5000dg.reshape(-1, 1)

b256_1000_dg = dot256 + e1000dg.reshape(-1, 1)
b256_2000_dg = dot256 + e2000dg.reshape(-1, 1)
b256_3000_dg = dot256 + e3000dg.reshape(-1, 1)
b256_5000_dg = dot256 + e5000dg.reshape(-1, 1)

b512_1000_dg = dot512 + e1000dg.reshape(-1, 1)
b512_2000_dg = dot512 + e2000dg.reshape(-1, 1)
b512_3000_dg = dot512 + e3000dg.reshape(-1, 1)
b512_5000_dg = dot512 + e5000dg.reshape(-1, 1)

b1024_1000_dg = dot1024 + e1000dg.reshape(-1, 1)
b1024_2000_dg = dot1024 + e2000dg.reshape(-1, 1)
b1024_3000_dg = dot1024 + e3000dg.reshape(-1, 1)
b1024_5000_dg = dot1024 + e5000dg.reshape(-1, 1)

# b2048_1000_dg = dot2048 + e1000dg.reshape(-1, 1)
# b2048_2000_dg = dot2048 + e2000dg.reshape(-1, 1)
# b2048_3000_dg = dot2048 + e3000dg.reshape(-1, 1)
# b2048_5000_dg = dot2048 + e5000dg.reshape(-1, 1)

# b4096_1000_dg = dot4096 + e1000dg.reshape(-1, 1)
# b4096_2000_dg = dot4096 + e2000dg.reshape(-1, 1)
# b4096_3000_dg = dot4096 + e3000dg.reshape(-1, 1)
# b4096_5000_dg = dot4096 + e5000dg.reshape(-1, 1)

b128_1000_uni = dot128 + e1000uni.reshape(-1, 1)
b128_2000_uni = dot128 + e2000uni.reshape(-1, 1)
b128_3000_uni = dot128 + e3000uni.reshape(-1, 1)
b128_5000_uni = dot128 + e5000uni.reshape(-1, 1)

b256_1000_uni = dot256 + e1000uni.reshape(-1, 1)
b256_2000_uni = dot256 + e2000uni.reshape(-1, 1)
b256_3000_uni = dot256 + e3000uni.reshape(-1, 1)
b256_5000_uni = dot256 + e5000uni.reshape(-1, 1)

b512_1000_uni = dot512 + e1000uni.reshape(-1, 1)
b512_2000_uni = dot512 + e2000uni.reshape(-1, 1)
b512_3000_uni = dot512 + e3000uni.reshape(-1, 1)
b512_5000_uni = dot512 + e5000uni.reshape(-1, 1)

b1024_1000_uni = dot1024 + e1000uni.reshape(-1, 1)
b1024_2000_uni = dot1024 + e2000uni.reshape(-1, 1)
b1024_3000_uni = dot1024 + e3000uni.reshape(-1, 1)
b1024_5000_uni = dot1024 + e5000uni.reshape(-1, 1)

# b2048_1000_uni = dot2048 + e1000uni.reshape(-1, 1)
# b2048_2000_uni = dot2048 + e2000uni.reshape(-1, 1)
# b2048_3000_uni = dot2048 + e3000uni.reshape(-1, 1)
# b2048_5000_uni = dot2048 + e5000uni.reshape(-1, 1)

# b4096_1000_uni = dot4096 + e1000uni.reshape(-1, 1)
# b4096_2000_uni = dot4096 + e2000uni.reshape(-1, 1)
# b4096_3000_uni = dot4096 + e3000uni.reshape(-1, 1)
# b4096_5000_uni = dot4096 + e5000uni.reshape(-1, 1)

# 将结果保存到文件b128-x000-y.csv中
print("写入b")
np.savetxt('../b2/b128-1000-dg.csv', b128_1000_dg, delimiter=',', fmt='%d')
np.savetxt('../b2/b128-2000-dg.csv', b128_2000_dg, delimiter=',', fmt='%d')
np.savetxt('../b2/b128-3000-dg.csv', b128_3000_dg, delimiter=',', fmt='%d')
np.savetxt('../b2/b128-5000-dg.csv', b128_5000_dg, delimiter=',', fmt='%d')

np.savetxt('../b2/b256-1000-dg.csv', b256_1000_dg, delimiter=',', fmt='%d')
np.savetxt('../b2/b256-2000-dg.csv', b256_2000_dg, delimiter=',', fmt='%d')
np.savetxt('../b2/b256-3000-dg.csv', b256_3000_dg, delimiter=',', fmt='%d')
np.savetxt('../b2/b256-5000-dg.csv', b256_5000_dg, delimiter=',', fmt='%d')

np.savetxt('../b2/b512-1000-dg.csv', b512_1000_dg, delimiter=',', fmt='%d')
np.savetxt('../b2/b512-2000-dg.csv', b512_2000_dg, delimiter=',', fmt='%d')
np.savetxt('../b2/b512-3000-dg.csv', b512_3000_dg, delimiter=',', fmt='%d')
np.savetxt('../b2/b512-5000-dg.csv', b512_5000_dg, delimiter=',', fmt='%d')

np.savetxt('../b2/b1024-1000-dg.csv', b1024_1000_dg, delimiter=',', fmt='%d')
np.savetxt('../b2/b1024-2000-dg.csv', b1024_2000_dg, delimiter=',', fmt='%d')
np.savetxt('../b2/b1024-3000-dg.csv', b1024_3000_dg, delimiter=',', fmt='%d')
np.savetxt('../b2/b1024-5000-dg.csv', b1024_5000_dg, delimiter=',', fmt='%d')

# np.savetxt('../b/b2048-1000-dg.csv', b2048_1000_dg, delimiter=',', fmt='%d')
# np.savetxt('../b/b2048-2000-dg.csv', b2048_2000_dg, delimiter=',', fmt='%d')
# np.savetxt('../b/b2048-3000-dg.csv', b2048_3000_dg, delimiter=',', fmt='%d')
# np.savetxt('../b/b2048-5000-dg.csv', b2048_5000_dg, delimiter=',', fmt='%d')

# np.savetxt('../b/b4096-1000-dg.csv', b4096_1000_dg, delimiter=',', fmt='%d')
# np.savetxt('../b/b4096-2000-dg.csv', b4096_2000_dg, delimiter=',', fmt='%d')
# np.savetxt('../b/b4096-3000-dg.csv', b4096_3000_dg, delimiter=',', fmt='%d')
# np.savetxt('../b/b4096-5000-dg.csv', b4096_5000_dg, delimiter=',', fmt='%d')

np.savetxt('../b2/b128-1000-uni.csv', b128_1000_uni, delimiter=',', fmt='%d')
np.savetxt('../b2/b128-2000-uni.csv', b128_2000_uni, delimiter=',', fmt='%d')
np.savetxt('../b2/b128-3000-uni.csv', b128_3000_uni, delimiter=',', fmt='%d')
np.savetxt('../b2/b128-5000-uni.csv', b128_5000_uni, delimiter=',', fmt='%d')

np.savetxt('../b2/b256-1000-uni.csv', b256_1000_uni, delimiter=',', fmt='%d')
np.savetxt('../b2/b256-2000-uni.csv', b256_2000_uni, delimiter=',', fmt='%d')
np.savetxt('../b2/b256-3000-uni.csv', b256_3000_uni, delimiter=',', fmt='%d')
np.savetxt('../b2/b256-5000-uni.csv', b256_5000_uni, delimiter=',', fmt='%d')

np.savetxt('../b2/b512-1000-uni.csv', b512_1000_uni, delimiter=',', fmt='%d')
np.savetxt('../b2/b512-2000-uni.csv', b512_2000_uni, delimiter=',', fmt='%d')
np.savetxt('../b2/b512-3000-uni.csv', b512_3000_uni, delimiter=',', fmt='%d')
np.savetxt('../b2/b512-5000-uni.csv', b512_5000_uni, delimiter=',', fmt='%d')

np.savetxt('../b2/b1024-1000-uni.csv', b1024_1000_uni, delimiter=',', fmt='%d')
np.savetxt('../b2/b1024-2000-uni.csv', b1024_2000_uni, delimiter=',', fmt='%d')
np.savetxt('../b2/b1024-3000-uni.csv', b1024_3000_uni, delimiter=',', fmt='%d')
np.savetxt('../b2/b1024-5000-uni.csv', b1024_5000_uni, delimiter=',', fmt='%d')

# np.savetxt('../b/b2048-1000-uni.csv', b2048_1000_uni, delimiter=',', fmt='%d')
# np.savetxt('../b/b2048-2000-uni.csv', b2048_2000_uni, delimiter=',', fmt='%d')
# np.savetxt('../b/b2048-3000-uni.csv', b2048_3000_uni, delimiter=',', fmt='%d')
# np.savetxt('../b/b2048-5000-uni.csv', b2048_5000_uni, delimiter=',', fmt='%d')

# np.savetxt('../b/b4096-1000-uni.csv', b4096_1000_uni, delimiter=',', fmt='%d')
# np.savetxt('../b/b4096-2000-uni.csv', b4096_2000_uni, delimiter=',', fmt='%d')
# np.savetxt('../b/b4096-3000-uni.csv', b4096_3000_uni, delimiter=',', fmt='%d')
# np.savetxt('../b/b4096-5000-uni.csv', b4096_5000_uni, delimiter=',', fmt='%d')
