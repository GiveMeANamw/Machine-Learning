from scipy import sparse
import numpy as np
# 创建二维对角矩阵
eye = np.eye(4)
print(eye)

# 稀疏矩阵的CSR格式表示
sparse_matrix = sparse.csc_matrix(eye)
print(sparse_matrix)

# arange()用于生成数组
#稀疏矩阵的COO格式表示
data = np.ones(4)   #np.ones()用以创建指定形式的数组，如果没有参数，则内部元素为float，可以为每一列指定类型，也可以指定行存储或者列存储
row_indices = np.arange(4)
col_indices = np.arange(4)
eye_coo = sparse.coo_matrix((data,(row_indices,col_indices)))
print(eye_coo)
