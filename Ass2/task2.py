import random
import time
import sys
from array import array
import numpy as np


matrix_size = 512

def DGEMM_list(matrix_1, matrix_2, matrix_3, n):
    for i in range(n):
        for j in range(n):
            for k in range(n):
                 matrix_3[i][j] += matrix_1[i][k] * matrix_2[k][j]
    return matrix_3


def DGEMM_array(matrix_1, matrix_2, matrix_3, n):
    for i in range(n):
        for j in range(n):
            for k in range(n):
                 matrix_3[i][j] += matrix_1[i][k] * matrix_2[k][j]
    return matrix_3


def matrix_fill_list(matrix_size):
    matrix = [[random.randint(0, 100) for i in range(matrix_size)] for j in range(matrix_size)]
    return matrix


def matrix_fill_array(matrix_size, typecode='i'):
    matrix = []
    for j in range(matrix_size):
        row = array(typecode, (random.randint(0, 100) for i in range(matrix_size)))
        matrix.append(row)
    return matrix


def matrix_fill_numpy(matrix_size):
    return np.random.randint(0, 100, size=(matrix_size, matrix_size))


def DGEMM_numpy(matrix_1, matrix_2, matrix_3):
    matrix_3 += np.dot(matrix_1, matrix_2)
    return matrix_3


def main(argv):
    if len(argv) < 2:
        print("Usage: python task2.py <matrix_size>")
        return 1
    
    n = int(argv[1])
    print(f"Testing {n}x{n} matrix multiplication...\n")
    
    # Test with lists
    print("Building list matrices...")
    matrix_list_1 = matrix_fill_list(n)
    matrix_list_2 = matrix_fill_list(n)
    matrix_list_3 = [[0] * n for _ in range(n)]
    
    print("Running DGEMM_list...")
    start = time.time()
    DGEMM_list(matrix_list_1, matrix_list_2, matrix_list_3, n)
    list_time = time.time() - start
    
    # Test with arrays
    print("Building array matrices...")
    matrix_array_1 = matrix_fill_array(n)
    matrix_array_2 = matrix_fill_array(n)
    matrix_array_3 = [array('i', [0] * n) for _ in range(n)]
    
    print("Running DGEMM_array...")
    start = time.time()
    DGEMM_array(matrix_array_1, matrix_array_2, matrix_array_3, n)
    array_time = time.time() - start
    
    # Test with NumPy
    print("Building NumPy matrices...")
    matrix_numpy_1 = matrix_fill_numpy(n)
    matrix_numpy_2 = matrix_fill_numpy(n)
    matrix_numpy_3 = np.zeros((n, n), dtype=int)
    
    print("Running DGEMM_numpy...")
    start = time.time()
    DGEMM_numpy(matrix_numpy_1, matrix_numpy_2, matrix_numpy_3)
    numpy_time = time.time() - start
    
    print(f"\n{'='*50}")
    print(f"List time:  {list_time:.4f}s")
    print(f"Array time: {array_time:.4f}s")
    print(f"NumPy time: {numpy_time:.4f}s")
    print(f"{'='*50}")
    print(f"NumPy vs List:  {list_time/numpy_time:.1f}x faster")
    print(f"NumPy vs Array: {array_time/numpy_time:.1f}x faster")
    print(f"{'='*50}")
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv))

