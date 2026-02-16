import pytest
import numpy as np
from array import array
from task2 import DGEMM_list, DGEMM_array, DGEMM_numpy


def test_small_matrix_list():
    
    A = [[1, 2, 3],
         [4, 5, 6],
         [7, 8, 9]]
    
    B = [[9, 8, 7],
         [6, 5, 4],
         [3, 2, 1]]
    
    C = [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]
    
    result = DGEMM_list(A, B, C, 3)
    
    # Check a few values manually
    assert result[0][0] == 30 
    assert result[0][1] == 24  
    assert result[1][0] == 84 


def test_small_matrix_array():
    # Same test but with arrays
    A = [array('i', [1, 2, 3]),
         array('i', [4, 5, 6]),
         array('i', [7, 8, 9])]
    
    B = [array('i', [9, 8, 7]),
         array('i', [6, 5, 4]),
         array('i', [3, 2, 1])]
    
    C = [array('i', [0, 0, 0]),
         array('i', [0, 0, 0]),
         array('i', [0, 0, 0])]
    
    result = DGEMM_array(A, B, C, 3)
    
    assert result[0][0] == 30
    assert result[0][1] == 24
    assert result[1][0] == 84


def test_small_matrix_numpy():
    A = np.array([[1, 2, 3],
                  [4, 5, 6],
                  [7, 8, 9]])
    
    B = np.array([[9, 8, 7],
                  [6, 5, 4],
                  [3, 2, 1]])
    
    C = np.zeros((3, 3), dtype=int)
    
    result = DGEMM_numpy(A, B, C)
    
    assert result[0, 0] == 30
    assert result[0, 1] == 24
    assert result[1, 0] == 84


def test_identity():
    A = [[2, 3],
         [4, 5]]
    
    I = [[1, 0],
         [0, 1]]
    
    C = [[0, 0],
         [0, 0]]
    
    result = DGEMM_list(A, I, C, 2)
    
    # A * I = A
    assert result[0][0] == 2
    assert result[0][1] == 3
    assert result[1][0] == 4
    assert result[1][1] == 5


def test_compare_list_and_numpy():
   
    A = [[1, 2],
         [3, 4]]
    
    B = [[5, 6],
         [7, 8]]
    
    C_list = [[0, 0],
              [0, 0]]
    
    result_list = DGEMM_list(A, B, C_list, 2)
    
    A_np = np.array(A)
    B_np = np.array(B)
    C_np = np.zeros((2, 2), dtype=int)
    
    result_numpy = DGEMM_numpy(A_np, B_np, C_np)
    
    assert result_list[0][0] == result_numpy[0, 0]
    assert result_list[0][1] == result_numpy[0, 1]
    assert result_list[1][0] == result_numpy[1, 0]
    assert result_list[1][1] == result_numpy[1, 1]
