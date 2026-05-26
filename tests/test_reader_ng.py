import numpy as np
from reader_ng import find_maximum   # adjust import name

def test_find_maximum():
    arr = np.array([1, 3, 2, 5, 4])
    val, idx = find_maximum(arr)
    assert val == 5
    assert idx == 3

def test_find_maximum_with_limits():
    arr = np.array([1, 3, 5, 2, 4])
    val, idx = find_maximum(arr, start=1, end=4)  # indices 1,2,3 → values 3,5,2
    assert val == 5
    assert idx == 2