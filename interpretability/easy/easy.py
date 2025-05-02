from torch import Tensor

x = Tensor([1, 5])

w = Tensor([
    [2, 0, 3],
    [3, 5, 0],
])

b = Tensor([4, 1, 1])

y = x @ w + b

print(y)

