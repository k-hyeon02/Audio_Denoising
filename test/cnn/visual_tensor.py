import numpy as np

# visualization of tensor 
# (mxn) to (mn)

a = np.array([
    [
        [
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6]
        ],
        [
            [4, 5, 6, 7],
            [5, 6, 7, 8],
            [6, 7, 8, 9]
        ]
    ]
])

print(a, a.shape)       # (1, 2, 3, 4)

b = a.reshape((1, 2, 12))
print(b)
c = a.reshape((1, 24))
print(c)
d = a.reshape((1, 6, 4))
print(d)


# np.transpose

e = a.transpose(1, 2, 3, 0)
print(e, e.shape)
f = a.transpose(3, 1, 0, 2)
print(f, f.shape)