M = Matrix(
    [1.0 2 3;
    4 5 6;
    7 8 9])
L = zeros(3, 3)

interface = (
    contentread = (
        k_r = [1.0, 2, 0],
        β_r = 5.0,
        k_w = [4.0, 0, 6],
        β_w = 5.0,
        erase = [1.0, 1, 1],
        add = [10.0, 20, 30],
        free = 1,
        alloc_gate = 0.0,
        alloc_write = 0.0,
        readmode = [0, 1, 0]
    ),
    contentwrite = (
        k_r = [1.0, 2, 0],
        β_r = 5.0,
        k_w = [4.0, 0, 6],
        β_w = 5.0,
        erase = [1.0, 1, 1],
        add = [10.0, 20, 30],
        free = 1,
        alloc_gate = 0.0,
        alloc_write = 1.0,
        readmode = [0, 1, 0]
    )
)
