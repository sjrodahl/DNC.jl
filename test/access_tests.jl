M = Matrix(
    [1.0 2 3;
    -1 -2  -3;
    1 -2 3])

interface = (
    contentread = (
        k_r = [1.0, 2, 0],
        β_r = 100.0,
        k_w = [4.0, 0, 6],
        β_w = 5.0,
        erase = [1.0, 1, 1],
        add = [10.0, 20, 30],
        free = 1.0,
        alloc_gate = 0.0,
        write_gate = 0.0,
        readmode = [0.0, 1, 0]
    ),
    contentwrite = (
        k_r = [1.0, 2, 0],
        β_r = 10.0,
        k_w = [-1.0, -2.0, -3.0],
        β_w = 100.0,
        erase = [1.0, 1, 1],
        add = [10.0, 20, 30],
        free = 1.0,
        alloc_gate = 0.0,
        write_gate = 1.0,
        readmode = [0.0, 1, 0]
    )
)

state = State(
    Matrix(zeros(3, 3)),
    zeros(3),
    zeros(3),
    [0.0, 0, 1],
    [1.0, 0, 0]
)

@testset "Sharp read/write" begin
    @testset "Read" begin
        r = readmem(M, interface.contentread, state)
        @test r == M[1,:]
    end
    @testset "Write" begin
        writemem!(M, interface.contentwrite, state)
        @test M[2,:] == interface.contentwrite.add
    end
end
