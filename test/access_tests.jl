using Zygote

M = Matrix(
    [1.0 2 3;
    -1 -2  -3;
    1 -2 3])

# Content-based read
contentread = ReadHead(
    k = [1.0, 2.0, 0.0],
    β = 100.0,
    f = 1.0,
    π = [0.0, 1.0, 0.0]
    )

# Temporal linkage based read
backwardread = ReadHead(
    k = [-1.0, -2.0, -5.0],
    β = 100.0,
    f = 0.0,
    π = [1.0, 0.0, 0.0]
    )

forwardread = ReadHead(
    k = [-1.0, -2.0, -5.0],
    β = 100.0,
    f = 0.0,
    π = [0.0, 0.0, 1.0]
    )

contentwrite = WriteHead(
    k = [1.0, 2, 0],
    β = 10.0,
    e = [1.0, 1.0, 1.0],
    v = [10.0, 20.0, 30.0],
    g_a = 0.0,
    g_w = 1.0
)

allocationwrite = WriteHead(
    k = [1.0, 1.0, 1.0],
    β = 1.0,
    e = [1.0, 1.0, 1.0],
    v = [10.0, 20.0, 30.0],
    g_a = 1.0,
    g_w = 1.0
)

state = State(
    # Write history is 1 -> 2 -> 3
    L = [0.0 0.0 0.0;
         1.0 0.0 0.0;
         0.0 1.0 0.0],
    p = zeros(3),
    u = [1.0, 1.0, 0.0], # 3 is (artificially) set to unused/ free to be allocated
    w_w = [0.0, 1.0, 0],
    w_r = [[0.0, 1.0, 0.0]] # last read was 2, so forward points to 3, back to 1
)

@testset "Sharp read/write" begin
    @testset "Read" begin
        L, wr = state.L, state.wr
        r1 = readmem(M, contentread, L, wr[1])
        g = gradient(contentread) do rh
            sum(readmem(M, rh, L, w_r[1]))
        end
        @test r1 == M[1,:]
        r2 = readmem(M, backwardread, L, w_r[1])
        g = gradient(backwardread) do rh
            sum(readmem(M, rh, L, w_r[1]))
        end
        @test r2 == M[1, :]
        r3 = readmem(M, forwardread, L, w_r[1])
        g = gradient(forwardread) do rh
            sum(readmem(M, rh, L, w_r[1]))
        end
        @test r3 == M[3, :]
    end
    @testset "Write" begin
        ww, wr, u = state.ww, state.wr, state.u
        newM = writemem(M, contentwrite, [contentread.f], ww, wr, u)
        g = gradient(M, contentwrite, [contentread.f], ww, wr, u) do M, wh, rh, ww, wr, u
            sum(writemem(M, wh, rh, ww, wr, u))
        end
        @test isapprox(newM[1,:], contentwrite.v, atol=1e-2)
        newM2 = writemem(M, allocationwrite, [contentread.f], w_w, w_r, u)
        @test isapprox(newM2[3,:], contentwrite.v, atol=1e-2)
    end
end

@testset "Gradients" begin
    @testset "Read gradient" begin
        L, wr = state.L, state.wr
        r1 = readmem(M, contentread, L, wr[1])
        function f(M, contentread, L, wr)
            sum(readmem(M, contentread, L, wr))
        end
        readmem_g = gradient(f, M, contentread, L, w_r[1])
        @test true
    end
end

@testset "Erase and add" begin
    new = DNC.erase_and_add(M, [0.0, 0, 1], ones(3), [2.0, 2.0, 2.0])
    @test new[3,:] == [2.0, 2.0, 2.0]
end
