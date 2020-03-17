using Zygote

N = 3
W = 3

M = Matrix(
    [1.0 2 3;
    -1 -2  -3;
    1 -2 3])

# Content-based read
contentread = (
    kr = Matrix([1.0 2.0 0.0]'),
    βr = [100.0],
    f = [1.0],
    readmode = Matrix([0.0 1.0 0.0]')
    )

# Temporal linkage based read
backwardread = (
    kr = Matrix([-1.0 -2.0 -5.0]'),
    βr = 100.0,
    f = 0.0,
    readmode = Matrix([1.0 0.0 0.0]')
    )

forwardread = (
    kr = Matrix([-1.0 -2.0 -5.0]'),
    βr = 100.0,
    f = 0.0,
    readmode = Matrix([0.0 0.0 1.0]')
    )

contentwrite = (
    kw = Matrix([1.0 2 0]'),
    βw = 10.0,
    e = [1.0, 1.0, 1.0],
    v = [10.0, 20.0, 30.0],
    ga = 0.0,
    gw = 1.0,
    f = 1.0
)

allocationwrite = (
    kw = Matrix([1.0 1.0 1.0]'),
    βw = 1.0,
    e = [1.0, 1.0, 1.0],
    v = [10.0, 20.0, 30.0],
    ga =1.0,
    gw =1.0,
    f = 1.0
)


state = State(
    # Write history is 1 -> 2 -> 3
    [0.0 0.0 0.0;
         1.0 0.0 0.0;
         0.0 1.0 0.0],
    zeros(3),
    [1.0, 1.0, 0.0], # 3 is (artificially) set to unused/ free to be allocated
    [0.0, 1.0, 0],
    Matrix([0.0 1.0 0.0]') # last read was 2, so forward points to 3, back to 1
)

@testset "Sharp read/write" begin
    @testset "Readweights" begin
        L, wr = state.L, state.wr
        r1 = readweights(M, contentread, L, wr)
        @test round.(r1; digits=5) == Matrix([1.0 0 0]')
        r2 = readweights(M, backwardread, L, wr)
        @test round.(r2; digits=5) == Matrix([1.0 0 0]')
        r3 = readweights(M, forwardread, L, wr)
        @test round.(r3; digits=5) == Matrix([0.0 0 1]')
    end
    @testset "Writeweights" begin
        ww, wr, u = state.ww, state.wr, state.u
        ww1 = writeweights(M, contentwrite, ww, wr, u)
        @test round.(ww1; digits=3) == Matrix([1.0 0.0 0.0]')
        ww2 = writeweights(M, allocationwrite, ww, wr, u)
        @test round.(ww2; digits=3) == Matrix([0.0 0.0 1.0]')
    end
end

@testset "Gradients" begin
    @testset "Read gradient" begin
        L, wr = state.L, state.wr
        r1 = readweights(M, contentread, L, wr)
        function f(M, contentread, L, wr)
            sum(readweights(M, contentread, L, wr))
        end
        readweights_g = gradient(f, M, contentread, L, wr)
        @test true
    end
end

@testset "Erase and add" begin
    new = DNC.eraseandadd(M, [0.0, 0, 1], ones(3), [2.0, 2.0, 2.0])
    @test new[3,:] == [2.0, 2.0, 2.0]
    ww2 = round.(writeweights(M, allocationwrite, state.ww, state.wr, state.u); digits=3)
    new2 = DNC.eraseandadd(M, ww2, allocationwrite[:e], allocationwrite[:v]) 
    @test new2[3,:] == allocationwrite[:v]
end

using Random
rng = MersenneTwister(234)
@testset "MemoryAccess" begin
    insize, N, W, R = 20, 5, 10, 2
    ma = DNC.MemoryAccess(insize, N, W, R)
    inputs = rand(rng, insize)
    @testset "Dimensions" begin
        @test size(ma.M) == (N, W)
        @test size(ma.state.wr) == (N, R)
        readvectors = ma(inputs)
        @test size(readvectors) == (W, R)
    end
    @testset "Gradient" begin
        g = gradient(inputs) do inputs
            sum(ma(inputs))
        end
        @test !isnothing(g)
    end

end
