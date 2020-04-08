using Zygote

N = 3
W = 3

M = Float32[1 2 3;
    -1 -2  -3;
    1 -2 3]

# Content-based read
contentread = (
    kr = Matrix([1.0f0 2.0f0 0.0f0]'),
    βr = [100.0f0],
    f = [1.0f0],
    readmode = Matrix([0.0f0 1.0f0 0.0f0]')
    )

# Temporal linkage based read
backwardread = (
    kr = Matrix([-1.0f0 -2.0f0 -5.0f0]'),
    βr = [100.0f0],
    f = [0.0f0],
    readmode = Matrix([1.0f0 0.0f0 0.0f0]')
    )

forwardread = (
    kr = Matrix([-1.0f0 -2.0f0 -5.0f0]'),
    βr = [100.0f0],
    f = [0.0f0],
    readmode = Matrix([0.0f0 0.0f0 1.0f0]')
    )

contentwrite = (
    kw = Matrix([1.0f0 2 0]'),
    βw = [10.0f0],
    e = [1.0f0, 1.0f0, 1.0f0],
    v = [10.0f0, 20.0f0, 30.0f0],
    ga = [0.0f0],
    gw = [1.0f0],
    f = [1.0f0]
)

allocationwrite = (
    kw = Matrix([1.0f0 1.0f0 1.0f0]'),
    βw = [1.0f0],
    e = [1.0f0, 1.0f0, 1.0f0],
    v = [10.0f0, 20.0f0, 30.0f0],
    ga =[1.0f0],
    gw =[1.0f0],
    f = [1.0f0]
)


state = State(
    # Write history is 1 -> 2 -> 3
    [0.0f0 0.0f0 0.0f0;
         1.0f0 0.0f0 0.0f0;
         0.0f0 1.0f0 0.0f0],
    zeros(3),
    [1.0f0, 1.0f0, 0.0f0], # 3 is (artificially) set to unused/ free to be allocated
    [0.0f0, 1.0f0, 0],
    Matrix([0.0f0 1.0f0 0.0f0]') # last read was 2, so forward points to 3, back to 1
)

@testset "Sharp read/write" begin
    @testset "Readweights" begin
        L, wr = state.L, state.wr
        r1 = readweights(M, contentread, L, wr)
        @test eltype(r1) == Float32
        @test round.(r1; digits=5) == Matrix([1.0 0 0]')
        r2 = readweights(M, backwardread, L, wr)
        @test eltype(r2) == Float32
        @test round.(r2; digits=5) == Matrix([1.0 0 0]')
        r3 = readweights(M, forwardread, L, wr)
        @test eltype(r3) == Float32
        @test round.(r3; digits=5) == Matrix([0.0 0 1]')
    end
    @testset "Writeweights" begin
        ww, wr, prev_u = state.ww, state.wr, state.u
        u = DNC.usage(prev_u, ww, wr, contentwrite.f)
        ww1 = writeweights(M, contentwrite, u)
        @test eltype(ww1) == Float32
        @test round.(ww1; digits=3) == Matrix([1.0 0.0 0.0]')
        u = DNC.usage(prev_u, ww, wr, allocationwrite.f)
        ww2 = writeweights(M, allocationwrite, u)
        @test eltype(ww2) == Float32
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
    u = DNC.usage(state.u, state.ww, state.wr, allocationwrite.f)
    new = DNC.eraseandadd(M, [0.0f0, 0, 1], ones(Float32, 3), [2.0f0, 2.0f0, 2.0f0])
    @test eltype(new) == Float32
    @test new[3,:] == [2.0, 2.0, 2.0]
    ww2 = round.(writeweights(M, allocationwrite, u); digits=3)
    new2 = DNC.eraseandadd(M, ww2, allocationwrite[:e], allocationwrite[:v]) 
    @test new2[3,:] == allocationwrite[:v]
end

using Random
rng = MersenneTwister(234)

@testset "MemoryAccess" begin
    insize, N, W, R = 20, 5, 10, 2
    ma = DNC.MemoryAccess(insize, N, W, R)
    inputs = rand(rng, Float32, insize)
    @testset "Dimensions" begin
        @test size(ma.M) == (N, W)
        @test size(ma.state.wr) == (N, R)
        readvectors = ma(inputs)
        @test eltype(readvectors) == Float32
        @test size(readvectors) == (W, R)
    end
    @testset "Gradient" begin
        g = gradient(inputs) do inputs
            sum(ma(inputs))
        end
        @test !isnothing(g)
    end

end

insize, N, W, R, B = 20, 5, 10, 2, 4
ma = DNC.MemoryAccess(insize, N, W, R, B)
M = rand(rng, Float32, N, W, B)
L = rand(rng, Float32, N, N, B)
prev_wr = rand(rng, Float32, N, R, B)
inputsraw = rand(rng, Float32, insize, B)
mappings = DNC.inputmappings(20, R, W)
inputs = DNC.split_ξ(inputsraw, mappings)
state = State(N, R, B)

@testset "Batch training" begin
    r = readweights(M, inputs, L, prev_wr)
    @test size(r) == (N, R, B)
    u = DNC.usage(state.u, state.ww, state.wr, inputs.f)
    w = writeweights(M, inputs, u)
    @test size(w) == (N, 1, B)
    newM = DNC.eraseandadd(M, w, inputs.e, inputs.v)
    @test size(newM) == (N, W, B)
    res = ma(inputsraw)
    @test size(res) == (W, R, B)
end
