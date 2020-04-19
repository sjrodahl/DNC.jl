using Zygote

N, W, R, B= 3, 3, 1, 1


function expand(arr::AbstractVecOrMat, dims...)
    res = rand(Float32, dims...)
    res[:, :, 1] = Float32.(arr)
    res
end

M = expand([1 2 3;
           -1 -2 -3;
           1 -2 3], N, N, B)

# Content-based read
contentread = (
    kr = expand([1, 2, 0], W, R, B),
    βr = expand([100], R, B),
    f = expand([1], R, B),
    readmode = expand([0, 1, 0], (3, R, B))
    )

# Temporal linkage based read
backwardread = (
    kr =expand([-1, -2, -5], W, R, B),
    βr = expand([100], R, B),
    f = expand([0], R, B),
    readmode = expand([1, 0, 0], 3, R, B)
    )

forwardread = (
    kr = expand([-1,-2,-5], W, R, B),
    βr = expand([100], R, B),
    f = expand([0], R, B),
    readmode = expand([0,0,1], 3, R, B)
    )

contentwrite = (
    kw = expand([1,2, 0], W, R, B),
    βw = expand([10], R, B),
    e = expand([1, 1, 1], W, B),
    v = expand([10, 20, 30], W, B),
    ga = expand([0], 1, B),
    gw = expand([1], 1, B),
    f = expand([1], R, B)
)

allocationwrite = (
    kw = expand([1,1,1], W, R, B),
    βw = expand([1], R, B),
    e = expand([1, 1, 1], W, B),
    v = expand([10, 20, 30], W, B),
    ga =expand([1], 1, B),
    gw =expand([1], 1, B),
    f = expand([1], R, B)
)


state = State(
    # Write history is 1 -> 2 -> 3
    expand([0 0 0;
            1 0 0;
            0 1 0], N, N, B),
    zeros(Float32, N, B),
    expand([1, 1, 0], N, B), # 3 is (artificially) set to unused/ free to be allocated
    expand([0, 1, 0], N, 1, B),
    expand([0, 1, 0], N, R, B) # last read was 2, so forward points to 3, back to 1
)

@testset "Sharp read/write" begin
    @testset "Readweights" begin
        L, wr = state.L, state.wr
        r1 = readweights(M, contentread, L, wr)
        @test eltype(r1) == Float32
        @test round.(r1; digits=5)[:,1] == [1.0, 0, 0]
        r2 = readweights(M, backwardread, L, wr)
        @test eltype(r2) == Float32
        @test round.(r2; digits=5)[:, 1] == [1.0, 0, 0]
        r3 = readweights(M, forwardread, L, wr)
        @test eltype(r3) == Float32
        @test round.(r3; digits=5)[:, 1] == [0.0, 0, 1]
    end
    @testset "Writeweights" begin
        ww, wr, prev_u = state.ww, state.wr, state.u
        u = DNC.usage(prev_u, ww, wr, contentwrite.f)
        ww1 = writeweights(M, contentwrite, u)
        @test eltype(ww1) == Float32
        @test round.(ww1; digits=3)[:, 1, 1] == [1.0, 0.0, 0.0]
        u = DNC.usage(prev_u, ww, wr, allocationwrite.f)
        ww2 = writeweights(M, allocationwrite, u)
        @test eltype(ww2) == Float32
        @test round.(ww2; digits=3)[:, 1, 1] == [0.0, 0.0, 1.0]
    end
end

@testset "Erase and add" begin
    B = 1
    u = DNC.usage(state.u, state.ww, state.wr, allocationwrite.f)
    new = DNC.eraseandadd(M, reshape([0.0f0, 0, 1], N, 1, B), ones(Float32, 3, B), 2*ones(Float32, W, B))
    @test eltype(new) == Float32
    @test new[3,:, 1] == [2.0, 2.0, 2.0]
    ww2 = round.(writeweights(M, allocationwrite, u); digits=3)
    new2 = DNC.eraseandadd(M, ww2, allocationwrite[:e], allocationwrite[:v]) 
    @test new2[3,:, 1] == allocationwrite[:v][:, 1]
end

using Random
rng = MersenneTwister(234)

@testset "MemoryAccess" begin
    insize, N, W, R, B = 20, 5, 10, 2, 1
    ma = DNC.MemoryAccess(insize, N, W, R, B)
    inputs = rand(rng, Float32, insize, B)
    @testset "Dimensions" begin
        @test size(ma.M) == (N, W, B)
        @test size(ma.state.wr) == (N, R, B)
        readvectors = ma(inputs)
        @test eltype(readvectors) == Float32
        @test size(readvectors) == (W, R, B)
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
