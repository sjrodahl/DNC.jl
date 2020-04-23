import Flux: softmax, σ
using Random
using Zygote

rng = MersenneTwister(234)

xs = sort(Float32.([-1000, -10.0, -1.5, 0.0, 0.1, 1.0, 15.0]))
@testset "oneplus" for i in 1:length(xs)
    # Outputs correct range
    @test typeof(DNC.oneplus(xs[i])) == Float32
    @test DNC.oneplus(xs[i]) >= 1.0
    # Maintains order
    if i>1
        @test DNC.oneplus(xs[i]) > DNC.oneplus(xs[i-1])
    end
end

@testset "clip" begin
    arr = [-3, -2.5, -1, 1.5, 3.5]
    @test DNC.clip(arr, nothing) == arr
    @test DNC.clip(arr, 2) == [-2, -2, -1, 1.5, 2]
end

@testset "Calc output" begin
    outsize = 5
    N, W, R, B = 3, 5, 2, 1
    readvectors = rand(rng, Float32, W*R, B)
    Wr = rand(rng, Float32, outsize, R*W, B)
    v = rand(rng, Float32, outsize, B)
    res = DNC.calcoutput(v, readvectors, Wr)
    @test size(res) == (outsize, B)
    @test eltype(res) == Float32
end

@testset "Split ξ $(R) read head" for R in 1:2
    W, B = 5, 1
    ξ = rand(rng, Float32, 10, B)
    transforms = DNC.inputmappings(10, R, W)
    inputs = DNC.split_ξ(ξ, transforms)

    @testset "Type Float32" for inp in inputs
        eltype(inp) == Float32
    end

    @testset "Dimensions" begin
        @test size(inputs.kr) == (W, R, B)
        @test size(inputs.kw) == (W, 1, B)
        @test size(inputs.βr) == (R,B)
        @test size(inputs.βw) == (1,B)
        @test size(inputs.ga) == (1,B)
        @test size(inputs.gw) == (1,B)
        @test size(inputs.v) == (W,B)
        @test size(inputs.e) == (W,B)
        @test size(inputs.f) == (R,B)
        @test size(inputs.readmode) == (3, R, B)
    end

    @testset "Domain" begin
        # β ∈ [1, ∞)
        @test !any(x->x<1, inputs.βr)
        @test !any(x->x<1, inputs.βw)
        # g ∈ [0, 1]
        @test 0 <= inputs.ga[1] <= 1
        @test 0 <= inputs.gw[1] <= 1
        # e ∈ [0, 1]^W
        @test !any(x -> x<0 || x>1, inputs.e) 
        # f ∈ [0, 1]^R
        @test !any(x -> x<0 || x>1, inputs.f) 
        # π_i ∈ S_3
        @test round.(sum(inputs.readmode; dims=1)[1,:]; digits=5) == ones(R)
    end
    
    @testset "Differentiability" begin
        g = gradient(ξ, transforms) do ξ, transforms
            inputs = DNC.split_ξ(ξ, transforms)
            tot = 0
            for v in inputs
                tot += sum(v)
            end
            tot
        end
        @test !isnothing(g)
    end
end

