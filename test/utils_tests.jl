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

@testset "weighted softmax" begin
    # Sharpener of 1 does not affect result (softmax is from NNlib package)
    @test DNC.weightedsoftmax([1.0, 2.0], 1) == softmax([1.0, 2.0])
    # Test for correct results
    @testset "loop" for
        xs in [[0.0, 1.0], [1.0, 2.0], [0.5, -0.6], [-1.3, -5.0]],
            β in [1 , 2, 10]
        @test DNC.weightedsoftmax(xs, β)[1] ≈ exp(xs[1]*β)/(exp(xs[1]*β)+exp(xs[2]*β))
        @test DNC.weightedsoftmax(xs, β)[2] ≈ exp(xs[2]*β)/(exp(xs[1]*β)+exp(xs[2]*β))
    end
    # Test for Float32 type stability
    @test eltype(DNC.weightedsoftmax([0.f0, 1.f0], 2.f0)) == Float32
end

@testset "Cosine similarity" begin
    a, b = ([1.f0, 1.f0], [0.f0, 0.5f0])
    @test DNC.cosinesim([1, 2], [1, 2]) ≈ 1
    @test DNC.cosinesim([1, 1], [0,1]) ≈ cos(π/4)
    @test DNC.cosinesim([-1, 1], [1, 0]) ≈ cos(3π/4)
    @test DNC.cosinesim([1, 0, 0], [0, 1, 0]) == 0
    @test eltype(DNC.cosinesim(a, b)) == Float32
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

