import Flux: softmax, σ
using Random
using Zygote

rng = MersenneTwister(234)

xs = sort([-1000, -10.0, -1.5, 0.0, 0.1, 1.0, 15.0])
@testset "oneplus" for i in 1:length(xs)
    # Outputs correct range
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
end

@testset "Cosine similarity" begin
    @test DNC.cosinesim([1, 2], [1, 2]) ≈ 1
    @test DNC.cosinesim([1, 1], [0,1]) ≈ cos(π/4)
    @test DNC.cosinesim([-1, 1], [1, 0]) ≈ cos(3π/4)
    @test DNC.cosinesim([1, 0, 0], [0, 1, 0]) == 0
end

@testset "Calc output" begin
    outsize = 5
    N, W, R = 3, 5, 2
    readvectors = rand(rng, W, R)
    Wr = rand(rng, outsize, R*W)
    v = rand(rng, outsize)
    res = DNC.calcoutput(v, readvectors, Wr)
    @test size(res) == (outsize,)

end

@testset "Split ξ $(R) read head" for R in 1:2
    W = 5
    ξ = rand(rng, 10)
    transforms = DNC.inputmappings(10, R, W)
    inputs = DNC.split_ξ(ξ, transforms)

    @testset "Dimensions" begin
        @test size(inputs.kr) == (W, R)
        @test size(inputs.kw) == (W, 1)
        @test size(inputs.βr) == (R,)
        @test size(inputs.βw) == (1,)
        @test size(inputs.ga) == (1,)
        @test size(inputs.gw) == (1,)
        @test size(inputs.v) == (W,)
        @test size(inputs.e) == (W,)
        @test size(inputs.f) == (R,)
        @test size(inputs.readmode) == (3, R)
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

