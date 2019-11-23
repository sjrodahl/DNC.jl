import Flux: softmax

@testset "Cosine similarity" begin
    @test DNC.cosinesim([1, 2], [1, 2]) ≈ 1
    @test DNC.cosinesim([1, 1], [0,1]) ≈ cos(π/4)
    @test DNC.cosinesim([-1, 1], [1, 0]) ≈ cos(3π/4)
    @test DNC.cosinesim([1, 0, 0], [0, 1, 0]) == 0
end

@testset "Utils" begin
    # Sharpener of 1 does not affect result (softmax is from NNlib package)
    @test DNC.weighted_softmax([1.0, 2.0], 1) == softmax([1.0, 2.0])
    # Test for correct results
    @testset "weighted softmax" for xs in [[0.0, 1.0], [1.0, 2.0], [0.5, -0.6], [-1.3, -5.0]], β in [-1, 0, 0.5, 2, 10]
        @test DNC.weighted_softmax(xs, β)[1] ≈ exp(xs[1]*β)/(exp(xs[1]*β)+exp(xs[2]*β))
        @test DNC.weighted_softmax(xs, β)[2] ≈ exp(xs[2]*β)/(exp(xs[1]*β)+exp(xs[2]*β))
    end
end

@testset "Content-based addressing" begin
    M = Matrix(
        [0.1 0.5 1.5;
        -1.2 0.8 0.0]
        )
    key = M[1,:]
    key[1] = 0 # Avoid exact match
    β = 100
    # With a high sharpener β, the match is found
    @test contentaddress(key, M, β)[1] ≈ 1
    β = 10
    # Should return equal values for parallel memory rows
    M = Matrix([1 0; 2 0])
    key = [1, 1]
    β = 1
    w_c = contentaddress(key, M, β)
    @test w_c[1] == w_c[2]
end
