import Flux: softmax, σ

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
    @test DNC.weighted_softmax([1.0, 2.0], 1) == softmax([1.0, 2.0])
    # Test for correct results
    @testset "loop" for
        xs in [[0.0, 1.0], [1.0, 2.0], [0.5, -0.6], [-1.3, -5.0]],
            β in [1 , 2, 10]
        @test DNC.weighted_softmax(xs, β)[1] ≈ exp(xs[1]*β)/(exp(xs[1]*β)+exp(xs[2]*β))
        @test DNC.weighted_softmax(xs, β)[2] ≈ exp(xs[2]*β)/(exp(xs[1]*β)+exp(xs[2]*β))
    end
end

@testset "Cosine similarity" begin
    @test DNC.cosinesim([1, 2], [1, 2]) ≈ 1
    @test DNC.cosinesim([1, 1], [0,1]) ≈ cos(π/4)
    @test DNC.cosinesim([-1, 1], [1, 0]) ≈ cos(3π/4)
    @test DNC.cosinesim([1, 0, 0], [0, 1, 0]) == 0
end

import Base.==
==(a::ReadHead, b::ReadHead) = a.k==b.k && a.β==b.β && a.f==b.f && a.π==b.π
==(a::WriteHead, b::WriteHead) = a.k==b.k && a.β==b.β && a.e==b.e && a.v==b.v && a.g_a==b.g_a && a.g_w==b.g_w

@testset "Split ξ" begin
    tooshort = [1, 2, 3]
    @test_throws ErrorException DNC.split_ξ(tooshort, 1, 3)
    @testset "One read head" begin
        R = 1
        W = 3
        k_r = [1.0, 2, 3]
        β_r = 10.0
        k_w = [4.0, 5, 6]
        β_w = 10.0
        erase = [1.0, 1, 1]
        add = [11.0, 12, 13]
        free = 1.0
        g_a = 0.5
        g_w = 1.0
        readmode = [0.0, 0.0, 1]

        readhead = ReadHead(k_r, β_r, σ(free), softmax(readmode))
        writehead = WriteHead(k_w, β_w, σ.(erase), add, σ(g_a), σ(g_w))

        ξ = [k_r; β_r; k_w; β_w; erase; add; free; g_a; g_w; readmode]

        rhs, wh = DNC.split_ξ(ξ, R, W)
        @test rhs[1] == readhead
        @test wh == writehead
    end # begin

    @testset "Multiple read heads" begin
        R = 2
        W = 5
        k_r = [[1.0, 2, 3, 4, 5],[6.0, 7, 8, 9, 10]]
        β_r = [10.0, 20.0]
        k_w = [11.0, 12, 13, 14, 15]
        β_w = 10.0
        erase = [1.0, 1, 1, 3, 4]
        add = [21.0, 22, 23, 24, 25]
        free = [1.0, 0.0]
        g_a = 0.5
        g_w = 1.0
        readmode = [[0.5, 0.5, 1.0],[0.0, 0.0, 1]]
        rh1 = ReadHead(k_r[1], β_r[1], σ(free[1]), softmax(readmode[1]))
        rh2 = ReadHead(k_r[2], β_r[2], σ(free[2]), softmax(readmode[2]))
        writehead = WriteHead(k_w, β_w, σ.(erase), add, σ(g_a), σ(g_w))
        ξ = [k_r...; β_r...; k_w; β_w; erase...; add...; free; g_a; g_w; readmode...]

        rhs, wh = DNC.split_ξ(ξ, R, W)
        @test rhs[1] == rh1
        @test rhs[2] == rh2
        @test wh == writehead
        end # begin
end # begin
