using Flux: σ, softmax

@testset "Split ξ" begin
    tooshort = [1, 2, 3]
    @test_throws ErrorException DNC.split_ξ(tooshort, 1, 3)
    @testset "One read head" begin
        R = 1
        W = 3
        k_r = [1.0, 2, 3]
        β_r = 10
        k_w = [4.0, 5, 6]
        β_w = 10
        erase = [1.0, 1, 1]
        add = [11.0, 12, 13]
        free = 1.0
        g_a = 0.5
        g_w = 1.0
        readmode = [0.0, 0.0, 1]

        ξ = [k_r; β_r; k_w; β_w; erase; add; free; g_a; g_w; readmode]

        vars = DNC.split_ξ(ξ, R, W)
        @test vars.k_r == [k_r]
        @test vars.β_r == [β_r]
        @test vars.k_w == k_w
        @test vars.erase == σ.(erase)
        @test vars.add == add
        @test vars.free == [σ(free)]
        @test vars.alloc_gate == σ(g_a)
        @test vars.write_gate == σ(g_w)
        @test vars.readmode == [softmax(readmode)]
    end # begin

    @testset "Multiple read heads" begin
        R = 2
        W = 5
        k_r = [[1.0, 2, 3, 4, 5],[6.0, 7, 8, 9, 10]]
        β_r = [10, 20]
        k_w = [11.0, 12, 13, 14, 15]
        β_w = 10
        erase = [1.0, 1, 1, 3, 4]
        add = [21.0, 22, 23, 24, 25]
        free = [1.0, 0.0]
        g_a = 0.5
        g_w = 1.0
        readmode = [[0.5, 0.5, 1.0],[0.0, 0.0, 1]]

        ξ = [k_r...; β_r...; k_w; β_w; erase...; add...; free; g_a; g_w; readmode...]

        vars = DNC.split_ξ(ξ, R, W)
        @test vars.k_r == k_r
        @test vars.β_r == β_r
        @test vars.k_w == k_w
        @test vars.erase == σ.(erase)
        @test vars.add == add
        @test vars.free == σ.(free)
        @test vars.alloc_gate == σ(g_a)
        @test vars.write_gate == σ(g_w)
        @test vars.readmode == softmax.(readmode)

        end # begin
end # begin
