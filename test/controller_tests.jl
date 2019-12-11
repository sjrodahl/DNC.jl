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

        readhead = ReadHead(k_r, β_r, free, readmode)
        writehead = WriteHead(k_w, β_w, erase, add, g_a, g_w)

        ξ = [k_r; β_r; k_w; β_w; erase; add; free; g_a; g_w; readmode]

        rhs, wh = DNC.split_ξ(ξ, R, W)
        @test rhs[1] == readhead
        @test wh == writehead
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
        rh1 = ReadHead(k_r[1], β_r[1], free[1], readmode[1])
        rh2 = ReadHead(k_r[2], β_r[2], free[2], readmode[2])
        writehead = WriteHead(k_w, β_w, erase, add, g_a, g_w)
        ξ = [k_r...; β_r...; k_w; β_w; erase...; add...; free; g_a; g_w; readmode...]

        rhs, wh = DNC.split_ξ(ξ, R, W)
        @test rhs[1] == rh1
        @test rhs[2] == rh2
        @test wh == writehead
        end # begin
end # begin
