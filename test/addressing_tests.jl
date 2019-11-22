
@testset "Cosine similarity" begin
    @test DNC.cosinesim([1, 2], [1, 2]) ≈ 1
    @test DNC.cosinesim([1, 1], [0,1]) ≈ cos(π/4)
    @test DNC.cosinesim([-1, 1], [1, 0]) ≈ cos(3π/4)
    @test DNC.cosinesim([1, 0, 0], [0, 1, 0]) == 0
end
