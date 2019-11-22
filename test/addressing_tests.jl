
@testset "Cosine similarity" begin
    @test DNC.cosinesim([1, 2], [1, 2]) ≈ 1
end
