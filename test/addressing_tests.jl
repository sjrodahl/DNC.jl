import Flux: softmax

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
