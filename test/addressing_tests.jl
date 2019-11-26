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


@testset "Memory allocation" begin
    usage_case_1 = (w_r = [0.5, 0.25, 0.25], f=1, w_w = [0.25, 0.5, 0.25])
    usage_case_2 = (w_r = [[0.6, 0.3, 0.1], [0.0, 0.5, 0.5]], f=[1, 0], w_w = [0.25, 0.5, 0.25])
    u_prev = [1.0, 0.0, 0.0]

    @testset "Memory retention 𝜓" begin
        @test DNC.memoryretention(usage_case_1.w_r, usage_case_1.f) == [0.5, 0.75, 0.75]
        # Two read heads
        @test DNC.memoryretention(usage_case_2.w_r, usage_case_2.f) == [0.4, 0.7, 0.9]
    end

    @testset "Usage u⃗" begin
        w_r, f, w_w = usage_case_1
        𝜓 = DNC.memoryretention(w_r, f)
        @test DNC.usage(u_prev, w_w, 𝜓) == [1//2, 3//8, 3//16]
        # Two read heads
        w_r, f, w_w = usage_case_2
        𝜓 = DNC.memoryretention(w_r, f)
        @test DNC.usage(u_prev, w_w, 𝜓) == [0.4, 0.35, 0.225]
    end

    @testset "Allocation a⃗" begin
        w_r, f, w_w = usage_case_2
        u = DNC.usage(u_prev,w_w, DNC.memoryretention(w_r, f))
        @test DNC.allocationweighting(u) ≈ [0.04725, 0.14625, 0.775]
        # Allocation is zero if all usages are 1
        u_1 = ones(5)
        @test DNC.allocationweighting(u_1) == (zeros(5))
    end

end
