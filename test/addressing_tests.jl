
function generate_Sn(n)
    vec = rand(n)
    vec = exp.(vec)/sum(exp.(vec))
    vec
end

function generate_Δn(n)
    vec = rand(n)
    vec = exp.(vec)/(sum(exp.(vec).+(rand(n)/2)))
    vec
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

    @testset "Write weighting w_w" begin
        g_a = 0
        g_w = 1
        a = generate_Δn(3)
        c_w = generate_Sn(3)
        @test DNC.writeweight(c_w, a, g_w, g_a) == c_w
    end
end

@testset "Temporal linkage" begin
    @testset "Precedence weights p⃗" begin
        w_w = [0.1, 0.2, 0.3, 0.4]
        p_0 = zeros(4)
        @test DNC.precedenceweight(p_0, w_w) == w_w
    end

    # Write weights. Writes to location 1, 2, and 3 iteratively.
    w_w = [[1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]]
    # Temporal link matrix
    L = Matrix(zeros(3, 3))
    # precedence weights
    p = zeros(3)
    # Expected evolution of L
    expected = [
        Matrix(zeros(3, 3)),
        Matrix([0.0 0.0 0.0;
                1.0 0.0 0.0;
                0.0 0.0 0.0]),
        Matrix([0.0 0.0 0.0;
                1.0 0.0 0.0;
                0.0 1.0 0.0])
    ]
    @testset "Link matrix L" begin
        for i in 1:length(w_w)
            DNC.updatelinkmatrix!(L, p, w_w[i])
            @test L == expected[i]
            p = DNC.precedenceweight(p, w_w[i])
        end
        # Last read was location 2. The forward weight should point to 3, backward to 1.
        w_r = [0.0, 1, 0]
        @test DNC.forwardweight(L, w_r) == [0, 0, 1]
        @test DNC.backwardweight(L, w_r) == [1, 0, 0]
    end

    @testset "Read weighting w_t^r" begin
        b = [0, 0, 1]
        f = [1, 0, 0]
        c = [0, 1, 0]
        pi = [0, 1, 0]
        readw = DNC.readweight(b, c, f, pi)
        @test in_Δn(readw)
    end

end
