using Zygote
using DNC: contentaddress

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

in_Sn(vec) = sum(vec) == 1.0
in_Δn(vec) = sum(vec) >= 0.0 && sum(vec) <= 1.0

@testset "Content-based addressing" begin
    M = Matrix(
        [0.1 0.5 1.5;
        -1.2 0.8 0.0]
        )
    key = Matrix(M')
    key[1, 1] = 0 # Avoid exact match
    β = [100.0, 100.0]
    # With a high sharpener β, the match is found
    @test contentaddress(key, M, β)[1] ≈ 1
    g = gradient(key, M, β) do k, M, β
        sum(sum(contentaddress(k, M, β)))
    end
    @test length(g) == 3
    # Should return equal values for parallel memory rows
    M = [1 0; 2 0]
    key = Matrix([1 1]')
    β = 1
    wc = contentaddress(key, M, β)
    @test wc[1] == wc[2]
end


@testset "Memory allocation" begin
    usagecase1 = (wr = Matrix([0.5 0.25 0.25]'), f=1, ww = [0.25, 0.5, 0.25])
    usagecase2 = (wr = [0.6 0.0;
                        0.3 0.5;
                        0.1 0.5],
                  f=[1, 0],
                  ww = [0.25, 0.5, 0.25])
    u_prev = [1.0, 0.0, 0.0]
    state2 = State(3, 2)
    state2.ww = usagecase2.ww
    state2.wr = usagecase2.wr
    state2.u = u_prev

    @testset "Memory retention 𝜓" begin
        @test DNC.memoryretention(usagecase1.wr, usagecase1.f) == [0.5, 0.75, 0.75]
        g = gradient(usagecase1.wr, usagecase1.f) do wr, f
            sum(DNC.memoryretention(wr, f))
        end
        @test length(g) == 2
        # Two read heads
        @test DNC.memoryretention(usagecase2.wr, usagecase2.f) == [0.4, 0.7, 0.9]
        g = gradient(usagecase2.wr, usagecase2.f) do wr, f
            sum(DNC.memoryretention(wr, f))
        end
        @test length(g) == 2
    end

    @testset "Usage u⃗" begin
        wr, f, ww = usagecase1
        𝜓 = DNC.memoryretention(wr, f)
        @test DNC.usage(u_prev, ww, 𝜓) == [1//2, 3//8, 3//16]
        g = gradient(ww, 𝜓) do ww, 𝜓
            sum(DNC.usage(u_prev, ww, 𝜓))
        end
        @test length(g) == 2
        # Two read heads
        wr, f, ww = usagecase2
        𝜓 = DNC.memoryretention(wr, f)
        @test DNC.usage(u_prev, ww, 𝜓) == [0.4, 0.35, 0.225]
        g = gradient(ww, 𝜓) do ww, 𝜓
            sum(DNC.usage(u_prev, ww, 𝜓))
        end
        @test length(g) == 2
    end

    @testset "Allocation a⃗" begin
        # Using approximation due to DNC's use of _EPSILON to avoid num. instability
        wr, f, ww = usagecase2
        u = DNC.usage(u_prev,ww, DNC.memoryretention(wr, f))
        alloc = DNC.allocationweighting(u)
        @test isapprox(alloc, [0.04725, 0.14625, 0.775]; atol=DNC._EPSILON*10)
        @test DNC.allocationweighting(f, wr, ww, u_prev) == alloc
        @test DNC.allocationweighting(f, state2) == alloc
        g = gradient(f, wr, ww) do f, wr, ww
            sum(DNC.allocationweighting(f, wr, ww, u_prev))
        end
        @test length(g) == 3
        # Allocation is zero if all usages are 1
        allused = ones(5)
        @test isapprox(DNC.allocationweighting(allused), (zeros(5)); atol=DNC._EPSILON*10)
        # First location is allocated if all is available
        noneused = zeros(5)
        @test isapprox(DNC.allocationweighting(noneused), [1.0, 0, 0, 0, 0]; atol=DNC._EPSILON*10)
    end

    @testset "Write weighting ww" begin
        ga = 0
        gw = 1
        a = generate_Δn(3)
        cw = generate_Sn(3)
        @test DNC.writeweight(cw, a, gw, ga) == cw
    end
end

@testset "Temporal linkage" begin
    @testset "Precedence weights p⃗" begin
        ww = [0.1, 0.2, 0.3, 0.4]
        p0 = zeros(4)
        @test DNC.precedenceweight(p0, ww) == ww
    end

    # Write weights. Writes to location 1, 2, and 3 iteratively.
    ww = [[1.0, 0.0, 0.0],
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
        for i in 1:length(ww)
            DNC.updatelinkmatrix!(L, p, ww[i])
            @test L == expected[i]
            p = DNC.precedenceweight(p, ww[i])
        end
        # Last read was location 2. The forward weight should point to 3, backward to 1.
        wr = [0.0, 1, 0]
        @test DNC.forwardweight(L, wr) == [0, 0, 1]
        @test DNC.backwardweight(L, wr) == [1, 0, 0]
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
