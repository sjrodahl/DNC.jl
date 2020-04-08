using Flux: σ
using Zygote
using DNC: contentaddress

function generate_Sn(n)
    vec = rand(Float32, n)
    vec = exp.(vec)/sum(exp.(vec))
    vec
end

function generate_Δn(n)
    vec = rand(Float32, n)
    vec = exp.(vec)/(sum(exp.(vec).+(rand(Float32, n)/2)))
    vec
end

in_Sn(vec) = sum(vec) == 1.0
in_Δn(vec) = sum(vec) >= 0.0 && sum(vec) <= 1.0

N, W, R, B = 4, 8, 2, 4
keyr = rand(Float32, W, R, B)
keyw = rand(Float32, W, 1, B)
mem = rand(Float32, N, W, B)
βr = DNC.oneplus.(rand(Float32, R, B))
βw = DNC.oneplus.(rand(Float32, 1, B))
inputs = (wr=rand(Float32, N, R, B),
          f=σ.(rand(Float32, R, B)),
          ww=rand(Float32, N, 1, B),
          readmode=rand(Float32, 3, B))
state = State(
    zeros(Float32, N, N, B),
    zeros(Float32, N, B),
    zeros(Float32, N, B),
    zeros(Float32, N, B),
    zeros(Float32, N, R, B))

@testset "Batch-training" begin
    cr= DNC.contentaddress(keyr, mem, βr)
    @test size(cr) == (N, R, B)
    @test eltype(cr) == Float32
    g = gradient(keyr, mem, βr) do k, m, b
        sum(DNC.contentaddress(k, m, b))
    end
    @test length(g) == 3
    cw = DNC.contentaddress(keyw, mem, βw)
    @test size(cw) == (N, 1, B)
    memret = DNC.memoryretention(inputs.wr, inputs.f)
    @test eltype(memret) == Float32
    @test size(memret) == (N, B)
    u = DNC.usage(state.u, state.ww, state.wr, inputs.f)
    @test eltype(u) == Float32
    a = DNC.allocationweighting(u)
    @test eltype(a) == Float32
    gw, ga = rand(Float32, B), rand(Float32, B)
    ww = DNC.writeweight(cw, a, gw, ga)
    @test eltype(ww) == Float32
    @test size(ww) == (N, 1, B)
    forw = DNC.forwardweight(state.L, state.wr)
    backw = DNC.backwardweight(state.L, state.wr)
    wr = DNC.readweight(backw, cr, forw, inputs.readmode)
    @test eltype(forw) == Float32
    @test eltype(backw) == Float32
    @test eltype(wr) == Float32
    @test size(wr) == (N, R, B)
    p = DNC.precedenceweight(state.p, ww)
    @test size(p) == (N, B)
    L = zeros(Float32, N, N, B) # Don't interfere with later tests
    DNC.updatelinkmatrix!(L, p, ww)
    @test eltype(L) == Float32

end

@testset "Content-based addressing" begin
    M =Float32[0.1 0.5 1.5;
                  -1.2 0.8 0.0]
    key = Matrix(M')
    key[1, 1] = 0 # Avoid exact match
    β = [100.0f0, 100.0f0]
    # With a high sharpener β, the match is found
    res = contentaddress(key, M, β)
    @test res[1] ≈ 1
    @test eltype(res) == Float32
    g = gradient(key, M, β) do k, M, β
        sum(sum(contentaddress(k, M, β)))
    end
    @test length(g) == 3
    for grad in g
        @test eltype(grad) == Float32
    end
    # Should return equal values for parallel memory rows
    M = Float32[1 0; 2 0]
    key = Float32.(Matrix([1 1]'))
    β = [1.f0]
    wc = contentaddress(key, M, β)
    @test wc[1] == wc[2]
end


@testset "Memory allocation" begin
    usagecase1 = (wr = Float32.(Matrix([0.5 0.25 0.25]')), f=[1.f0], ww = [0.25f0, 0.5f0, 0.25f0])
    usagecase2 = (wr =Float32.([0.6 0.0;
                        0.3 0.5;
                        0.1 0.5]),
                  f=[1.f0, 0.f0],
                  ww = [0.25f0, 0.5f0, 0.25f0])
    u_prev = [1.0f0, 0.0f0, 0.0f0]
    state2 = State(3, 2)
    state2.ww = usagecase2.ww
    state2.wr = usagecase2.wr
    state2.u = u_prev

    @testset "Memory retention 𝜓" begin
        memret = DNC.memoryretention(usagecase1.wr, usagecase1.f) 
        @test memret == [0.5, 0.75, 0.75]
        @test eltype(memret) == Float32
        g = gradient(usagecase1.wr, usagecase1.f) do wr, f
            sum(DNC.memoryretention(wr, f))
        end
        @test length(g) == 2
        # Two read heads
        for grad in g
            @test eltype(grad) == Float32
        end
        memret = DNC.memoryretention(usagecase2.wr, usagecase2.f)
        @test isapprox(memret, [0.4, 0.7, 0.9], atol=1e-5)
        @test eltype(memret) == Float32
        g = gradient(usagecase2.wr, usagecase2.f) do wr, f
            sum(DNC.memoryretention(wr, f))
        end
        @test length(g) == 2
        for grad in g
            @test eltype(grad) == Float32
        end
    end

    @testset "Usage u⃗" begin
        wr, f, ww = usagecase1
        u = DNC.usage(u_prev, ww, wr, f)
        @test u == [1//2, 3//8, 3//16]
        @test eltype(u) == Float32
        g = gradient(ww, wr, f) do ww, wr, f
            sum(DNC.usage(u_prev, ww, wr, f))
        end
        @test length(g) == 3
        for grad in g
            @test eltype(grad) == Float32
        end
        # Two read heads
        wr, f, ww = usagecase2
        u = DNC.usage(u_prev, ww, wr, f)
        @test isapprox(u, [0.4, 0.35, 0.225], atol=1e-5)
        @test eltype(u) == Float32
        g = gradient(ww, wr, f) do ww, wr, f
            sum(DNC.usage(u_prev, ww, wr, f))
        end
        @test length(g) == 3
        for grad in g
            @test eltype(grad) == Float32
        end
    end

    @testset "Allocation a⃗" begin
        # Using approximation due to DNC's use of _EPSILON to avoid num. instability
        wr, f, ww = usagecase2
        u = DNC.usage(u_prev,ww, wr, f)
        alloc = DNC.allocationweighting(u)
        @test eltype(alloc) == Float32
        @test isapprox(alloc, [0.04725, 0.14625, 0.775]; atol=DNC._EPSILON*10)
        g = gradient(x->sum(DNC.allocationweighting(x)), u)
        @test length(g) == 1
        for grad in g
            @test eltype(grad) == Float32
        end
        # Allocation is zero if all usages are 1
        allused = ones(5)
        @test isapprox(DNC.allocationweighting(allused), (zeros(5)); atol=DNC._EPSILON*10)
        # First location is allocated if all is available
        noneused = zeros(5)
        @test isapprox(DNC.allocationweighting(noneused), [1.0, 0, 0, 0, 0]; atol=DNC._EPSILON*10)
    end

    @testset "Write weighting ww" begin
        ga = 0.f0
        gw = 1.f0
        a = generate_Δn(3)
        cw = generate_Sn(3)
        ww = DNC.writeweight(cw, a, gw, ga) 
        @test ww == cw
        @test eltype(ww) == Float32
    end
end

@testset "Temporal linkage" begin
    @testset "Precedence weights p⃗" begin
        ww = [0.1f0, 0.2f0, 0.3f0, 0.4f0]
        p0 = zeros(Float32, 4)
        p = DNC.precedenceweight(p0, ww) 
        @test p == ww
        @test eltype(p) == Float32
    end

    # Write weights. Writes to location 1, 2, and 3 iteratively.
    ww = [[1.0f0, 0.0f0, 0.0f0],
            [0.0f0, 1.0f0, 0.0f0],
            [0.0f0, 0.0f0, 1.0f0]]
    # Temporal link matrix
    L = Matrix(zeros(Float32, 3, 3))
    # precedence weights
    p = zeros(Float32, 3)
    # Expected evolution of L
    expected = [
        zeros(Float32, 3, 3),
        [0.0f0 0.0f0 0.0f0;
        1.0f0 0.0f0 0.0f0;
        0.0f0 0.0f0 0.0f0],
        [0.0f0 0.0f0 0.0f0;
        1.0f0 0.0f0 0.0f0;
        0.0f0 1.0f0 0.0f0]
    ]
    @testset "Link matrix L" begin
        for i in 1:length(ww)
            DNC.updatelinkmatrix!(L, p, ww[i])
            @test L == expected[i]
            @test eltype(L) == Float32
            p = DNC.precedenceweight(p, ww[i])
        end
        # Last read was location 2. The forward weight should point to 3, backward to 1.
        wr = [0.f0, 1, 0]
        forward = DNC.forwardweight(L, wr)
        backward = DNC.backwardweight(L, wr)
        @test  forward == [0, 0, 1]
        @test backward == [1, 0, 0]
        @test eltype(forward) == Float32
        @test eltype(backward) == Float32
    end

    @testset "Read weighting w_t^r" begin
        b = [0.f0, 0, 1]
        f = [1.f0, 0, 0]
        c = [0.f0, 1, 0]
        pi = [0.f0, 1, 0]
        readw = DNC.readweight(b, c, f, pi)
        @test in_Δn(readw)
        @test eltype(readw) == Float32
    end

end
