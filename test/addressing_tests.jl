using Flux: σ
using Zygote
using DNC: contentaddress


function generate_Sn(dims...)
    arr = rand(Float32, dims...)
    softmax(arr; dims=1)
end


function generate_Δn(dims...)
    fuzzysoftmax(arr; dims=1) = 
        exp.(arr)./(sum(exp.(arr); dims=dims) .+ rand(Float32, size(arr))/2)
    arr = rand(Float32, dims...)
    fuzzysoftmax(arr; dims=1)
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
          readmode=rand(Float32, 3, R, B))
state = State(
    zeros(Float32, N, N, B),
    zeros(Float32, N, B),
    zeros(Float32, N, B),
    zeros(Float32, N, 1, B),
    zeros(Float32, N, R, B))

@testset "Batch-training" begin
    cr= DNC.contentaddress(keyr, mem, βr)
    @test size(cr) == (N, R, B)
    @test eltype(cr) == Float32
    cw = DNC.contentaddress(keyw, mem, βw)
    @test size(cw) == (N, 1, B)
    memret = DNC.memoryretention(inputs.wr, inputs.f)
    @test eltype(memret) == Float32
    @test size(memret) == (N, B)
    u = DNC.usage(state.u, state.ww, state.wr, inputs.f)
    @test eltype(u) == Float32
    a = DNC.allocationweighting(u)
    @test eltype(a) == Float32
    gw, ga = rand(Float32, 1, B), rand(Float32, 1, B)
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
    M =reshape(Float32[0.1 0.5 1.5;
                       -1.2 0.8 0.0], 2, 3, 1)
    key = rand(Float32, 3, 1, 1)
    key[:, 1, 1] = M[1, : , 1]
    key[1, 1, 1] = 0 # Avoid exact match
    β = reshape([100.0f0, 100.0f0], 1, 2)
    # With a high sharpener β, the match is found
    res = contentaddress(key, M, β)
    @test res[1] ≈ 1
    @test eltype(res) == Float32
end


@testset "Memory allocation" begin
    N, W, R, B = 3, 4, 1, 1
    wr1 = rand(Float32, N, R, B)
    wr1[:, 1, 1] = [0.5f0, 0.25f0, 0.25f0]
    f1 = ones(Float32, R, B)
    ww1 = rand(Float32, N, 1, B)
    ww1[:, 1, 1] = [0.25f0, 0.5f0, 0.25f0]
    case1 = (wr=wr1, f=f1, ww=ww1)
    R = 2
    wr2 = rand(Float32, N, R, B)
    wr2[:, :, 1] = Float32.([0.6 0.0;
                             0.3 0.5;
                             0.1 0.5])
    f2 = rand(Float32, R, B)
    f2[:, 1] = [1.f0, 0.f0]
    ww2 = rand(Float32, N, 1, B)
    ww2[:, 1, 1] = [0.25f0, 0.5f0, 0.25f0]
    case2 = (wr=wr2, f=f2, ww=ww2)
    u_prev = rand(Float32, N, B)
    u_prev[:, 1] = [1.0f0, 0.0f0, 0.0f0]
    state2 = State(N, R, B)
    state2.ww = case2.ww
    state2.wr = case2.wr
    state2.u = u_prev

    @testset "Memory retention 𝜓" begin
        memret = DNC.memoryretention(case1.wr, case1.f) 
        @test memret[:, 1] == [0.5, 0.75, 0.75]
        @test eltype(memret) == Float32
        # Two read heads
        memret = DNC.memoryretention(case2.wr, case2.f)
        @test isapprox(memret[:, 1], [0.4, 0.7, 0.9], atol=1e-5)
        @test eltype(memret) == Float32
    end

    @testset "Usage u⃗" begin
        wr, f, ww = case1
        u = DNC.usage(u_prev, ww, wr, f)
        @test u[:, 1] == [1//2, 3//8, 3//16]
        @test eltype(u) == Float32
        # Two read heads
        wr, f, ww = case2
        u = DNC.usage(u_prev, ww, wr, f)
        @test isapprox(u[:, 1], [0.4, 0.35, 0.225], atol=1e-5)
        @test eltype(u) == Float32
    end

    @testset "Allocation a⃗" begin
        # Using approximation due to DNC's use of _EPSILON to avoid num. instability
        wr, f, ww = case2
        u = DNC.usage(u_prev,ww, wr, f)
        alloc = DNC.allocationweighting(u)
        @test eltype(alloc) == Float32
        @test isapprox(alloc[:, 1], [0.04725, 0.14625, 0.775]; atol=DNC._EPSILON*10)
        # Allocation is zero if all usages are 1
        allused = ones(5, B)
        @test isapprox(DNC.allocationweighting(allused), (zeros(5, B)); atol=DNC._EPSILON*10)
        # First location is allocated if all is available
        noneused = zeros(5, B)
        @test isapprox(DNC.allocationweighting(noneused)[:, 1], [1.0, 0, 0, 0, 0]; atol=DNC._EPSILON*10)
    end

    @testset "Write weighting ww" begin
        ga = zeros(Float32, 1, B)
        gw = ones(Float32, 1, B)
        a = generate_Δn(N, B)
        cw = generate_Sn(N, 1, B)
        ww = DNC.writeweight(cw, a, gw, ga) 
        @test ww == cw
        @test eltype(ww) == Float32
    end
end

@testset "Temporal linkage" begin
    @testset "Precedence weights p⃗" begin
        N, B = 4, 1
        ww = rand(Float32, N, 1, B)
        ww[:, :, 1] = [0.1f0, 0.2f0, 0.3f0, 0.4f0]
        p0 = zeros(Float32, N, B)
        p = DNC.precedenceweight(p0, ww) 
        @test p == dropdims(ww; dims=2)
        @test eltype(p) == Float32
    end

    # Write weights. Writes to location 1, 2, and 3 iteratively.
    N, R, B = 3, 1, 1
    ww = rand(Float32, N, 1, B)
    wwlist = [[1.0f0, 0.0f0, 0.0f0],
            [0.0f0, 1.0f0, 0.0f0],
            [0.0f0, 0.0f0, 1.0f0]]
    # Temporal link matrix
    L = zeros(Float32, N, N, B)
    # precedence weights
    p = zeros(Float32, N, B)
    # Expected evolution of L
    expected = [zeros(Float32, N, N, B) for i in 1:3]
    expected[2][2, 1, 1] = 1.0f0
    expected[3][2, 1, 1] = 1.0f0
    expected[3][3, 2, 1] = 1.0f0

    @testset "Link matrix L" begin
        for i in 1:length(wwlist)
            ww[:, :, 1] = wwlist[i]
            DNC.updatelinkmatrix!(L, p, ww)
            @test L == expected[i]
            @test eltype(L) == Float32
            p = DNC.precedenceweight(p, ww)
        end
        # Last read was location 2. The forward weight should point to 3, backward to 1.
        wr = rand(Float32, N, R, B)
        wr[:, :, 1] = [0.f0, 1, 0]
        forward = DNC.forwardweight(L, wr)
        backward = DNC.backwardweight(L, wr)
        @test  forward[:, 1] == [0, 0, 1]
        @test backward[:, 1] == [1, 0, 0]
        @test eltype(forward) == Float32
        @test eltype(backward) == Float32
    end

    @testset "Read weighting w_t^r" begin
        b = reshape([0.f0, 0, 1], 3, 1, 1)
        f = reshape([1.f0, 0, 0], 3, 1, 1)
        c = reshape([0.f0, 1, 0], 3, 1, 1)
        readmode = reshape([0.f0, 1, 0], 3, 1, 1)
        readw = DNC.readweight(b, c, f, readmode)
        @test readw == c
        @test in_Δn(readw)
        @test eltype(readw) == Float32
        readmode2 = reshape([0.5f0, 0.5f0, 0.0f0], 3, 1, 1)
        readw = DNC.readweight(b, c, f, readmode2)
        @test readw == 0.5f0b .+ 0.5f0c
    end

end
