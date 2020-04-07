using Flux
using Flux: params

using Random
rng = MersenneTwister(2345)

@testset "Computer runs without error" begin
    R = 1
    N = 5
    W = 10
    X = 2
    Y = 2
    inputsize = DNC.inputsize(X, R, W)
    outputsize = DNC.outputsize(R, N, W, X, Y)
    x = [1.f0, 2]
    y = [1.f0, 2]
    controller = Chain(Dense(inputsize, R*W*2, relu),
                       Dense(R*W*2, R*W*2, relu),
                       Dense(R*W*2, outputsize, relu))
    @test eltype(controller(rand(rng, Float32, inputsize))) == Float32
    controller2 = LSTM(inputsize, outputsize)
    @test eltype(controller2(rand(rng, Float32, inputsize))) == Float32
    dnc1 = Dnc(controller, X, Y, N, W, R)
    out1 = dnc1(x)
    @test length(out1) == Y
    @test eltype(out1) == Float32
    dnc2 = Dnc(controller2, X, Y, N, W, R)
    out2 = dnc2(x)
    @test length(out2) == Y
    @test eltype(out2) == Float32
    @testset "Get gradient" begin
        for dnc in (dnc1, dnc2)
            g = gradient(x->sum(dnc(x)), x)
            @test !isnothing(g)
        end
    end
end

@testset "Learn" begin
    R = 2
    N = 10
    W = 16
    X = 4
    Y = 4
    inputsize = DNC.inputsize(X, R, W)
    outputsize = DNC.outputsize(R, N, W, X, Y)

    controller = LSTM(inputsize, outputsize)
    dnc = Dnc(controller, X, Y, N, W, R)

    function generatedata(in, out, n=100)
        w = rand(rng, Float32, out, in)
        b = rand(rng, Float32, out)
        Xs = [rand(rng, Float32, in) for i in 1:n]
        Ys = [w*x .+ b for x in Xs]
        zip(Xs, Ys)
    end

    data = generatedata(X, Y)
    @test eltype(dnc.cell.memoryaccess.M) == Float32
    @test eltype(dnc(first(data)[1])) == Float32
    loss(x, y) = Flux.mse(dnc(x), y)
    opt = ADAM(0.01)
    evalcb() = @show loss(first(data)[1], first(data)[2])
    Flux.train!(loss, params(dnc), data, opt, cb=Flux.throttle(evalcb, 10))
    @test eltype(dnc(first(data)[1])) == Float32
    @test eltype(dnc.cell.memoryaccess.M) == Float32
    @test loss(first(data)[1], first(data)[2]) < 1.0
end
