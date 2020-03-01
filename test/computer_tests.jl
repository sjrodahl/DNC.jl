using Flux

@testset "Computer runs without error" begin
    R = 1
    N = 5
    W = 10
    X = 2
    Y = 2
    inputsize = DNC.inputsize(X, R, W)
    outputsize = DNC.outputsize(R, N, W, X, Y)
    x = [1, 2]
    y = [1, 2]
    controller = Chain(Dense(inputsize, R*W*2, relu),
                       Dense(R*W*2, R*W*2, relu),
                       Dense(R*W*2, outputsize, relu))
    controller2 = LSTM(inputsize, outputsize)
    dnc1 = Dnc(controller, X, Y, N, W, R)
    @test length(dnc1(x)) == Y
    dnc2 = Dnc(controller2, X, Y, N, W, R)
    @test length(dnc2(x)) == Y
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

    function generate_data(in, out, n=100)
        w = rand(out, in)
        b = rand(out)
        Xs = [rand(in) for i in 1:n]
        Ys = [w*x .+ b for x in Xs]
        zip(Xs, Ys)
    end

    data = generate_data(X, Y, 100)
    loss(x, y) = Flux.mse(dnc(x), y)
    opt = ADAM(0.01)
    evalcb() = @show loss(first(data)[1], first(data)[2])
    Flux.train!(loss, params(dnc), data, opt, cb=Flux.throttle(evalcb, 10))
    @test loss(first(data)[1], first(data)[2]) < 1.0
end
