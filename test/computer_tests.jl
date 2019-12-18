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
    R = 4
    outputsize = DNC.outputsize(R, N, W, X, Y)
    controller3(x) = rand(outputsize)
    dnc3 = Dnc(controller3, X, Y, N, W, R)
    @test length(dnc3(x)) == Y
end

"""
@testset "Learn" begin
    R = 2
    N = 10
    W = 16
    X = 4
    Y = 4
    inputsize = DNC.inputsize(X, R, W)
    outputsize = DNC.outputsize(R, N, W, X, Y)
    function constructdata(size, numsamples)
        function genitem(size)
            i = rand(size)
            (i, i)
        end
        [genitem(size) for i in 1:numsamples]
    end
    data = constructdata(X, 10000)
    controller = LSTM(inputsize, outputsize)
    dnc = Dnc(controller, X, Y, N, W, R)
    loss(x, y) = Flux.mse(dnc(x), y)
    test_x, test_y = (accumulate(+, ones(X)), accumulate(+, ones(X)))
    opt = ADAM(0.01)
    evalcb = @show loss(test_x, test_y)
    Flux.train!(loss, params(dnc), data, opt, cb=Flux.throttle(evalcb, 5))
    @test_broken loss(test_x, test_y) < 0.1
end
"""
