using Flux

@testset "Computer runs without error" begin
    R = 1
    N = 5
    W = 10
    X = 2
    Y = 2
    outputsize = DNC.outputsize(R, N, W, X, Y)
    x = [1, 2]
    y = [1, 2]
    controller = Chain(Dense(X+R*W, R*W*2, relu),
                       Dense(R*W*2, R*W*2, relu),
                       Dense(R*W*2, outputsize, relu))
    controller2(x) = rand(outputsize)
    dnc1 = Dnc(controller, X, Y, N, W, R)
    @test_broken length(dnc1(x)) == Y
    dnc2 = Dnc(controller2, X, Y, N, W, R)
    @test length(dnc2(x)) == Y
    R = 4
    outputsize = DNC.outputsize(R, N, W, X, Y)
    controller3(x) = rand(outputsize)
    dnc3 = Dnc(controller3, X, Y, N, W, R)
    @test length(dnc3(x)) == Y
end
