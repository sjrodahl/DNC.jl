using Flux
using Flux: params
using Flux.Data: DataLoader

using Random
rng = MersenneTwister(2345)

@testset "Computer runs without error" for B in 1:2
    R = 1
    N = 5
    W = 10
    X = 2
    Y = 2
    inputsize = DNC.inputsize(X, R, W)
    outputsize = DNC.outputsize(R, N, W, X, Y)
    x = rand(Float32, X, B)
    y = rand(Float32, Y, B)
    controller = Chain(Dense(inputsize, R*W*2, relu),
                       Dense(R*W*2, outputsize, relu))
    controller2 = LSTM(inputsize, outputsize)
    dnc1 = Dnc(controller, X, Y, N, W, R, B)
    out1 = dnc1(x)
    @test size(out1) == (Y, B)
    @test eltype(out1) == Float32
    dnc2 = Dnc(controller2, X, Y, N, W, R, B)
    out2 = dnc2(x)
    @test size(out2) == (Y, B)
    @test eltype(out2) == Float32
end

R, N, W, X, Y, B = 2, 10, 16, 2, 2, 4
inputsize = DNC.inputsize(X, R, W)
outputsize = DNC.outputsize(R, N, W, X, Y)
dnc = Dnc(X, Y, N, W, R, B)

function generatedata(in, out, n=100)
    w = rand(rng, Float32, out, in)
    b = rand(rng, Float32, out)
    Xs = rand(rng, Float32, in, n)
    Ys = w*Xs .+ b
    Xs, Ys
end

xs, ys = generatedata(X, Y)
batcheddata = DataLoader(xs, ys, batchsize=B)
loss(x, y) = Flux.mse(dnc(x), y)
opt = ADAM(0.01)
onebatch = Base.iterate(batcheddata, 0)[1]

@testset "Learn" begin
    Flux.train!(loss, params(dnc), batcheddata, opt)
    @test eltype(dnc(onebatch[1])) == Float32
    @test eltype(dnc.cell.memoryaccess.M) == Float32
    @test loss(onebatch...) < 1.0*B
end
