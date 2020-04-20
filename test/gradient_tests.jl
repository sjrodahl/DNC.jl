using Random
using Zygote


# Copied from Zygote's test code
function ngradient(f, xs::AbstractArray...)
  grads = zero.(xs)
  for (x, Δ) in zip(xs, grads), i in 1:length(x)
    δ = sqrt(eps())
    tmp = x[i]
    x[i] = tmp - δ/2
    y1 = f(xs...)
    x[i] = tmp + δ/2
    y2 = f(xs...)
    x[i] = tmp
    Δ[i] = (y2-y1)/δ
  end
  return grads
end

function gradcheck(f, xs...)
    res = all(isapprox.(ngradient(f, xs...),
                   gradient(f, xs...), rtol = 1e-3, atol = 1e-3))
    res
end

gradtest(f, xs::AbstractArray...) = gradcheck((xs...) -> sum(sin.(f(xs...))), xs...)
gradtest(f, dims...) = gradtest(f, rand.(Float64, dims)...)



function gradtestnoerror(f, xs...)
    try
        gradient(f, xs...)
        return true
    catch e
        throw(e)
        return false
    end
end


Random.seed!(0)

X, Y, N, W, R, B = 2, 2, 3, 8, 2, 4
controut = 16
 
# Dimension map
d = Dict(
    :kr => (W, R, B),
    :βr => (R, B),
    :kw => (W, 1, B),
    :βw => (1, B),
    :wr => (N, R, B),
    :ww => (N, 1, B),
    :u => (N, B),
    :f => (1, B),
    :g => (1, B),
    :π => (3, R, B),
    :L => (N, N, B),
    :M => (N, W, B),
    :controllerout => (Y, B),
    :readvec => (R*W, B),
    :Wr => (Y, R*W, B),
    :ξ => (DNC.outputsize(R, N, W, X, Y)-Y, B),
    :erase => (W, B),
   )
    
tr = DNC.inputmappings(d[:ξ][1], R, W)
inputs = DNC.split_ξ(rand(d[:ξ]...), tr)

@testset "Addressing" begin
    @test gradtest(DNC.contentaddress, d[:kr], d[:M], d[:βr])
    @test gradtest(DNC.contentaddress, d[:kw], d[:M], d[:βw])
    @test gradtest(DNC.memoryretention, d[:wr], d[:f])
    @test gradtest(DNC.usage, d[:u], d[:ww], d[:wr], d[:f])
    @test gradtest(DNC.writeweight, d[:ww], d[:u], d[:g], d[:g])
    @test gradtest(DNC.precedenceweight, d[:u], d[:ww])
    @test gradtest(DNC.forwardweight, d[:L], d[:wr])
    @test gradtest(DNC.backwardweight, d[:L], d[:wr])
    @test gradtest(DNC.readweight, d[:wr], d[:wr], d[:wr], d[:π])
end

@testset "Utils" begin
    @test gradtest(x -> DNC.oneplus.(x), [-10.0, -1, 0, 1, 10])
    @test gradtest(DNC.calcoutput, d[:controllerout], d[:readvec], d[:Wr])
    @test_broken gradtest(x->DNC.split_ξ(x, tr).kr, d[:ξ])
end

M = rand(Float32, d[:M]...)
L = rand(Float32, d[:L]...)
wr = rand(Float32, d[:wr]...)
u = rand(Float32, d[:u]...)
ma = DNC.MemoryAccess(d[:ξ][1], N, W, R, B)

@testset "Access" begin
    @test gradtest(DNC.eraseandadd, d[:M], d[:ww], d[:erase], d[:erase])
    @test gradtest(DNC.readmem, d[:M], d[:wr])
    # readweights and writeweights use inputs-tuple as argument, so gradtest won't work.
    @test gradtestnoerror((m, i, u) ->sum(DNC.writeweights(m, i, u)), M, inputs, u)
    @test gradtestnoerror((m, i, l, wr) -> sum(DNC.readweights(m, i, l, wr)), M, inputs, L, wr)
    @test gradtestnoerror(x->sum(ma(x)), rand(Float32, d[:ξ]...))
end

model = Dnc(X, Y, controut, N, W, R, B)

@testset "Computer" begin
    @test gradtestnoerror(x->sum(model(x)), rand(Float32, X, B))
end
