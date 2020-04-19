import Flux: softmax
using LinearAlgebra
using NNlib
using TensorCast


"""
cumprodexclusive(arr::AbstractArray) 
Exclusive cumulative product

# Examples
```jldoctest
julia> DNC.cumprodexclusive([1, 2, 3, 4])
4-element Array{Float64,1}:
1.0
1.0
2.0
6.0
```
"""
cumprodexclusive(arr::AbstractArray; dims=1) = cumprod(arr; dims=dims) ./ arr

oneplus(x) = 1 + log(1+exp(x))

inputsize(X::Int, R::Int, W::Int) = X + R * W
outputsize(R::Int, N::Int, W::Int, X::Int, Y::Int) = W*R + 3W + 5R +3 + Y


function calcoutput(v::AbstractArray{T, 2}, r::AbstractArray{T, 2}, Wr::AbstractArray{T, 3}) where T
    r = reshape(r, size(r, 1), 1, size(r, 2))
    memoryoutput = dropdims(batched_mul(Wr, r); dims=2)
    v .+ memoryoutput
end


function inputmappings(numinputs,R, W)
    lin(outsize) = Dense(numinputs, outsize)
    function lin(firstdim, seconddim)
        transformed  = Dense(numinputs, firstdim * seconddim)
        Chain(transformed, x-> reshape(x, firstdim, seconddim, :))
    end
    (v = lin(W),
    ê = lin(W),
    f̂ = lin(R),
    ĝa = lin(1),
    ĝw = lin(1),
    readmode = lin(3, R),
    kr = lin(W, R),
    βr = lin(R),
    kw = lin(W, 1),
    βw = lin(1))
end

function split_ξ(ξ, transformfuncs)
    v = transformfuncs.v(ξ)
    ê = transformfuncs.ê(ξ)
    f̂ = transformfuncs.f̂(ξ)
    ĝa = transformfuncs.ĝa(ξ)
    ĝw = transformfuncs.ĝw(ξ)
    readmode = transformfuncs.readmode(ξ)
    kr = transformfuncs.kr(ξ)
    βr = transformfuncs.βr(ξ)
    kw = transformfuncs.kw(ξ)
    βw = transformfuncs.βw(ξ)
    return (
        kr = kr,
        βr = oneplus.(βr),
        kw = kw,
        βw = oneplus.(βw),
        v = v,
        e = σ.(ê),
        f = σ.(f̂),
        ga = σ.(ĝa),
        gw = σ.(ĝw),
        readmode = Flux.softmax(readmode; dims=1) 
    )
end




