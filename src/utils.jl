import Flux: softmax, identity, σ
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
cumprodexclusive(arr::AbstractArray) = cumprod(arr) ./ arr

function inputmappings(numinputs,R, W)
    lin(outsize; activation=identity) = Dense(numinputs, outsize, activation)
    function lin(firstdim, seconddim; activation=identity)
        transformed  = Dense(numinputs, firstdim * seconddim, activation)
        Chain(transformed, x-> reshape(x, firstdim, seconddim, :))
    end
    (v = lin(W),
    e = lin(W; activation=σ),
    f = lin(R; activation=σ),
    ga = lin(1; activation=σ),
    gw = lin(1; activation=σ),
    readmode = lin(3, R),
    kr = lin(W, R),
    βr = lin(R; activation=oneplus),
    kw = lin(W, 1),
    βw = lin(1; activation=oneplus))
end

function split_ξ(ξ, transformfuncs)
    v = transformfuncs.v(ξ)
    e = transformfuncs.e(ξ)
    f = transformfuncs.f(ξ)
    ga = transformfuncs.ga(ξ)
    gw = transformfuncs.gw(ξ)
    readmode = transformfuncs.readmode(ξ)
    kr = transformfuncs.kr(ξ)
    βr = transformfuncs.βr(ξ)
    kw = transformfuncs.kw(ξ)
    βw = transformfuncs.βw(ξ)
    return (
        kr = kr,
        βr = βr,
        kw = kw,
        βw = βw,
        v = v,
        e = e,
        f = f,
        ga = ga,
        gw = gw,
        readmode = softmax(readmode; dims=1) 
    )
end




