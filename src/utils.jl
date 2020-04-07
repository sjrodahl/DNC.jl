import Flux: softmax
using LinearAlgebra

mynorm(itr) = sqrt(sum(x->x^2, itr))

cosinesim(u, v) = dot(u, v)/(mynorm(u)*mynorm(v))

weightedsoftmax(xs, weight) = softmax(xs.*weight)

oneplus(x) = 1 + log(1+exp(x))

inputsize(X::Int, R::Int, W::Int) = X + R * W
outputsize(R::Int, N::Int, W::Int, X::Int, Y::Int) = W*R + 3W + 5R +3 + Y

function calcoutput(v, r, Wr)
    return v .+ Wr*r
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
    lin(outsize) = Dense(numinputs, outsize)
    function lin(firstdim, seconddim)
        transformed  = Dense(numinputs, firstdim * seconddim)
        Chain(transformed, x-> reshape(x, firstdim, seconddim))
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




