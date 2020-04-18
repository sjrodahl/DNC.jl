import Flux: softmax
using LinearAlgebra
using TensorCast


"""
    mysoftmax(a)

Numerically stable method to calculate the softmax columnwise of a 3D tensor
"""
function mysoftmax(a)
    # Subtract the maximum value for numerical stability
    @cast submax[i, j, k] := a[i, j, k] - @reduce [_, j, k] := maximum(i) a[i, j, k]
    @cast r[i, j, k] := exp(submax[i, j, k]) / @reduce [_, j, k] := sum(i) exp(submax[i, j, k])
end

"""
    weightedcosinesim(a, b, β)

Compute the cosine similarity between the columns of `a` and rows of `b` weighted by `β`.

Weighted cosine similarity is defined as

```
    (dot(a, b) / ||a||*||b||) * β
```

"""
function weightedcosinesim(a, b, β)
    @reduce similarity[i, j, k] := sum(s) a[s, j, k] * b[i, s, k] /
        sqrt( @reduce [_, j, k] := sum(s') a[s', j, k]^2) /
        sqrt( @reduce [i, _, k] := sum(s'') b[i, s'', k]^2)
    @cast weighted[i, j, k] := similarity[i, j, k] * β[j, k]
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
cumprodexclusive(arr::AbstractArray; dims=1) = cumprod(arr; dims=dims) ./ arr

import Base.lastindex

Base.lastindex(b::Zygote.Buffer) = Base.lastindex(b.data)
Base.lastindex(b::Zygote.Buffer, d) = Base.lastindex(b.data, d)

oneplus(x) = 1 + log(1+exp(x))

inputsize(X::Int, R::Int, W::Int) = X + R * W
outputsize(R::Int, N::Int, W::Int, X::Int, Y::Int) = W*R + 3W + 5R +3 + Y


function calcoutput(v::AbstractArray{T, 2}, r::AbstractArray{T, 2}, Wr::AbstractArray{T, 3}) where T
    Y, _, B = size(Wr)
    out = Zygote.Buffer(v, T, (Y, B))
    @views for b in 1:B
        out[:, b] = v[:, b] .+ Wr[:, :, b]*r[:, b]
    end
    copy(out)
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




