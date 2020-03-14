import Flux: softmax
using LinearAlgebra


cosinesim(u, v) = dot(u, v)/(norm(u)*norm(v))

weightedsoftmax(xs, weight) = softmax(xs.*weight)

oneplus(x) = 1 + log(1+exp(x))

inputsize(X::Int, R::Int, W::Int) = X + R * W
outputsize(R::Int, N::Int, W::Int, X::Int, Y::Int) = W*R + 3W + 5R +3 + Y

function calcoutput(v, r, Wr)
    return v .+ Wr*r
end

function split_ξ(ξ, R, W)
    lin(outsize) = Dense(size(ξ)[1], outsize)(ξ)
    function lin(firstdim, seconddim)
        transformed  = Dense(size(ξ)[1], firstdim * seconddim)(ξ)
        transformed = reshape(transformed, :, firstdim, seconddim)
        transformed
    end
    v = lin(W)
    ê = lin(W)
    f̂ = lin(R)
    ĝa = lin(1)
    ĝw = lin(1)
    readmode = lin(R, 3)
    kr = lin(R, W)
    βr = lin(1)
    kw = lin(W)
    βw = lin(1)
    return Dict(
        :kr => kr,
        :βr => βr,
        :kw => kr,
        :βw => βw,
        :v => v,
        :e => σ.(ê),
        :f => σ.(f̂),
        :ga => σ.(ĝa),
        :gw => σ.(ĝw),
        :π => Flux.softmax(readmode; dims=3) 
    )
end




