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

#"""
#Assuming R=1. Otherwise, the code will break
#"""
#function split_ξ(ξ, W::Int)
#    R = 1
#    length(ξ) != (W*R)+3W+5R+3 &&
#        error("Length of xi-vector is incorrect. Expected $((W*R)+3W+5R+3), got $(length(ξ))")
#    kr = ξ[1:W]
#    βr = ξ[W+1]
#    kw = ξ[(W+2):(2W+1)]
#    βw = ξ[2W+2]
#    ê = ξ[(2W+3):(3W+2)]
#    v = ξ[(3W+3):(4W+2)]
#    f̂ = ξ[4W+3]
#    ĝa = ξ[4W+4]
#    ĝw = ξ[4W+5]
#    readmode = ξ[(4W+6):length(ξ)]
#    rh = ReadHead(kr, βr, σ(f̂), softmax(readmode))
#    wh = WriteHead(kw, βw, σ.(ê), v, σ(ĝa), σ(ĝw))
#    return (rh, wh)
#end
#

function split_ξ(ξ, R::Int, W::Int)
    length(ξ) != (W*R)+3W+5R+3 &&
        error("Length of xi-vector is incorrect. Expected $((W*R)+3W+5R+3), got $(length(ξ))")
    # read keys
    kr = [ξ[((r-1)*W+1):r*W] for r in 1:R]
    βr = ξ[(R*W+1):(R*W+R)]
    kw = ξ[(R*W+1+R):(R*W+R+W)]
    βw = ξ[(R*W+R+W+1)]
    ê = ξ[(R*W+R+W+2):(R*W+R+2W+1)]
    v = ξ[(R*W+R+2W+2):(R*W+R+3W+1)]
    f̂ = ξ[(R*W+R+3W+2):(R*W+2R+3W+1)]
    ĝa = ξ[(R*W+2R+3W+2)]
    ĝw = ξ[(R*W+2R+3W+3)]
    rest = ξ[(R*W+2R+3W+4):length(ξ)]
    readmode = [rest[((r-1)*3+1):3r] for r in 1:R]
    rhs = [ReadHead(
            kr[i],
            βr[i],
            σ(f̂[i]),
            Flux.softmax(readmode[i])) for i in 1:R]
    wh = WriteHead(
            kw,
            βw,
            σ.(ê),
            v,
            σ(ĝa),
            σ(ĝw)
    )
    return (rhs, wh)
end
