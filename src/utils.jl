import Flux: softmax
using LinearAlgebra


cosinesim(u, v) = dot(u, v)/(norm(u)*norm(v))

weighted_softmax(xs, weight) = softmax(xs.*weight)

oneplus(x) = 1 + log(1+exp(x))

inputsize(X::Int, R::Int, W::Int) = X + R * W
outputsize(R::Int, N::Int, W::Int, X::Int, Y::Int) = W*R + 3W + 5R +3 + Y

function calcoutput(v, r, W_r)
    return v .+ W_r*r
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
    k_r = [ξ[((r-1)*W+1):r*W] for r in 1:R]
    β_r = ξ[(R*W+1):(R*W+R)]
    k_w = ξ[(R*W+1+R):(R*W+R+W)]
    β_w = ξ[(R*W+R+W+1)]
    ê = ξ[(R*W+R+W+2):(R*W+R+2W+1)]
    v = ξ[(R*W+R+2W+2):(R*W+R+3W+1)]
    f̂ = ξ[(R*W+R+3W+2):(R*W+2R+3W+1)]
    ĝ_a = ξ[(R*W+2R+3W+2)]
    ĝ_w = ξ[(R*W+2R+3W+3)]
    rest = ξ[(R*W+2R+3W+4):length(ξ)]
    readmode = [rest[((r-1)*3+1):3r] for r in 1:R]
    rhs = [ReadHead(
            k=k_r[i],
            β=β_r[i],
            f=σ(f̂[i]),
            π=Flux.softmax(readmode[i])) for i in 1:R]
    wh = WriteHead(
            k=k_w,
            β=β_w,
            e=σ.(ê),
            v=v,
            g_a = σ(ĝ_a),
            g_w = σ(ĝ_w)
    )
    return (rhs, wh)
end
