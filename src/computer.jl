using Flux: softmax, σ
using Parameters

import Flux.hidden


@with_kw mutable struct DNCCell{V, A}
    controller
    readvectors::V
    Wr::A
    M::A
    R::Int
    W::Int
    X::Int
    Y::Int
    state::State
end

DNCCell(controller, in::Int, out::Int, N::Int, W::Int, R::Int; init=Flux.glorot_uniform) =
    DNCCell(
        controller=controller,
        readvectors= zeros(R*W),
        Wr=init(out, R*W),
        M=init(N, W),
        R=R,
        W=W,
        X=in,
        Y=out,
        state=State(N, R)
    )

function (m::DNCCell)(h, x)
    @unpack L, w_w, w_r, u = m.state
    out = m.controller([x;h])
    v = out[1:m.Y]
    ξ = out[m.Y+1:length(out)]
    rhs, wh = split_ξ(ξ, m.R, m.W)
    m.M = writemem(m.M, wh, rhs, w_w, w_r, u)
    update_state_after_write!(m.state, m.M, wh, [rh.f for rh in rhs])
    r = [readmem(m.M, rhs[i], L, w_r[i]) for i in 1:m.R]
    update_state_after_read!(m.state, m.M, rhs)
    m.readvectors = vcat(r...) # Flatten list of lists
    return m.readvectors, calcoutput(v, r, m.Wr)
end

hidden(m::DNCCell) = m.readvectors


"""
    Dnc(controller, in::Int, out::Int, N::Int, W::Int, R::Int)

Initialise a Differentiable Neural Computer with memory size (N, W) and R read heads.

"""
Dnc(a...; ka...) = Flux.Recur(DNCCell(a...; ka...))
