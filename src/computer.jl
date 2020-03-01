using Flux: @functor
using Flux: softmax, σ

import Flux.hidden


mutable struct DNCCell
    controller
    readvectors
    Wr
    M
    R::Int
    W::Int
    X::Int
    Y::Int
    state::State
end

DNCCell(controller, in::Int, out::Int, N::Int, W::Int, R::Int; init=Flux.glorot_uniform) =
    DNCCell(
        controller,
        zeros(R*W),
        init(out, R*W),
        init(N, W),
        R,
        W,
        in,
        out,
        State(N, R)
    )

DNCCell(in::Int, out::Int, N::Int, W::Int, R::Int; init=Flux.glorot_uniform) =
    DNCCell(
        LSTM(inputsize(in, R, W), outputsize(R, N, W, in, out)),
        in, out, N, W, R; init=init
    )

function (m::DNCCell)(h, x)
    L, ww, wr, u = m.state.L, m.state.ww, m.state.wr, m.state.u
    numreads = m.R
    out = m.controller([x;h])
    v = out[1:m.Y]
    ξ = out[m.Y+1:length(out)]
    rhs, wh = splitparams(ξ, numreads, m.W)
    freegate = [rh.f for rh in rhs]
    m.M = writemem(m.M, wh, freegate, ww, wr, u)
    update_state_after_write!(m.state, m.M, wh, freegate)
    r = [readmem(m.M, rh, L, wr[1]) for rh in rhs]
    r = vcat(r...)
    update_state_after_read!(m.state, m.M, rhs)
    m.readvectors = r # Flatten list of lists
    return r, calcoutput(v, r, m.Wr)
end

hidden(m::DNCCell) = m.readvectors

@functor DNCCell controller, Wr

"""
    Dnc(controller, in::Int, out::Int, N::Int, W::Int, R::Int)

Initialise a Differentiable Neural Computer with memory size (N, W) and R read heads.

"""
Dnc(a...; ka...) = Flux.Recur(DNCCell(a...; ka...))
