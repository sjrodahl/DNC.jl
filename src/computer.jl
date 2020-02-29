using Flux: @functor
using Flux: softmax, σ
using Parameters

import Flux.hidden


mutable struct DNCCell
    controller
    readvectors
    Wr
    M
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
        W,
        in,
        out,
        State(N, R)
    )

DNCCell(in::Int, out::Int, memsize::Tuple, R::Int; init=Flux.glorot_uniform) =
    DNCCell(
        LSTM(inputsize(in, R, memsize[2]), outputsize(R, memsize[1], memsize[2], in, out)),
        zeros(R*memsize[2]),
        init(out, R*memsize[2]),
        init(memsize...),
        memsize[2],
        in,
        out,
        State(memsize[1], R)
    )

function (m::DNCCell)(h, x)
    @unpack L, w_w, w_r, u = m.state
    out = m.controller([x;h])
    v = out[1:m.Y]
    ξ = out[m.Y+1:length(out)]
    rh, wh = split_ξ(ξ, m.W)
    freegate = [rh.f]
    m.M = writemem(m.M, wh, freegate, w_w, w_r, u)
    update_state_after_write!(m.state, m.M, wh, freegate)
    r = readmem(m.M, rh, L, w_r[1])
    update_state_after_read!(m.state, m.M, [rh])
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
