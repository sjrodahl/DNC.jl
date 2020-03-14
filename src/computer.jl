using Flux: @functor
using Flux: softmax, σ

import Flux.hidden, Flux.trainable


mutable struct DNCCell
    controller
    readvectors
    Wr
    R::Int
    W::Int
    X::Int
    Y::Int
    memoryaccess::MemoryAccess
end

DNCCell(controller, in::Int, out::Int, N::Int, W::Int, R::Int; init=Flux.glorot_uniform) =
    DNCCell(
        controller,
        zeros(R*W),
        init(out, R*W),
        R,
        W,
        in,
        out,
        MemoryAccess(N, W, R)
    )

DNCCell(in::Int, out::Int, N::Int, W::Int, R::Int; init=Flux.glorot_uniform) =
    DNCCell(
        LSTM(inputsize(in, R, W), outputsize(R, N, W, in, out)),
        in, out, N, W, R; init=init
    )

function (m::DNCCell)(h, x)
    out = m.controller([x;h])
    v = out[1:m.Y]
    ξ = out[m.Y+1:end]
    inputs = split_ξ(ξ, m.R, m.W)
    r = m.memoryaccess(inputs)
    return r, calcoutput(v, r, m.Wr)
end

hidden(m::DNCCell) = m.readvectors

@functor DNCCell
trainable(m::DNCCell) = m.controller, m.Wr

import Base.show
function Base.show(io::IO, l::DNCCell)
    print(io, "DNCCell($(l.X), $(l.Y))\n")
    print(io, "Memory size: ($(size(l.M, 1)), $(size(l.M, 2)))\n")
    print(io, "Read heads: $(l.R)")
end


"""
    Dnc(controller, in::Integer, out::Integer, N::Integer, W::Integer, R::Integer)

Initialise a Differentiable Neural Computer with memory size (N, W) and R read heads.

"""
Dnc(a...; ka...) = Flux.Recur(DNCCell(a...; ka...))
