using Flux: @functor
using Flux: softmax, σ

import Flux.hidden, Flux.trainable, Flux.LSTMCell


mutable struct DNCCell{C, T, S, M}
    controller::C
    readvectors::S
    Wr::T
    R::Int
    W::Int
    X::Int
    Y::Int
    memoryaccess::MemoryAccess{M, T, S}
end

DNCCell(controller, in::Int, out::Int, controut::Int, N::Int, W::Int, R::Int, B::Int; init=Flux.glorot_uniform) = 
    DNCCell(
        controller,
        zeros(Float32, R*W, B),
        init(out, R*W, B),
        R, W, in, out,
        MemoryAccess(controut-out, N, W, R, B))

DNCCell(in::Int, out::Int, controut::Int, N::Int, W::Int, R::Int, B::Int; init=Flux.glorot_uniform) = 
    DNCCell(
        MyLSTM(B, inputsize(in, R, W), controut),
        zeros(Float32, R*W, B),
        init(out, R*W, B),
        R, W, in, out,
        MemoryAccess(controut-out, N, W, R, B))


function (m::DNCCell)(h, x)
    B = size(m.Wr, 3)
    out = m.controller([x;h])
    v = view(out, 1:m.Y, :)
    ξ = view(out, (m.Y+1):size(out, 1), :)
    r = m.memoryaccess(ξ)
    r = reshape(r, size(r,1)*size(r, 2), B)
    return r, calcoutput(v, r, m.Wr)
end

hidden(m::DNCCell) = m.readvectors

trainable(m::DNCCell) = m.controller, m.Wr
@functor DNCCell

import Base.show
function Base.show(io::IO, l::DNCCell)
    print(io, "DNCCell($(l.X), $(l.Y))")
end

mutable struct MyRecur{T, S}
  cell::T
  init::S
  state::S
end

MyRecur(m, h = hidden(m)) = MyRecur(m, h, h)

function (m::MyRecur)(xs...)
  h, y = m.cell(m.state, xs...)
  m.state = h
  return y
end

@functor MyRecur cell, init

Base.show(io::IO, m::MyRecur) = print(io, "MyRecur(", m.cell, ")")



"""
    Dnc(controller, in::Integer, out::Integer, N::Integer, W::Integer, R::Integer)

Initialise a Differentiable Neural Computer with memory size (N, W) and R read heads.

"""
Dnc(a...; ka...) = MyRecur(DNCCell(a...; ka...))

function MyLSTM(batchsize, a...; ka...)
    cell = LSTMCell(a..., ka...)
    h = reshape.(repeat.(hidden(cell), batchsize), :, batchsize)
    MyRecur(cell, h)
end
