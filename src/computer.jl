using Flux: @functor
using Flux: softmax, σ

import Flux.hidden, Flux.trainable, Flux.LSTMCell


mutable struct DNCCell{C, C2, T, S, M}
    controller::C
    readvectors::S
    outputlayer::C2
    memoryaccess::MemoryAccess{M, T, S}
end

DNCCell(controller, in::Int, out::Int, controut::Int, N::Int, W::Int, R::Int, B::Int; init=Flux.glorot_uniform) = 
    DNCCell(
        controller,
        zeros(Float32, R*W, B),
        Dense(controut+R*W, out),
        MemoryAccess(controut, N, W, R, B; init=init))

DNCCell(in::Int, out::Int, controut::Int, N::Int, W::Int, R::Int, B::Int; init=Flux.glorot_uniform) = 
    DNCCell(
        MyLSTM(B, inputsize(in, R, W), controut),
        zeros(Float32, R*W, B),
        Dense(controut+R*W, out),
        MemoryAccess(controut, N, W, R, B; init=init))


function (m::DNCCell)(h, x)
    out = m.controller([x;h])
    r = m.memoryaccess(out)
    r = reshape(r, size(r,1)*size(r, 2), size(r, 3))
    return r, m.outputlayer([out;r])
end

hidden(m::DNCCell) = m.readvectors

trainable(m::DNCCell) = (m.controller, m.outputlayer, trainable(m.memoryaccess))
@functor DNCCell

import Base.show
function Base.show(io::IO, l::DNCCell)
    readvecsize = size(l.readvectors, 1)
    controllerin = size(l.controller.cell.Wi, 2)
    X = controllerin - readvecsize
    Y = size(l.outputlayer.W, 1)
    print(io, "DNCCell($(X), $(Y))")
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
