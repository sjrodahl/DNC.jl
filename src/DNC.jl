module DNC

using Flux
using LinearAlgebra
using Parameters

export contentaddress
export predict
export readmem
export writemem
export ReadHead
export State
export WriteHead
export Dnc
export DNCCell

@with_kw mutable struct State{A<:AbstractArray, V<:AbstractArray}
    L::Matrix
    p::V
    u::V
    w_w::V
    w_r::A
end

State(N::Int, R::Int) = State(
    L=zeros(N, N),
    p = zeros(N),
    u = zeros(N),
    w_w = zeros(N),
    w_r = [zeros(N) for i in 1:R]
    )

@with_kw struct WriteHead{A<:AbstractArray, T<:Real}
    k::A # Write key
    β::T # Key strength
    e::A # erase
    v::A # add
    g_a::T # allocation gate
    g_w::T # write gate
end

@with_kw struct ReadHead{A<:AbstractArray, T<:Real}
    k::A # read key
    β::T # key strength
    f::T # free gate
    π::A # stractArray{A, 1} # read mode
end


include("computer.jl")
include("access.jl")
include("addressing.jl")
include("utils.jl")

end # module
