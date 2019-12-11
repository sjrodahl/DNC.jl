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

@with_kw mutable struct State{M<:AbstractArray, A<:AbstractArray, A2<:AbstractArray}
    L::M
    p::A
    u::A
    w_w::A
    w_r::A2
end

State(N::Int, R::Int) = State(
    L=zeros(N, N),
    p = zeros(N),
    u = zeros(N),
    w_w = zeros(N),
    w_r = [zeros(N) for i in 1:R]
    )

@with_kw struct WriteHead
    k::Vector{Float64} # Write key
    β::Float64 # Key strength
    e::Vector{Float64} # erase
    v::Vector{Float64} # add
    g_a::Float64 # allocation gate
    g_w::Float64 # write gate
end

@with_kw struct ReadHead
    k::Vector{Float64} # read key
    β::Float64 # key strength
    f::Float64 # free gate
    π::Vector{Float64} # read mode
end


include("controller.jl")
include("access.jl")
include("addressing.jl")
include("utils.jl")

end # module
