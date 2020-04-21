using Zygote: @adjoint
import Flux: trainable


mutable struct State{T, S}
    L::T
    p::S
    u::S
    ww::T
    wr::T
end


State(N::Int, R::Int, B::Int) = State(
    zeros(Float32, N, N, B),
    zeros(Float32, N, B),
    zeros(Float32, N, B),
    zeros(Float32, N, 1, B),
    zeros(Float32, N, R, B)
    )


mutable struct MemoryAccess{R, T, S}
    M::T
    state::State{T, S}
    inputmaps::R
end


MemoryAccess(inputsize, N, W, R, B; init=Flux.glorot_uniform) = 
    MemoryAccess(init(N, W, B), State(N, R, B), inputmappings(inputsize, R, W))


trainable(ma::MemoryAccess) = ma.inputmaps

function (ma::MemoryAccess)(inputs)
    p_prev, u_prev, ww_prev, wr_prev = ma.state.p, ma.state.u, ma.state.ww, ma.state.wr
    inputs = split_ξ(inputs, ma.inputmaps)
    u = usage(u_prev, ww_prev, wr_prev, inputs.f)
    ww = writeweights(ma.M, inputs, u)
    ma.M = eraseandadd(ma.M, ww, inputs.e, inputs.v)
    update_state_after_write!(ma.state, ww, u)
    wr = readweights(ma.M, inputs, ma.state.L, wr_prev)
    update_state_after_read!(ma.state, wr)
    readvectors = readmem(ma.M, wr)
    readvectors
end

function readmem(M::AbstractArray{T, 3}, wr::AbstractArray{T, 3}) where T
    batched_mul(PermutedDimsArray(M, (2, 1, 3)), wr)
end

"""
    readweights(M, inputs, L, prev_wr)

# Arguments
- `M`: (N, W, B) memory
- `inputs`: Named Tuple of controller parameters
- `L`: (N, N, B) link matrix
- `prev_wr`: (N, R, B) last iterations read weights
"""
function readweights(M, inputs, L, prev_wr)
    k, β, readmode= inputs.kr, inputs.βr, inputs.readmode
    cr = contentaddress(k, M, β)
    b = backwardweight(L, prev_wr)
    f = forwardweight(L, prev_wr)
    wr = readweight(b, cr, f, readmode)
end

"""
    writeweights(M, inputs, usage)

Fuzzy write to memory. Location is based on either content similarity or row usage.

"""
function writeweights(M, inputs, usage)
    k, β, ga, gw = inputs.kw, inputs.βw, inputs.ga, inputs.gw
    cw = contentaddress(k, M, β)
    a = allocationweighting(usage)
    ww = writeweight(cw, a, gw, ga)
end

function update_state_after_write!(state, ww, usage)
    state.u = usage
    state.ww = ww
    updatelinkmatrix!(state.L, state.p, state.ww)
    state.p = precedenceweight(state.p, state.ww)
end

@adjoint update_state_after_write!(state::State, ww, usage) =
    update_state_after_write!(state, ww, usage), _ -> nothing

function update_state_after_read!(state, wr)
    state.wr = wr
end

@adjoint update_state_after_read!(state, wr) =
    update_state_after_read!(state, wr), _ -> nothing

function eraseandadd(M::AbstractArray{T, 3}, ww::AbstractArray{T, 3}, e::AbstractArray{T,2}, a::AbstractArray{T, 2}) where T
    e = reshape(e, size(e, 1), 1, size(e, 2))
    a = reshape(a, size(a, 1), 1, size(a, 2))
    erasematrix = batched_mul(ww, PermutedDimsArray(e, (2, 1, 3)))
    addmatrix = batched_mul(ww, PermutedDimsArray(a, (2, 1, 3)))
    @cast newM[i, j, k] := M[i, j, k] * (1-erasematrix[i, j, k]) + addmatrix[i, j, k]
end

