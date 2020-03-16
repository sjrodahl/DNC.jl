using Zygote: @adjoint


mutable struct State
    L
    p
    u
    ww
    wr
end

State(N::Int, R::Int) = State(
    zeros(N, N),
    zeros(N),
    zeros(N),
    zeros(N),
    zeros(N, R)
    )

struct MemoryAccess
    M
    state
end
MemoryAccess(N, W, R; init=Flux.glorot_uniform) = MemoryAccess(init(N, W), State(N, R))

function (ma::MemoryAccess)(inputs)
    R = size(ma.state.wr)[2]
    W = size(ma.M)[2]
    inputs = split_ξ(inputs, R, W)
    p, u, ww, wr = ma.state.p, ma.state.u, ma.state.ww, ma.state.wr
    u = usage(u, ww, wr, inputs[:f])
    ww= writeweights(M, inputs, ww, wr, u)
    ma.M = eraseandadd(ma.M, ww, inputs[:e], inputs[:v])
    update_state_after_write(ma.state, ww, u)
    wr = readweights(inputs, ma.state.L, wr)
    update_state_after_read(ma.state, wr)
    readvectors = ma.M' * wr
    r
end


"""
    readweights(inputs, L::Matrix, prev_wr)

Fuzzy read the memory M. 
"""
function readweights(M, inputs, L, prev_wr)
    k, β, π = inputs[:kr], inputs[:βr], inputs[:π]
    cr = contentaddress(k, M, β)
    b = backwardweight(L, prev_wr)
    f = forwardweight(L, prev_wr)
    wr = readweight(b, cr, f, π)
end

"""
    writeweights(M, inputs, free::AbstractArray, prev_ww::AbstractArray, prev_wr::AbstractArray, prev_usage::AbstractArray)

Fuzzy write to memory. Location is based on either content similarity or row usage.

"""
function writeweights(M, inputs,
        prev_ww,
        prev_wr,
        prev_usage)
    k, β, ga, gw, e, v, free = inputs[:kw], inputs[:βw], inputs[:ga], inputs[:gw], inputs[:e], inputs[:v], inputs[:f]
    cw = contentaddress(k, M, β)
    𝜓 = memoryretention(prev_wr, free)
    u = usage(prev_usage, prev_ww, 𝜓)
    a = allocationweighting(u)
    ww = writeweight(cw, a, gw, ga)
end

function update_state_after_write!(state, ww, usage)
    state.u = u
    state.ww = ww
    updatelinkmatrix!(state.L, state.p, state.ww)
    state.p = precedenceweight(state.p, state.ww)
end

@adjoint update_state_after_write!(state::State, ww, usage) =
    update_state_after_write!(state, M, wh, free), _ -> nothing

function update_state_after_read!(state, wr)
    state.wr = wr
end

@adjoint update_state_after_read!(state, wr) =
    update_state_after_read!(state, wr), _ -> nothing

eraseandadd(M, ww, e, a) = M .* (ones(size(M)) - ww * e') + ww * a'
