using Zygote: @adjoint


mutable struct State{A<:AbstractArray, V<:AbstractArray}
    L::Matrix
    p::V
    u::V
    ww::V
    wr::A
end

State(N::Int, R::Int) = State(
    zeros(N, N),
    zeros(N),
    zeros(N),
    zeros(N),
    [zeros(N) for i in 1:R]
    )

struct WriteHead{A<:AbstractArray, T<:Real}
    k::A # Write key
    β::T # Key strength
    e::A # erase
    v::A # add
    ga::T # allocation gate
    gw::T # write gate
end

struct ReadHead{A<:AbstractArray, T<:Real}
    k::A # read key
    β::T # key strength
    f::T # free gate
    π::A # read mode
end

# L should be updated before this
function readmem(M, rh::ReadHead, L::Matrix, prev_wr)
    k, β, π = rh.k, rh.β, rh.π
    cr = contentaddress(k, M, β)
    b = backwardweight(L, prev_wr)
    f = forwardweight(L, prev_wr)
    wr = readweight(b, cr, f, π)
    r = M' * wr
    r
end

function writemem(M,
        wh::WriteHead,
        free::AbstractArray,
        prev_ww::AbstractArray,
        prev_wr::AbstractArray,
        prev_usage::AbstractArray)
    k, β, ga, gw, e, v = wh.k, wh.β, wh.ga, wh.gw, wh.e, wh.v
    cw = contentaddress(k, M, β)
    𝜓 = memoryretention(prev_wr, free)
    u = usage(prev_usage, prev_ww, 𝜓)
    a = allocationweighting(u)
    ww = writeweight(cw, a, gw, ga)
    newmem = eraseandadd(M, ww, e, v)
    newmem
end

function update_state_after_write!(state::State, M, wh::WriteHead, free::AbstractArray)
    cw = contentaddress(wh.k, M, wh.β)
    𝜓 = memoryretention(state.wr, free)
    u = usage(state.u, state.ww, 𝜓)
    a = allocationweighting(u)
    ww = writeweight(cw, a, wh.gw, wh.ga)
    state.u = u
    state.ww = ww
    updatelinkmatrix!(state.L, state.p, state.ww)
    state.p = precedenceweight(state.p, state.ww)
end

@adjoint update_state_after_write!(state::State, M, wh::WriteHead, free::AbstractArray) =
    update_state_after_write!(state, M, wh, free), _ -> nothing

function update_state_after_read!(state::State, M, rhs::AbstractArray)
    function new_wr(L, old_wr, M, rh)
        cr = contentaddress(rh.k, M, rh.β)
        b = backwardweight(L, old_wr)
        f = forwardweight(L, old_wr)
        wr = readweight(b, cr, f, rh.π)
        wr
    end
    state.wr = [new_wr(state.L, state.wr[i], M, rhs[i]) for i in 1:length(rhs)]
    state
end

@adjoint update_state_after_read!(state::State, M, rhs::AbstractArray) =
    update_state_after_read!(state, M, rhs), _ -> nothing

eraseandadd(M, ww, e, a) = M .* (ones(size(M)) - ww * e') + ww * a'
