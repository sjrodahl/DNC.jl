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

# L should be updated before this
function readmem(M, rh::ReadHead, L::Matrix, prev_w_r)
    @unpack k, β, π = rh
    c_r = contentaddress(k, M, β)
    b = backwardweight(L, prev_w_r)
    f = forwardweight(L, prev_w_r)
    w_r = readweight(b, c_r, f, π)
    r = M' * w_r
    r
end

function writemem(M,
        wh::WriteHead,
        rhs::AbstractArray,
        prev_w_w::AbstractArray,
        prev_w_r::AbstractArray,
        prev_usage::AbstractArray)
    @unpack k, β, g_a, g_w, e, v = wh
    free = [rh.f for rh in rhs]
    c_w = contentaddress(k, M, β)
    𝜓 = memoryretention(prev_w_r, free)
    u = usage(prev_usage, prev_w_w, 𝜓)
    a = allocationweighting(u)
    w_w = writeweight(c_w, a, g_w, g_a)
    newmem = erase_and_add(M, w_w, e, v)
    newmem
end

function update_state_after_write!(state::State, M, wh::WriteHead, free::AbstractArray)
    c_w = contentaddress(wh.k, M, wh.β)
    𝜓 = memoryretention(state.w_r, free)
    u = usage(state.u, state.w_w, 𝜓)
    a = allocationweighting(u)
    w_w = writeweight(c_w, a, wh.g_w, wh.g_a)
    state.u = u
    state.w_w = w_w
    updatelinkmatrix!(state.L, state.p, state.w_w)
    state.p = precedenceweight(state.p, state.w_w)
end

function update_state_after_read!(state::State, M, rhs::AbstractArray)
    for i in 1:length(rhs)
        c_r = contentaddress(rhs[i].k, M, rhs[i].β)
        b = backwardweight(state.L, state.w_r[i])
        f = forwardweight(state.L, state.w_r[i])
        w_r = readweight(b, c_r, f, rhs[i].π)
        state.w_r[i] = w_r
    end
end

erase_and_add(M, w_w, e, a) = M .* (ones(size(M)) - w_w * e') + w_w * a'
