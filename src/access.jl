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


function readmem(M, interface, state)
    c_r = contentaddress(interface.k_r, M, interface.β_r)
    updatelinkmatrix!(state.L, state.p, state.w_w)
    state.p = precedenceweight(state.p, state.w_w)
    b = backwardweight(state.L, state.w_r)
    f = forwardweight(state.L, state.w_r)
    w_r = readweight(b, c_r, f, interface.readmode)
    state.w_r = w_r
    r = M' * w_r
    r
end

function writemem(M, interface, state)
    c_w = contentaddress(interface.k_w, M, interface.β_w)
    𝜓 = memoryretention(state.w_r, interface.free)
    u = usage(state.u, state.w_w, 𝜓)
    a = allocationweighting(u)
    w_w = writeweight(c_w, a, interface.write_gate, interface.alloc_gate)
    newmem = erase_and_add(M, w_w, interface.erase, interface.add)
    newmem
end

function writemem(M::Matrix,
        wh::WriteHead,
        rhs::Vector{ReadHead},
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

function updatestate!(state, M, wh::WriteHead, rh::Vector{ReadHead})
    free = [rh.f for rh in rhs]
    c_w = contentaddress(wh.k, M, wh.β)
    𝜓 = memoryretention(state.w_r, free)
    u = usage(state.u, state.w_w, 𝜓)
    a = allocationweighting(u)
    w_w = writeweight(c_w, a, wh.g_w, wh.g_a)
    state.u = u
    state.w_w = w_w
    updatelinkmatrix!(state.L, state.p, state.w_w)
    state.p = precedenceweight(state.p, state.w_w)
    b = backwardweight(state.L, state.w_r)
    f = forwardweight(state.L, state.w_r)
    w_r = [readweight(b, c_r, f, rh.π) for rh in rhs]
    state.w_r = w_r
end

function updatestate!(state, M, interface)
    c_w = contentaddress(interface.k_w, M, interface.β_w)
    𝜓 = memoryretention(state.w_r, interface.free)
    u = usage(state.u, state.w_w, 𝜓)
    a = allocationweighting(u)
    w_w = writeweight(c_w, a, interface.write_gate, interface.alloc_gate)
    state.u = u
    state.w_w = w_w
end

erase_and_add(M, w_w, e, a) = M .* (ones(size(M)) - w_w * e') + w_w * a'
