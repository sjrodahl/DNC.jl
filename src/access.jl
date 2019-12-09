
mutable struct State
    L::Matrix
    p::Array
    u::Array
    w_w::Array
    w_r::Array
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

function writemem!(M, interface, state)
    c_w = contentaddress(interface.k_w, M, interface.β_w)
    𝜓 = memoryretention(state.w_r, interface.free)
    u = usage(state.u, state.w_w, 𝜓)
    a = allocationweighting(u)
    w_w = writeweight(c_w, a, interface.alloc_gate, interface.write_gate)
    erase_and_add!(M, w_w, interface.erase, interface.add)
    state.u = u
    state.w_w = w_w
end

function erase_and_add!(M, w_w, e, a)
    M .*= ones(size(M)) - w_w*e'
    M .+= w_w*a'
end
