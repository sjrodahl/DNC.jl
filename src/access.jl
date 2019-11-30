
mutable struct State
    L::Matrix
    p::Array
    u::Array
    w_w::Array
    w_r::Array
end


function readmem(M, interface, state)
    c_r = contentaddress(interface.k_r, M, interface.β_r)
    @show c_r
    updatelinkmatrix!(state.L, state.p, state.w_w)
    state.p = precedenceweight(state.p, state.w_w)
    b = backwardweight(state.L, state.w_r)
    f = forwardweight(state.L, state.w_r)
    w_r = readweight(b, c_r, f, interface.readmode)
    @show(w_r)
    r = M' * w_r
    r
end
