function monitored_readweight(io::IO, backw, content, forw, readmode)
    rw = DNC.readweight(backw, content, forw, readmode)
    print(io, "readweight:",rw,"\n")
    rw
end

function monitored_readmem(io::IO, M, rh::ReadHead, L::Matrix, prev_wr)
    k, β, π = rh.k, rh.β, rh.π
    cr = DNC.contentaddress(k, M, β)
    b = DNC.backwardweight(L, prev_wr)
    f = DNC.forwardweight(L, prev_wr)
    wr = monitored_readweight(io, b, cr, f, π)
    r = M' * wr
    r
end

function monitored_writeweight(io::IO, cw, a, gw, ga)
    ww = DNC.writeweight(cw, a, gw, ga)
    print(io, "writeweight:", ww,"\n")
    ww
end


function monitored_writemem(io::IO, M,
        wh::WriteHead,
        free::AbstractArray,
        prev_ww::AbstractArray,
        prev_wr::AbstractArray,
        prev_usage::AbstractArray)
    k, β, ga, gw, e, v = wh.k, wh.β, wh.ga, wh.gw, wh.e, wh.v
    cw = DNC.contentaddress(k, M, β)
    𝜓 = DNC.memoryretention(prev_wr, free)
    u = DNC.usage(prev_usage, prev_ww, 𝜓)
    a = DNC.allocationweighting(u)
    ww = monitored_writeweight(io, cw, a, gw, ga)
    newmem = DNC.eraseandadd(M, ww, e, v)
    newmem
end

function monitoredprediction(io::IO, m, x)
    c = m.cell
    L, ww, wr, u = c.state.L, c.state.ww, c.state.wr, c.state.u
    numreads = c.R
    h = m.state
    out = c.controller([x;h])
    v = out[1:c.Y]
    ξ = out[c.Y+1:length(out)]
    rhs, wh = DNC.split_ξ(ξ, numreads, c.W)
    freegate = [rh.f for rh in rhs]
    c.M = monitored_writemem(io, c.M, wh, freegate, ww, wr, u)
    DNC.update_state_after_write!(c.state, c.M, wh, freegate)
    r = [monitored_readmem(io, c.M, rh, L, wr[1]) for rh in rhs]
    r = vcat(r...)
    DNC.update_state_after_read!(c.state, c.M, rhs)
    c.readvectors = r # Flatten list of lists
    return DNC.calcoutput(v, r, c.Wr)

end

function monitored_loss(model, x, y)
    open("monitor.txt", "a") do io
        ŷ = monitoredprediction(io, model, x)
        mask = x[end]
        loss = mask * Flux.logitcrossentropy(ŷ, y)
        loss
    end
end

        




