using DNC
using Flux
using DNC: MemoryAccess, split_ξ, update_state_after_write!, update_state_after_read!, eraseandadd, calcoutput
using DNC: contentaddress, backwardweight, forwardweight, readweight, readweights, writeweight, writeweights
using DNC: usage, memoryretention, allocationweighting

import DNC.readweight, DNC.readweights, DNC.writeweight, DNC.writeweights


function readweight(io::IO, backw, content, forw, readmode)
    rw = DNC.readweight(backw, content, forw, readmode)
    print(io, "readweight:",rw,"\n")
    rw
end

function readweights(io::IO, M, inputs, L, prev_wr)
    k, β, readmode= inputs.kr, inputs.βr, inputs.readmode
    cr = contentaddress(k, M, β)
    b = backwardweight(L, prev_wr)
    f = forwardweight(L, prev_wr)
    wr = readweight(io, b, cr, f, readmode)
end


function writeweight(io::IO, cw, a, gw, ga)
    ww = DNC.writeweight(cw, a, gw, ga)
    print(io, "writeweight:", ww,"\n")
    ww
end


function writeweights(io::IO, M, inputs,
        prev_ww,
        prev_wr,
        prev_usage)
    k, β, ga, gw, e, v, free = inputs.kw, inputs.βw, inputs.ga[1], inputs.gw[1], inputs.e, inputs.v, inputs.f
    cw = contentaddress(k, M, β)
    𝜓 = memoryretention(prev_wr, free)
    u = usage(prev_usage, prev_ww, 𝜓)
    a = allocationweighting(u)
    ww = writeweight(io, cw, a, gw, ga)
end


function (ma::MemoryAccess)(io::IO, inputs)
    R = size(ma.state.wr)[2]
    W = size(ma.M)[2]
    inputs = split_ξ(inputs, ma.inputmaps)
    p, u, ww, wr = ma.state.p, ma.state.u, ma.state.ww, ma.state.wr
    u = usage(u, ww, wr, inputs.f)
    ww= writeweights(io, ma.M, inputs, ww, wr, u)
    ma.M = eraseandadd(ma.M, ww, inputs.e, inputs.v)
    update_state_after_write!(ma.state, ww, u)
    wr = readweights(io, ma.M, inputs, ma.state.L, wr)
    update_state_after_read!(ma.state, wr)
    readvectors = ma.M' * wr
    readvectors
end

function monitoredprediction(io::IO, model, x)
    m = model.cell
    h = m.readvectors
    out = m.controller([x;h])
    v = out[1:m.Y]
    ξ = out[m.Y+1:end]
    r = m.memoryaccess(io, ξ)
    r = reshape(r, size(r)[1]*size(r)[2])
    return calcoutput(v, r, m.Wr)
end


function monitored_loss(model, x, y, mask)
    open("monitor.txt", "a") do io
        ŷ = monitoredprediction(io, model, x)
        loss = sum(mask * Flux.logitbinarycrossentropy.(ŷ, y))
        loss
    end
end

        




