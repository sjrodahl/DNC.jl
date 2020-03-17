using DNC
using Flux
using Zygote

include("repeatcopy.jl")

nbits = 4
maxrepeats = 1
minlength = 2
maxlength = 2

X = nbits+2
Y = nbits+1
N, W, R = 16, 16, 2

nseqs = 10000
seqs = [RepeatCopy(;
            nbits=nbits,
            maxrepeats=maxrepeats,
            minlength=minlength,
            maxlength=maxlength)
        for i in 1:nseqs]

model = Dnc(X, Y, N, W, R)

loss(rc::RepeatCopy; printoutput=false) = loss(model, rc; printoutput=printoutput)

using Flux: @progress, throttle
using Flux.Optimise: update!, runall, StopException
using Zygote: Params, gradient
function mytrain!(loss, ps, data, opt; cb=()->())
    ps = Params(ps)
    cb = runall(cb)
    @progress for d in data
        try
            gs = gradient(ps) do
                loss(d)
            end
            update!(opt, ps, gs)
            cb()
        catch ex
            if ex isa StopException
                break
            else
                rethrow(ex)
            end
        end
    end
end

opt = RMSProp(1e-3)
evalcb = throttle(10) do
    idx = rand(1:nseqs)
    loss(seqs[idx]; printoutput=true)
end

mytrain!(loss, params(model), seqs, opt; cb=evalcb)
