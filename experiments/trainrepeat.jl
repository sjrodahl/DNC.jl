cd(@__DIR__)
using Pkg
Pkg.activate("../.")
Pkg.instantiate()

using DNC
using Flux
using Zygote

using Flux.Data: DataLoader
include("repeatcopy.jl")

nbits = 4
maxrepeats = 1
minlength = 2
maxlength = 2

X = nbits+2
Y = nbits+1
N, W, R = 16, 16, 2

niter = 10
batchsize = 16
seqs = [RepeatCopy(;
            nbits=nbits,
            maxrepeats=maxrepeats,
            minlength=minlength,
            maxlength=maxlength)
        for i in 1:(niter*batchsize)]

batcheddata = DataLoader(seqs, batchsize=batchsize)

model = Dnc(X, Y, N, W, R)

loss(rc; printoutput=false) = loss(model, rc; printoutput=printoutput)

using BSON: @save
using Flux: @progress, throttle
using Flux.Optimise: update!, runall, StopException
using Zygote: Params, gradient
using Dates

savepath = "../data/$(today())-dnc-$niter-$batchsize.bson"

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
    @info "Saving model to $savepath"
    @save savepath model
end

opt = RMSProp(1e-3)
evalcb = throttle(10) do
    idx = rand(1:length(seqs))
    loss(seqs[idx]; printoutput=true)
end

mytrain!(loss, params(model), batcheddata, opt; cb=evalcb)
