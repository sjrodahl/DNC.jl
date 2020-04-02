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

niter = 200
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

opt = RMSProp(1e-3)

evalcb = ThrottleIterations(100) do
    idx = rand(1:length(seqs))
    loss(seqs[idx]; printoutput=true)
end

@time mytrain!(loss, params(model), batcheddata, opt; cb=evalcb)
