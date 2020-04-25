using DNC
using Flux
using Zygote

using Flux.Data: DataLoader
include("repeatcopy.jl")

nbits = 4
minrepeats = 1
maxrepeats = 1
minlength = 1
maxlength = 1

X = nbits+2
Y = nbits+1
N, W, R = 16, 16, 2
controllerout = 64
clipvalue = 20

niter = 700
batchsize = 16
seqs = [RepeatCopy(;
            nbits=nbits,
            minrepeats=minrepeats,
            maxrepeats=maxrepeats,
            minlength=minlength,
            maxlength=maxlength)
        for i in 1:(niter*batchsize)]

batcheddata = RepeatCopyBatchLoader(seqs, batchsize=batchsize)

controller = Chain(Dense(DNC.inputsize(X, R, W), 64), Dense(64, 64), Dense(64, controllerout))

model = Dnc(controller, X, Y, controllerout, N, W, R, batchsize; clipvalue=clipvalue)

loss(rc::RepeatCopy; printoutput=false) = loss(model, rc; printoutput=printoutput)
loss(batch::Tuple; printoutput=false) = loss(model, batch...; printoutput=printoutput)

opt = RMSProp(1e-3)

evalcb = ThrottleIterations(100) do
    idx = rand(1:(length(seqs)-batchsize))
    loss(Base.iterate(batcheddata, idx)[1]; printoutput=true)
end

@time mytrain!(loss, Flux.params(model), batcheddata, opt; cb=evalcb)
