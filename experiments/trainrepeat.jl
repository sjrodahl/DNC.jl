cd(@__DIR__)
#using Pkg
#Pkg.activate("../.")
#Pkg.instantiate()

using DNC
using Flux
using Zygote
using CuArrays

using Flux.Data: DataLoader
include("repeatcopy.jl")

nbits = 4
minrepeats = 1
maxrepeats = 2
minlength = 1
maxlength = 2

X = nbits+2
Y = nbits+1
N, W, R = 16, 16, 4
controllerout = 64

niter = 100000
batchsize = 16

seqs = [RepeatCopy(;
            nbits=nbits,
            minrepeats=minrepeats,
            maxrepeats=maxrepeats,
            minlength=minlength,
            maxlength=maxlength)
        for i in 1:(niter*batchsize)] |> gpu

batcheddata = RepeatCopyBatchLoader(seqs, batchsize=batchsize)

model = Dnc(X, Y, controllerout, N, W, R, batchsize) |> gpu

loss(rc::RepeatCopy; printoutput=false) = loss(model, rc; printoutput=printoutput)
loss(batch::Tuple; printoutput=false) = loss(model, batch...; printoutput=printoutput)

opt = RMSProp(1e-3)

evalcb = ThrottleIterations(100) do
    idx = rand(1:(length(seqs)-batchsize))
    loss(Base.iterate(batcheddata, idx)[1]; printoutput=true)
end

@time mytrain!(loss, Flux.params(model), batcheddata, opt; cb=evalcb)
