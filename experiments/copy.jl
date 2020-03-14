using Flux: @epochs
using DNC

include("copysetup.jl")
X = 4
Y = X-1
N = 5
W = X
R = 1

model = Dnc(X, Y, N, W, R)

opt = ADAM(0.01)

nbits = X-1
seqlength = 3
#nseqs = 10000
nseqs = 10

data = generatedata(nbits, seqlength, nseqs)
xs = [d[1] for d in data]
ys = [d[2] for d in data]
mask = [d[3] for d in data]
batcheddata = Flux.Data.DataLoader(xs, ys, mask, batchsize=2)
testseq = sequence_to_examples(singlesequence(nbits, seqlength))

evalcb = throttle(10) do
    println("Training model...")
    showseq(testseq)
    @save "model-$(nseqs)-checkpoint.bson" model
end
#train!(loss, params(model), batcheddata, opt, cb=evalcb)
