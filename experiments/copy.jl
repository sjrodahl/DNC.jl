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
nseqs = 50000

data = generatedata(nbits, seqlength, nseqs)
testseq = sequence_to_examples(singlesequence(nbits, seqlength))

evalcb = throttle(10) do
    println("Training model...")
    showseq(testseq)
    @save "model-$(nseqs)-checkpoint.bson" model
end
mytrain!(loss, params(model), data, opt, cb=evalcb)
