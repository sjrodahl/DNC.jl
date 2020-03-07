using DNC
using Flux
using Flux: throttle
using BSON: @save



function singlesequence(nbits, seqlength)
    tocopy = BitArray(Int.(round.(rand(nbits, seqlength))))
    empty_examples = BitArray(zeros(nbits, seqlength))
    flags = [falses(seqlength); trues(seqlength)]
    xseqs = cat(cat(tocopy, empty_examples; dims=2), flags';dims=1)
    yseqs = cat(BitArray(zeros(nbits, seqlength)), tocopy;dims=2)
    xseqs, yseqs
end

# Data should (eventually be on the form [(x1, y1), (x2, y2)...])
# However, I need to keep the sequences together to eg. extract validation set.
# So I will output the data as [(seqx1, seqy1),...]
function generatesequence(nbits, seqlength, ntimes)
    data = [singlesequence(nbits, seqlength) for i in 1:ntimes]
    data
end

# Assuming sequence is a tuple of (seqx, seqy)
#TODO: fix
sequence_to_examples(sequence::Tuple) =
    [(sequence[1][:,i], sequence[2][:,i]) for i in 1:size(sequence[1])[2]]

function generatedata(nbits, seqlength, ntimes)
    data = generatesequence(nbits, seqlength, ntimes)
    split_seqs = [sequence_to_examples(seq) for seq in data]
    return vcat(split_seqs...)
end

function loss(x, y; printres=false)
    res = model(x)
    printres && println(res)
    printres && println(y)
    if Bool(x[end])
        loss = Flux.logitcrossentropy(res, y)
    else
        # Ignore observation examples with flag=0
        loss = 0.0
    end
    printres && println(loss)
    loss
end

function showseq(testseq)
    for ex in testseq
        loss(ex[1], ex[2]; printres=true)
    end
end

using Flux: @progress
using Flux.Optimise: update!, runall, StopException
using Zygote: Params, gradient

function mytrain!(loss, ps, data, opt; cb=()->())
    ps = Params(ps)
    cb = runall(cb)
    @progress for d in data
        try
            istestexample = d[1][end]
            if istestexample
                gs = gradient(ps) do
                    loss(d...)
                end
                update!(opt, ps, gs)
            else
                loss(d...)
            end
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


X = 4
Y = X-1
N = 5
W = X
R = 1

model = Dnc(X, Y, N, W, R)

opt = ADAM(0.01)

nbits = X-1
seqlength = 3
nseqs = 500000

data = generatedata(nbits, seqlength, nseqs)
testseq = sequence_to_examples(singlesequence(nbits, seqlength))

evalcb = throttle(10) do
    println("Training model...")
    showseq(testseq)
    @save "model-checkpoint.bson" model
end
mytrain!(loss, params(model), data, opt, cb=evalcb)
