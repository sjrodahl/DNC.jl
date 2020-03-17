using Flux
using Flux: throttle
using BSON: @save



function singlesequence(nbits, seqlength)
    tocopy = BitArray(Int.(round.(rand(nbits, seqlength))))
    empty_examples = BitArray(zeros(nbits, seqlength))
    mask = [falses(seqlength); trues(seqlength)]
    xseqs =cat(tocopy, empty_examples; dims=2)
    yseqs = cat(BitArray(zeros(nbits, seqlength)), tocopy; dims=2)
    xseqs, yseqs, mask
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
    [(sequence[1][:,i], sequence[2][:,i], sequence[3][i]) for i in 1:size(sequence[1])[2]]

function generatedata(nbits, seqlength, ntimes)
    data = generatesequence(nbits, seqlength, ntimes)
    split_seqs = [sequence_to_examples(seq) for seq in data]
    return vcat(split_seqs...)
end

function loss(x, y, mask; printres=false)
    elem = cat(x, mask; dims=1)
    res = model(elem)
    printres && println(res)
    printres && println(y)
    l =  Flux.logitcrossentropy(res, y)
    printres && println(l)
    l
end

function showseq(testseq)
    for ex in testseq
        loss(ex...; printres=true)
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
            mask = d[3]
            if Bool(mask)
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

