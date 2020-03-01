using DNC
using Flux



function generatedata(nbits, seqlength, ntimes)
    function generatesequence(nbits, seqlength)
        tocopy = BitArray(Int.(round.(rand(nbits, seqlength))))
        empty_examples = BitArray(zeros(nbits, seqlength))
        flags = [falses(seqlength); trues(seqlength)]
        xseqs = cat(cat(tocopy, empty_examples; dims=2), flags';dims=1)
        yseqs = cat(BitArray(zeros(nbits, seqlength)), tocopy;dims=2)
        xseqs, yseqs
    end
    return [generatesequence(nbits, seqlength) for i in 1:ntimes]
end

function loss(x, y)
    !Bool(x[end]) && return 0
    Flux.mse(model(x), y)
end
