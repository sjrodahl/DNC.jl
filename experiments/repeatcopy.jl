using Flux
using Flux.Data: DataLoader
# Represent a single sequence with observation, target and mask
struct RepeatCopy
    obs
    target
    mask
end


function pad(arr::AbstractArray{T, 2}, dims::Integer...; left=false, padval=zero(T)) where T
    arrx, arry = size(arr)
    padlenx = dims[1] - arrx
    padleny = dims[2] - arry
    (padlenx < 0 || padleny < 0) && return arr
    hpadmat = fill(padval, (arrx, padleny))
    vpadmat = fill(padval, (padlenx, arry+padleny))
    if left
        return vcat(hcat(hpadmat, arr), vpadmat)
    else
        return vcat(hcat(arr, hpadmat), vpadmat)
    end
end

function printrow(row)
    row = map(row) do x
        x==0 && return "-"
        return "$x"
    end
    "+ $(join(row, " ")) +"
end


function prettyprint(data)
    r, c = size(data)
    for row in 1:r
        println(printrow(data[row, :]))
    end
    print("\n")
end

import Base.show
function Base.show(io::IO, rp::RepeatCopy)
    print(io, "RepeatCopy(obssize: $(size(rp.obs)), targetsize $(size(rp.target))")
end

function prettyprint(rp::RepeatCopy)
    print("Observation:\n")
    prettyprint(rp.obs)
    print("\nTarget:\n")
    prettyprint(rp.target)
end

function RepeatCopy(;
        nbits=4,
        minlength=1,
        maxlength=2,
        minrepeats=1,
        maxrepeats=2)
    # Reserve one channel for start marker and one for num-repeats marker
    fullobssize = nbits+2
    # Reserve one target channel for the end-marker
    fulltargsize = nbits+1
    timingmarker_idx = nbits+1
    numrepeatsmarker_idx = fullobssize

    # Sample variables for this sequence
    seqlength = rand(minlength:maxlength)
    numrepeats = rand(minrepeats:maxrepeats)
    # Get total size of sequence with padding
    totallength = (maxrepeats+1)*maxlength + 3

    obspattern = rand([0,1], nbits, seqlength)
    targpattern = repeat(obspattern, 1, numrepeats)

    # Attach start- and numrepeats markers to observation pattern
    startflag = Flux.onehot(timingmarker_idx, 1:fullobssize)
    repeatsflag = zeros(eltype(obspattern),  fullobssize)
    repeatsflag[numrepeatsmarker_idx] = numrepeats
    obspattern = pad(obspattern, fullobssize, seqlength)
    obspattern = hcat(startflag, obspattern, repeatsflag)
    # Pad observation with space for the target
    obs = pad(obspattern, fullobssize, totallength)

    # Same for target pattern, but with only one marker
    endflag = Flux.onehot(timingmarker_idx, 1:fulltargsize)
    targpattern = pad(targpattern, fulltargsize, size(targpattern)[2])
    targpattern = hcat(targpattern, endflag)
    targ = pad(targpattern, fulltargsize, size(obspattern)[2]+size(targpattern)[2]; left=true)
    targ = pad(targ, fulltargsize, totallength)

    # Create mask
    mask = [zeros(Int32, size(obspattern)[2]); ones(Int32, size(targpattern)[2])]
    mask = [mask; zeros(Int32, totallength-length(mask))]
    RepeatCopy(obs, targ, mask)
end

"""
    sigmoidlogitscrossentropy(logits, targets)

Formula taken from: https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits
"""
function sigmoidlogitscrossentropy(logits, targets)
    x, z = logits, targets
    return max.(x, 0) .- x .* z + log.(ones(size(x)) + exp.(-abs.(x)))
end

function maskedsigmoidcrossentropy(logits, target, mask)
    l = sum(Flux.logitbinarycrossentropy.(logits, target); dims=1) * mask
    l[1]
end

function printmodeloutput(logits, mask)
    r, c  = size(logits)
    convertedmatrix = repeat(mask', r) .* round.(Flux.σ.(logits))
    print("Model output:\n")
    prettyprint(convertedmatrix)
end


function runmodel(model, rp::RepeatCopy)
    r, c = size(rp.obs)
    inputs = [rp.obs[:,col] for col in 1:c]
    logits = model.(inputs)
    logits = reshape(vcat(logits...), size(rp.target))
end


function loss(model, rp::RepeatCopy; printoutput=false)
    logits = runmodel(model, rp)
    l = maskedsigmoidcrossentropy(logits, rp.target, rp.mask)
    if printoutput
        prettyprint(rp)
        printmodeloutput(logits, rp.mask)
        println("Loss = $l")
    end
    l
end

function loss(model, batcheddata::Array{RepeatCopy, 1}; printoutput=false)
    batchloss = 0
    for rp in batcheddata
        l = loss(model, rp; printoutput=printoutput)
        printoutput=false # only print from first batch
        batchloss += l
    end
    batchloss
end


