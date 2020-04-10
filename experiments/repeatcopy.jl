using Dates
using Flux
using Flux.Data: DataLoader


# Represent a single sequence with observation, target and mask
struct RepeatCopy
    obs
    target
    mask
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

    obspattern = rand([0.f0,1.f0], nbits, seqlength) #Use Float32
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
    mask = [zeros(Float32, size(obspattern)[2]); ones(Float32, size(targpattern)[2])]
    mask = [mask; zeros(Float32, totallength-length(mask))]
    RepeatCopy(obs, targ, mask)
end

import Base.show
function Base.show(io::IO, rp::RepeatCopy)
    print(io, "RepeatCopy(obssize: $(size(rp.obs)), targetsize $(size(rp.target))")
end

struct RepeatCopyBatchLoader
    dataloader::DataLoader
end


"""
    RepeatCopyBatchLoader(data...; batchsize=1, shuffle=false, partial=true)

An object that iterates over batches of RepeatCopy sequences.
The output of iteration are three list of matrices: observations, targets, and mask.
The lists represents the time steps, so the ``j``th column of observation matrix ``i`` is the ``i``th column of sequence ``j``.

See also: [`Flux.Data.DataLoader`](@ref), [`RepeatCopy`](@ref)

"""
RepeatCopyBatchLoader(a...; ka...) = RepeatCopyBatchLoader(DataLoader(a...; ka...))


function Base.iterate(d::RepeatCopyBatchLoader, i=0)
    i >= d.dataloader.imax && return nothing
    batch, nexti = Base.iterate(d.dataloader, i)
    batchsize = d.dataloader.batchsize
    obsbits, seqlength = size(batch[1].obs)
    targetbits = size(batch[1].target, 1)
    batchobsmatrix = [similar(batch[1].obs, eltype(batch[1].obs), (obsbits, batchsize))
                      for _ in 1:seqlength]
    @views for i in 1:seqlength
        for rc in 1:batchsize
            batchobsmatrix[i][:,rc] = batch[rc].obs[:, i]
        end
    end
    batchtargetmatrix = [similar(batch[1].target, eltype(batch[1].target), (targetbits, batchsize))
                      for _ in 1:seqlength]
    @views for i in 1:seqlength
        for rc in 1:batchsize
            batchtargetmatrix[i][:,rc] = batch[rc].target[:, i]
        end
    end
    batchmaskmatrix = [similar(batch[1].mask, eltype(batch[1].mask), (batchsize))
                      for _ in 1:seqlength]
    @views for i in 1:seqlength
        for rc in 1:batchsize
            batchmaskmatrix[i][rc] = batch[rc].mask[i]
        end
    end
    return ((batchobsmatrix, batchtargetmatrix, batchmaskmatrix), nexti)
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


function prettyprint(rp::RepeatCopy)
    print("Observation:\n")
    prettyprint(rp.obs)
    print("\nTarget:\n")
    prettyprint(rp.target)
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
    l = sum(Flux.logitbinarycrossentropy.(logits, target); dims=1) .* mask
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

"""
This loss function is intented to handle the output of a RepeatCopyBatchLoader iteration.
"""
function loss(model, obs::AbstractArray{T, 1},
              target::AbstractArray{T, 1},
              mask::AbstractArray
              ; printoutput = false) where {T<:AbstractArray{E, 2}} where E
    length(obs) == length(target) == length(mask) || throw(ArgumentError("Observations, targets and mask needs to be of same length"))
    logits = model.(obs)
    l = maskedsigmoidcrossentropy.(logits, target, mask)
    if printoutput
        first_rc = reconstruct_rc(obs, target, mask, 1)
        prettyprint(first_rc)
        first_logits = getcolumn(logits, 1)
        printmodeloutput(first_logits, first_rc.mask)
        println("Average loss = ", sum(l)/size(obs[1], 2))
        println("Total batch loss = ", sum(l))
    end
    sum(l)
end

getcolumn(arr::AbstractArray{T, 1}, colno::Integer) where {T<:AbstractArray} = 
    reshape(reduce(vcat, [arr[i][:, colno] for i in 1:length(arr)]), (size(arr[1], colno), length(arr)))


function reconstruct_rc(obs, target, mask, rcno::Integer)
    rcobs = getcolumn(obs, rcno)
    rctarget = getcolumn(target, rcno)
    rcmask = [mask[i][rcno] for i in 1:length(mask)]
    return RepeatCopy(rcobs, rctarget, rcmask)
end

using BSON: @save
using Flux: @progress, throttle
using Flux.Optimise: update!, runall, StopException
using Zygote: Params, gradient
using Dates

function mytrain!(loss, ps, data, opt; cb=()->())
    ps = Params(ps)
    cb = runall(cb)
    for d in data
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
    @save "$(today())-repeatcopy-$niter-$batchsize.bson" model
end

# Allow evalcb to report based on iteration number
mutable struct ThrottleIterations
    fn
    interval
    cnt
end

ThrottleIterations(fn, interval) = ThrottleIterations(fn, interval, 0)

function (fncint::ThrottleIterations)(a...;ka...)
    fncint.cnt += 1
    if fncint.cnt % fncint.interval == 0
        println(now(), ", iteration number ", fncint.cnt)
        return fncint.fn(a...,ka...)
    else
        return nothing
    end
end
