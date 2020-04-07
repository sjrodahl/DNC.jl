using Base: cumprod
using Flux: param


"""
    contentaddress(key, M, β[, K])

Compute the similarity K (default cosine similarity) between all rows of memory M and the key.
β acts as sharpener: high values concentrate weights, low values (<1) blurs them.
"""
function contentaddress(key, M, β, K=cosinesim)
    wordsize, numreadheads = size(key)
    numwords, _ = size(M)
    all = [_contentaddress(key[:,i], M, β[i]) for i in 1:numreadheads]
    return reshape(vcat(all...),numwords, numreadheads)
end

function _contentaddress(key, M, β, K=cosinesim)
    r, c = size(M)
    xs = [K(key, M[row,:]) for row in 1:r]
    weightedsoftmax(xs, β)
end

"""
    memoryretention(readweights::AbstractArray{<:Number, 1, freegate} # Single read head
    memoryretention(readweights::Array{<:AbstractArray, 1}, freegate) # Multiple read heads
    
Determine how much each memory location will not be freed by the free gates.
"""
function memoryretention(readweights, freegate)
    rs = one(eltype(readweights)) .- readweights.*reshape(freegate, 1, size(freegate)...)
    rs = prod(rs; dims=2)
    dropdims(rs; dims=2)
end

_usage(u_prev, ww_prev, 𝜓) = (u_prev + ww_prev - (u_prev.*ww_prev)) .* 𝜓
"""
    usage(u_prev, ww_prev, wr_prev, freegate)

Calculate the usage vector of the memory rows.
A row is considered used (u[i]=1) if they have recently been written to and haven't been retained by the free gates (𝜓[i] =1)
"""
function usage(u_prev, ww_prev, wr_prev, freegate)
    𝜓 = memoryretention(wr_prev, freegate)
    _usage(u_prev, ww_prev, 𝜓)
end


const _EPSILON = 1f-6

"""
    allocationweighting(usage::AbstractArray; eps::AbstractFloat=1e-6)

    Provide new locations for writing. If all locations are used, no writes can be made.

"""
function allocationweighting(u::AbstractArray; eps::AbstractFloat=_EPSILON)
    u = eps .+ (1 - eps) .* u # Ensure values are large enough for numerical stability in cumprodexclusive
    N = length(u)
    ϕ = sortperm(u[:,1])
    sortedusage = u[ϕ]
    prodsortedusage = cumprodexclusive(sortedusage)
    sortedalloc = (1 .- sortedusage) .* prodsortedusage
    a = sortedalloc[ϕ]
    a
 end

using Zygote: @adjoint
# The sorting of allocation weighting introduce discontinuities
# in the backward pass, so we set the pullback to 1
@adjoint allocationweighting(u::AbstractArray; eps=_EPSILON) =
    allocationweighting(u; eps=eps), Δ -> (Δ, Δ)

"""
    writeweight(contentweighting, allocationweighting, writegate, allocationgate)

Calculate the write weightings over the matrix rows
"""
function writeweight(cw, a, gw, ga)
    return gw*(ga.*(a) + (1-ga).*cw)
end

precedenceweight(p_prev, ww) = (1-sum(ww))*p_prev + ww

function updatelinkmatrix!(L, precedence, ww)
    N, _ = size(L)
    for i in 1:N
        for j in 1:N
            if i != j
                L[i, j] = (1 - ww[i] - ww[j]) * L[i, j] + ww[i]*precedence[j]
            end
        end
    end
    L
end

forwardweight(L, wr) = L*wr
backwardweight(L, wr) = L'*wr

"""
    readweight(backw, content, forw, readmode)

Interpolate the backward weighting, content weighting and forward weighting.
readmode is a vector of size 3 summing to 1.
"""
function readweight(backw, content, forw, readmode)
    return readmode[1]*backw + readmode[2]*content + readmode[3]*forw
end
