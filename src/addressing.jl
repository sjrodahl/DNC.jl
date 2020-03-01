using Base: cumprod
using Flux: param

"""
    contentaddress(key, M, β[, K])

Compute the similarity K (default cosine similarity) between all rows of memory M and the key.
β acts as sharpener: high values concentrate weights, low values (<1) blurs them.
"""
function contentaddress(key, M, β, K=cosinesim)
    r, c = size(M)
    xs = [K(key, M[row,:]) for row in 1:r]
    weightedsoftmax(xs, β)
end

# Single read head
function memoryretention(readweights::AbstractArray{<:Number, 1}, freegate)
    return ones(length(readweights)) .- freegate.*readweights
end

#  Multiple read heads
function memoryretention(readweights::Array{<:AbstractArray, 1}, freegate)
    R = length(readweights)
    rs = [ones(length(readweights[i])) .- freegate[i].*readweights[i] for i in 1:R]
    foldl(rs) do x, y
        x.*y
    end
end

usage(u_prev, writeweights, 𝜓) = (u_prev + writeweights - (u_prev.*writeweights)) .* 𝜓

const _EPSILON = 1e-6


cumprodexclusive(arr::AbstractArray) = cumprod(arr) ./ arr

function allocationweighting(u::AbstractArray; eps::AbstractFloat=_EPSILON)
    u = eps .+ (1 - eps) .* u # Ensure values are large enough for numerical stability in cumprodexclusive
    N = length(u)
    ϕ = sortperm(u)
    sortedusage = u[ϕ]
    prodsortedusage = cumprodexclusive(sortedusage)
    sortedalloc = (1 .- sortedusage) .* prodsortedusage
    a = sortedalloc[ϕ]
    a
 end

function allocationweighting(freegate, prev_wr, prev_ww, prev_usage; eps::AbstractFloat=_EPSILON)
    𝜓 = memoryretention(prev_wr, freegate)
    u = usage(prev_usage, prev_ww, 𝜓)
    allocationweighting(u)
end

function allocationweighting(freegate, state::State; eps::AbstractFloat=_EPSILON)
    wr, ww, u = state.wr, state.ww, state.u
    allocationweighting(freegate, wr, ww, u)
end

using Zygote: @adjoint
# The sorting of allocation weighting introduce discontinuities
# in the backward pass, so we set the pullback to 1
@adjoint allocationweighting(u::AbstractArray; eps=_EPSILON) =
    allocationweighting(u; eps=eps), Δ -> (Δ, Δ)

function writeweight(cw, a, gw, ga)
    return gw*(ga.*(a) + (1-ga)cw)
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
