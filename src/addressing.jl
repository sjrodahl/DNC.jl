using Base: cumprod
using Flux: param

"""
    contentaddress(key, M, β[, K])

Compute the similarity K (default cosine similarity) between all rows of memory M and the key.
β acts as sharpener: high values concentrate weights, low values (<1) blurs them.
"""
function contentaddress(key, M, β, K=cosinesim)
    xs = [K(key, row) for row in eachrow(M)]
    weighted_softmax(xs, β)
end

# Single read head
function memoryretention(read_weights::AbstractArray{<:Number, 1}, free_gate)
    return ones(length(read_weights)) .- free_gate.*read_weights
end

#  Multiple read heads
function memoryretention(read_weights::Array{<:AbstractArray, 1}, free_gate)
    R = length(read_weights)
    rs = [ones(length(read_weights[i])) .- free_gate[i].*read_weights[i] for i in 1:R]
    foldl(rs) do x, y
        x.*y
    end
end

usage(u_prev, write_weights, 𝜓) = (u_prev + write_weights - (u_prev.*write_weights)) .* 𝜓

const _EPSILON = 1e-6


cumprod_exclusive(arr::AbstractArray) = cumprod(arr) ./ arr

function allocationweighting(u::AbstractArray; eps::AbstractFloat=_EPSILON)
    u = eps .+ (1 - eps) .* u # Ensure values are large enough for numerical stability in cumprod_exclusive
    N = length(u)
    ϕ = sortperm(u)
    sortedusage = u[ϕ]
    prod_sortedusage = cumprod_exclusive(sortedusage)
    sortedalloc = (1 .- sortedusage) .* prod_sortedusage
    a = sortedalloc[ϕ]
    a
 end

function allocationweighting(free_gate, prev_w_r, prev_w_w, prev_usage; eps::AbstractFloat=_EPSILON)
    𝜓 = memoryretention(prev_w_r, free_gate)
    u = usage(prev_usage, prev_w_w, 𝜓)
    allocationweighting(u)
end

function allocationweighting(free_gate, state::State; eps::AbstractFloat=_EPSILON)
    @unpack w_r, w_w, u = state
    allocationweighting(free_gate, w_r, w_w, u)
end

function writeweight(c_w, a, g_w, g_a)
    return g_w*(g_a.*(a) + (1-g_a)c_w)
end

precedenceweight(p_prev, w_w) = (1-sum(w_w))*p_prev + w_w

function updatelinkmatrix!(L, precedence, w_w)
    N, _ = size(L)
    for i in 1:N
        for j in 1:N
            if i != j
                L[i, j] = (1 - w_w[i] - w_w[j]) * L[i, j] + w_w[i]*precedence[j]
            end
        end
    end
    L
end

forwardweight(L, w_r) = L*w_r
backwardweight(L, w_r) = L'*w_r

"""
    readweight(backw, content, forw, read_mode)

Interpolate the backward weighting, content weighting and forward weighting.
read_mode is a vector of size 3 summing to 1.
"""
function readweight(backw, content, forw, read_mode)
    return read_mode[1]*backw + read_mode[2]*content + read_mode[3]*forw
end
