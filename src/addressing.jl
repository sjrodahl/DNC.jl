using Base: cumprod
using Flux: param
using Zygote: Buffer


function _pairwise!(r::Zygote.Buffer,
                    metric::Function,
                    col::AbstractArray{T, 3},
                    row::AbstractArray{T, 3}, β::AbstractArray{T, 2}) where {T, S}
    nrow = size(row, 1)
    ncol = size(col, 2)
    batchsize = size(col, 3)
    size(r) == (nrow, ncol, batchsize) || throw(DimensionMismatch("Incorrect size of r. Expected $((nrow, ncol, batchsize)), got $(size(r))"))
    @inbounds for k = 1:batchsize
        @inbounds for j = 1:ncol
            colj = view(col, :, j, k)
            @inbounds for i = 1:nrow
                rowi = view(row, i, :, k)
                r[i, j, k] = metric(rowi, colj, β[j, k])
            end
        end
    end
    r
end

"""
    contentaddress(key::AbstractArray{T, 3}, mem::AbstractArray{T, 3}, β::AbstractArray{S, 2}, K=weightedcosinesim) where {T, S}

Compute the similarity K (default cosine similarity) between all rows of memory M and the key.
β acts as sharpener: high values concentrate weights, low values (<1) blurs them.

# Arguments
- `key`: (N x R x B)
- `mem`: (N x W x B)
- `β`: (R x B)
"""

function contentaddress(key::AbstractArray{T, 3}, mem::AbstractArray{T, 3}, β::AbstractArray{S, 2}, K=weightedcosinesim) where {T, S}
    wordsize, numreadheads, batchsize = size(key)
    numloc, _, _ = size(mem)
    out = Zygote.Buffer(key, eltype(key), (numloc, numreadheads, batchsize))
    _pairwise!(out, K, key, mem, β)
    @views for b in 1:batchsize
        out[:, :, b] = softmax(out[:, :, b]; dims=1)
    end
    copy(out)
end

"""
    memoryretention(wr, f)

# Arguments
- `wr` - (N, R, B): previous iteration read weights.
- `f` - (R, B): free gates
Determine how much each memory location will not be freed by the free gates.
Returns a tensor of size (N x B)
"""
function memoryretention(wr, f)
    rs = one(eltype(wr)) .- wr.*reshape(f, 1, size(f)...)
    rs = prod(rs; dims=2)
    return dropdims(rs; dims=2)
end

_usage(u_prev, ww_prev, 𝜓) = (u_prev + ww_prev - (u_prev.*ww_prev)) .* 𝜓

"""
usage(u_prev, ww_prev, wr_prev, freegate)

# Arguments
- `u_prev` - (N, B) usage after previous iteration
- `ww_prev` - (N, 1, B) previous write weights
- `wr_prev` - (N, R, B) previous read weights
- `freegate` - (R, B)

Calculate the usage vector of the memory rows.
A row is considered used (``u[i]=1``) if they have recently been written to and haven't been retained by the free gates (``𝜓[i]=1``)
"""
function usage(u_prev, ww_prev, wr_prev, freegate)
    if ndims(ww_prev) == 3
        ww_prev = dropdims(ww_prev; dims=2)
    end
    𝜓 = memoryretention(wr_prev, freegate)
    _usage(u_prev, ww_prev, 𝜓)
end


const _EPSILON = 1f-6

"""
    allocationweighting(u::AbstractMatrix; eps::AbstractFloat=_EPSILON)
# Arguments
- `u`: (N x B) usage tensor

# Returns
- `a`: (N x B) tensor

"""
function  allocationweighting(u::AbstractMatrix; eps::AbstractFloat=_EPSILON)
    u = eps .+ (1-eps) .* u
    ϕ = [sortperm(u[:, i]) for i in 1:size(u, 2)]
    ϕ = reshape(vcat(ϕ...), size(u))
    for i in CartesianIndices(ϕ)
        ϕ[i] += size(ϕ, 1)*(i[2]-1)
    end
    sortedusage = u[ϕ]
    prodsortedusage = cumprodexclusive(sortedusage;dims=1)
    sortedalloc = (1 .- sortedusage) .* prodsortedusage
    a = sortedalloc[ϕ]
    a
end

using Zygote: @adjoint
# The sorting of allocation weighting introduce discontinuities
# in the backward pass, so we set the pullback to 1
@adjoint allocationweighting(u::AbstractMatrix; eps=_EPSILON) =
allocationweighting(u; eps=eps), Δ -> (Δ, )

"""
    writeweight(cw::AbstractArray{T, 3},
                 a::AbstractArray{T, 2},
                gw::AbstractArray{T, 1},
                ga::AbstractArray{T, 1}) where T
Batch version
#Arguments
- `cw`: (N x 1 x B) write content weighting
- `a`: (N x B) allocation weighting
- `gw`: (1 x B) write gate
- `ga`: (1 x B) allocation gate
a:
"""
function writeweight(cw::AbstractArray{T, 3},
                     a::AbstractArray{T, 2},
                     gw::AbstractArray{T, 2},
                     ga::AbstractArray{T, 2}) where T
    a = reshape(a, size(a,1), 1, size(a, 2))
    gw = reshape(gw, 1, size(gw)...)
    ga = reshape(ga, 1, size(ga)...)
    return gw.*(ga .* a + (one(T) .- ga).*cw)
end


"""
    precedenceweight(p_prev::AbstractArray{T, 2}, ww::AbstractArray{T, 3}) where T

p_prev: (N, B)
ww: (N, 1, B)

return p; (N, B)
"""
function precedenceweight(p_prev::AbstractArray{T, 2}, ww::AbstractArray{T, 3}) where T
    ww = dropdims(ww; dims=2)
    wwsum = sum(ww; dims=1) #(1 x B)
    return (one(T).-wwsum).*p_prev + ww
end


"""
    updatelinkmatrix(L::AbstractArray{T, 3}, p::AbstractArray{T, 2}, ww::AbstractArray{T, 3}) where T

# Arguments
- `L`: (N x N x B)
- `p`: (N x B)
- `ww`: (N x 1 x B)
"""
function updatelinkmatrix!(L::AbstractArray{T, 3}, p::AbstractArray{T, 2}, ww::AbstractArray{T, 3}) where T
    ww = dropdims(ww; dims=2)
    N, _, B = size(L)
    for b in 1:B
        for i in 1:N
            for j in 1:N
                if i != j
                    L[i, j, b] = (one(T) - ww[i, b] - ww[j, b]) * L[i, j, b] + ww[i, b]*p[j, b]
                end
            end
        end
    end
    L
end


"""
    forwardweight(L::AbstractArray{T, 3}, wr::AbstractArray{T, 3}) where T
Location weight for reading the next-written-to location.

# Arguments
- `L`: (N x N x B) link matrix
- `wr`: (N x R x B) read weights

See also: [`backwardweight`](@ref)
"""
function forwardweight(L::AbstractArray{T, 3}, wr::AbstractArray{T, 3}) where T
    B = size(L, 3)
    res = Zygote.Buffer(wr)
    @views for batch in 1:B
        res[:, :, batch] =  L[:, :, batch]*wr[:, :, batch]
    end
    copy(res)
end


"""
    backwardweight(L::AbstractArray{T, 3}, wr::AbstractArray{T, 3}) where T
Location weight for reading the previous-written-to location.

# Arguments
- `L`: (N x N x B) link matrix
- `wr`: (N x R x B) read weights

See also: [`forwardweight`](@ref)
"""
function backwardweight(L::AbstractArray{T, 3}, wr::AbstractArray{T, 3}) where T
    B = size(L, 3)
    res = Zygote.Buffer(wr)
    @views for batch in 1:B
        res[:, :, batch] = L[:, :, batch]'*wr[:, :, batch]
    end
    copy(res)
end


function _readweight(backw, content, forw, readmode)
    return readmode[1]*backw + readmode[2]*content + readmode[3]*forw
end

"""
    readweigth(backw::AbstractArray::{T, 2},
    cr::AbstractArray{T, 3},
    forw::AbstractArray{T, 2},
    readmode::AbstractArray{T, 2}) where T

Interpolate the backward weighting, content weighting and forward weighting.
readmode is a vector of size 3 summing to 1.

# Arguments
- `backw`, `cr`, `forw`: (N x R x B)
- `readmode`: (3 x R x B)
```
# Returns 
- (N x R x B) tensor represented each read heads readweights
"""
function readweight(backw::AbstractArray{T, 3}, cr::AbstractArray{T, 3}, forw::AbstractArray{T, 3}, readmode::AbstractArray{T, 3}) where T
    out = Zygote.Buffer(cr)
    R = size(cr, 2)
    B = size(cr, 3)
    @views for b in 1:B
        for r in 1:R
            out[:, r, b] = _readweight(backw[:, r, b], cr[:, r, b], forw[:, r, b], readmode[:, r, b])
        end
    end
    copy(out)
end
