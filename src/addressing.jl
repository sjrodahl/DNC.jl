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

function allocationweighting(u)
    N = length(u)
    a = zeros(N)
    ϕ = sortperm(u) # Indices in ascending order of usage
    for j in 1:N
        a[ϕ[j]] = (1-u[ϕ[j]])*foldl(*, [u[ϕ[i]] for i in 1:(j-1)])
    end
    a
end
