import Flux: softmax


cosinesim(u, v) = dot(u, v)/(norm(u)*norm(v))

weighted_softmax(xs, weight) = softmax(xs.*weight)

"""
    contentaddress(key, M, β[, K])

Compute the similarity K (default cosine similarity) between all rows of memory M and the key.
β acts as sharpener: high values concentrate weights, low values (<1) blurs them.
"""
function contentaddress(key, M, β, K=cosinesim)
    xs = [K(key, row) for row in eachrow(M)]
    weighted_softmax(xs, β)
end
