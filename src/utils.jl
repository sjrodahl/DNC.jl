import Flux: softmax
using LinearAlgebra


cosinesim(u, v) = dot(u, v)/(norm(u)*norm(v))

weighted_softmax(xs, weight) = softmax(xs.*weight)

oneplus(x) = 1 + log(1+exp(x))
