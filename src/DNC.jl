module DNC

using Flux
using LinearAlgebra
using Zygote
using TensorCast

export Dnc


include("utils.jl")
include("access.jl")
include("addressing.jl")
include("computer.jl")

end # module
