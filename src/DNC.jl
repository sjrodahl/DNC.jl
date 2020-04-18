module DNC

using Flux
using LinearAlgebra
using Zygote
using TensorCast

export readweights
export writeweights
export State
export Dnc
export DNCCell


include("utils.jl")
include("access.jl")
include("addressing.jl")
include("computer.jl")

end # module
