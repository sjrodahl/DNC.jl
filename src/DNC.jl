module DNC

using Flux
using LinearAlgebra
using Zygote

export readweights
export writeweights
export State
export Dnc
export DNCCell


include("access.jl")
include("computer.jl")
include("addressing.jl")
include("utils.jl")

end # module
