module DNC

using Flux
using LinearAlgebra
using Parameters

export contentaddress
export predict
export readmem
export writemem
export ReadHead
export State
export WriteHead
export Dnc
export DNCCell


include("access.jl")
include("computer.jl")
include("addressing.jl")
include("utils.jl")

end # module
