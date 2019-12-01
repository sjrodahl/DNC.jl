module DNC

using LinearAlgebra

export contentaddress
export readmem
export writemem!
export State

include("addressing.jl")
include("access.jl")
include("utils.jl")

end # module
