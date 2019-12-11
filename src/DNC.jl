module DNC

using LinearAlgebra

export contentaddress
export predict
export readmem
export writemem
export State

include("addressing.jl")
include("access.jl")
include("controller.jl")
include("utils.jl")

end # module
