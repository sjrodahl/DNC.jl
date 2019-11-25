using DNC
using Test

@testset "DNC.jl" begin
    @testset "addressing.jl" begin include("addressing_tests.jl") end
    @testset "utils.jl" begin include("utils_tests.jl") end

end
