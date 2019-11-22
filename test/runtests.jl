using DNC
using Test

@testset "DNC.jl" begin
    # Write your own tests here.
    @testset "addressing.jl" begin include("addressing_tests.jl") end
end
