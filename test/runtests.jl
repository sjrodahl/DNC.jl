using DNC
using Test

in_Sn(vec) = sum(vec) == 1.0
in_Δn(vec) = sum(vec) >= 0.0 && sum(vec) <= 1.0


@testset "DNC.jl" begin
    @testset "utils.jl" begin include("utils_tests.jl") end
    @testset "addressing.jl" begin include("addressing_tests.jl") end
    @testset "access.jl" begin include("access_tests.jl") end
end
