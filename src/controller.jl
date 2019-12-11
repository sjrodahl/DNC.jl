using Flux: softmax

@with_kw mutable struct State{M<:AbstractArray, A<:AbstractArray, A2<:AbstractArray}
    L::M
    p::A
    u::A
    w_w::A
    w_r::A2
end

State(N::Int, R::Int) = State(
    L=zeros(N, N),
    p = zeros(N),
    u = zeros(N),
    w_w = zeros(N),
    w_r = [zeros(N) for i in 1:R]
    )

outputsize(R::Int, N::Int, W::Int, X::Int, Y::Int) = W*R + 3W + 5R +3 + Y

function constructdata(size, numsamples)
    function genitem(size)
        i = rand(size)
        (i, i)
    end
    return [genitem(size) for i in 1:numsamples]
end

loss(x, y) = Flux.mse(predict(x), y)

function calcoutput(v, r)
    return v + W_r*(vcat(r))
end


# TODO: So far assuming 1 read head
function predict(x, controller, state, M, R, W, Y)
    out = controller(x)
    v = out[1:Y]
    ξ = out[Y+1:length(out)]
    interface = split_ξ(ξ, R, W)
    writemem(M, interface, state)
    r = readmem(M, interface, state)
    return calcoutput(v, r)
end

function split_ξ(ξ, R::Int, W::Int)
    length(ξ) != (W*R)+3W+5R+3 &&
        error("Length of xi-vector is incorrect. Expected $((W*R)+3W+5R+3), got $(length(ξ))")
    # read keys
    k_r = [ξ[((r-1)*W+1):r*W] for r in 1:R]
    β_r = ξ[(R*W+1):(R*W+R)]
    k_w = ξ[(R*W+1+R):(R*W+R+W)]
    β_w = ξ[(R*W+R+W+1)]
    ê = ξ[(R*W+R+W+2):(R*W+R+2W+1)]
    v = ξ[(R*W+R+2W+2):(R*W+R+3W+1)]
    f̂ = ξ[(R*W+R+3W+2):(R*W+2R+3W+1)]
    ĝ_a = ξ[(R*W+2R+3W+2)]
    ĝ_w = ξ[(R*W+2R+3W+3)]
    rest = ξ[(R*W+2R+3W+4):length(ξ)]
    readmode = [rest[((r-1)*3+1):3r] for r in 1:R]
    return (
        k_r = k_r,
        β_r = β_r,
        k_w = k_w,
        β_w = β_w,
        erase = Flux.σ.(ê),
        add = v,
        free = Flux.σ.(f̂),
        alloc_gate = Flux.σ(ĝ_a),
        write_gate = Flux.σ(ĝ_w),
        readmode = Flux.softmax.(readmode)
    )
end
