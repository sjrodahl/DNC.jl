using Documenter, DNC

makedocs(;
    modules=[DNC],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/sjrodahl/DNC.jl/blob/{commit}{path}#L{line}",
    sitename="DNC.jl",
    authors="Sondre Rodahl",
    assets=String[],
)

deploydocs(;
    repo="github.com/sjrodahl/DNC.jl",
)
