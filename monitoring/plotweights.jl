
cd(@__DIR__)
lines = readlines("../monitor.txt")
lines = split.(lines, ":")

filterlines(name) = [eval(Meta.parse(l[2])) for l in filter(x->x[1]==name, lines)]

rws, wws = filterlines("readweight"), filterlines("writeweight")

rws = hcat(rws...)
wws = hcat(wws...)

