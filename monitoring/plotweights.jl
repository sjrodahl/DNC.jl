lines = readlines("../monitor.txt")
lines = split.(lines, ":")

readweights = [eval(Meta.parse(l[2])) for l in filter(x->x[1]=="readweight", lines)]
writeweights = [eval(Meta.parse(l[2])) for l in filter(x->x[1]=="writeweight", lines)]

readweights = hcat(readweights...)
writeweights = hcat(writeweights...)

