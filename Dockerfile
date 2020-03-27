FROM julia:latest

#Set container working directory
WORKDIR ~/.julia/dev/DNC
#Copy all files to the current container location
COPY . .
# Run training script on build
RUN julia experiments/trainrepeat.jl

