using MPI
using LinearAlgebra
using CSV
using Dates
using TimerOutputs
using FFTW
import FFTW: AbstractFFTs
using Random
using Statistics
using FLoops
using StableLinearAlgebra

const terminal = stdout
const terminal_error = stderr
const _mattype = Float64
const _mattypeC = ComplexF64

BLAS.set_num_threads(1)
FFTW.set_num_threads(1)

MPI.Init()

const original_directory = pwd()
cd(@__DIR__)

try
    include(joinpath(@__DIR__, "2initialize.jl"))
catch
    MPI.Finalize()
    rethrow()
finally
    redirect_stdout(terminal)
    redirect_stderr(terminal_error)
    cd(original_directory)
end

# Initialization already computes Fs using CalculateUDV!.
try
    if icpurank == rootcpurank
        G = zeros(_mattype, Nsite, Nsite)

        # Compute G = inv(I + L * Diagonal(d) * R) stably from Fs[1].
        StableLinearAlgebra.inv_IpA!(G, Fs[1], Ldr_ws)
        #Inv_IpA!(G, Fs[1], Ldr_ws)

        diagonal_G = diag(G)
        println("size(G) = ", size(G))
        println("Diagonal entries G[i,i]:")
        for i in eachindex(diagonal_G)
            println("G[", i, ",", i, "] = ", diagonal_G[i])
        end
        println("Maximum absolute deviation from 0.5: ",
                maximum(abs, diagonal_G .- 0.5))
        println("All entries finite: ", all(isfinite, G))
    end
finally
    MPI.Finalize()
end