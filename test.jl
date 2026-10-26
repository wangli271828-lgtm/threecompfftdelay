using MPI
using LinearAlgebra
using CSV
using Dates
using TimerOutputs
using FFTW
import FFTW: AbstractFFTs
using Random
using Statistics
using Printf
using FLoops
using StableLinearAlgebra
using MuTuner

const terminal = stdout
const terminal_error = stderr
const _mattype = Float64
const _mattypeC = ComplexF64

# Set thread counts before creating FFT plans.
BLAS.set_num_threads(1)
FFTW.set_num_threads(1)

MPI.Init()

# Load the original SU2 initialization at top level.
const original_directory = pwd()
cd(@__DIR__)

try
    include(joinpath(@__DIR__, "initialize.jl"))
catch
    MPI.Finalize()
    rethrow()
finally
    redirect_stdout(terminal)
    redirect_stderr(terminal_error)
    cd(original_directory)
end

function compare_multiply_times(A, eK, fft_ws; samples=5)
    F_fft = similar(A)
    F_direct = similar(A)

    # Warm up both methods before timing.
    for _ in 1:3
        rmulK_block!(
            F_fft, A, fft_ws.d.d, fft_ws.p, fft_ws.a,
        )
        mul!(F_direct, A, eK)
        max_error = maximum(abs, F_fft - F_direct)
        println("Maximum absolute error: ", max_error)
    end

    fft_times = zeros(samples)
    direct_times = zeros(samples)

    for sample in 1:samples
        # Alternate execution order between samples.
        for method in (isodd(sample) ? (1, 2) : (2, 1))
            GC.gc()

            if method == 1
                fft_times[sample] = @elapsed rmulK_block!(
                    F_fft, A, fft_ws.d.d, fft_ws.p, fft_ws.a,
                )
            else
                direct_times[sample] = @elapsed mul!(
                    F_direct, A, eK,
                )
            end
        end
    end

    fft_median = median(fft_times)
    direct_median = median(direct_times)

    println("FFT times (s): ", fft_times)
    println("BLAS mul! times (s): ", direct_times)
    println("Median FFT time (s): ", fft_median)
    println("Median BLAS time (s): ", direct_median)
    println("Speedup (BLAS / FFT): ", direct_median / fft_median)

    # Check correctness outside the timed sections.
    println(
        "isapprox (atol=1e-12, rtol=1e-12): ",
        isapprox(F_fft, F_direct; atol=1e-12, rtol=1e-12),
    )

    return nothing
end

try
    TimerOutputs.disable_timer!(timecount)

    if icpurank == rootcpurank
        # The SU2 FFT routine includes a chemical-potential factor.
        # Use zero chemical potential for this pure hopping comparison.
        @assert iszero(μμ[1]) "Set mu = 0 for this A * eK comparison."

        A = randn(
            MersenneTwister(618), _mattype, Nlayer, Nlayer,
        )

        println("size(A) = ", size(A), ", Δτ = ", Δτ)
        println("BLAS threads = ", BLAS.get_num_threads())
        println("FFTW threads = ", FFTW.get_num_threads())

        compare_multiply_times(A, eK, FFT_workspace)
    end
finally
    MPI.Finalize()
end