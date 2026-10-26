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

function test_lmul_inv_kv!(t, ldr_ws, fft_ws, emvg, eMK)
    A_initial = Matrix{_mattype}(I, Nsite, Nsite)
    A = copy(A_initial)
    A_direct = similar(A)

    # Populate emvg using the initialized HS field.
    CalculateExpV!(emvg, eMVB, IsingField, t)

    # Assemble a dense reference from the same local blocks.
    V_dense = zeros(_mattype, Nsite, Nsite)
    for i in 1:Nlayer, b in 1:3, a in 1:3
        V_dense[i + (a - 1) * Nlayer, i + (b - 1) * Nlayer] =
            emvg.data[a, b, i]
    end

    # Warm up both methods to exclude compilation.
    for _ in 1:3
        copyto!(A, A_initial)
        lmul_inv_kv!(t, A, ldr_ws, fft_ws, emvg)
        mul!(A_direct, V_dense, eMK)
    end

    fast_times = zeros(5)
    dense_times = zeros(5)

    for sample in eachindex(fast_times)
        # Alternate execution order between samples.
        for method in (isodd(sample) ? (1, 2) : (2, 1))
            if method == 1
                # Restore the identity outside the timed section.
                copyto!(A, A_initial)

                fast_times[sample] = @elapsed begin
                    lmul_inv_kv!(t, A, ldr_ws, fft_ws, emvg)
                end
            else
                dense_times[sample] = @elapsed begin
                    mul!(A_direct, V_dense, eMK)
                end
            end
        end
    end

    fast_median = median(fast_times)
    dense_median = median(dense_times)

    println("size(A) = ", size(A))
    println("size(ldr_ws.M) = ", size(ldr_ws.M))
    println("lmul_inv_kv! times (s): ", fast_times)
    println("Dense multiplication times (s): ", dense_times)
    println("Median lmul_inv_kv! time (s): ", fast_median)
    println("Median dense time (s): ", dense_median)
    println("Speedup (dense / lmul_inv_kv!): ", dense_median / fast_median)

    max_error = maximum(abs, A - A_direct)
    matches = isapprox(A, A_direct; atol=1e-12, rtol=1e-12)

    println("Maximum absolute error: ", max_error)
    println("isapprox (atol=1e-12, rtol=1e-12): ", matches)

    @assert matches "FFT/local multiplication differs from the dense reference."

    return nothing
end

try
    if icpurank == rootcpurank
        test_lmul_inv_kv!(1, Ldr_ws, FFT_workspace, e⁻ⱽ, eMK)
    end
finally
    MPI.Finalize()
end