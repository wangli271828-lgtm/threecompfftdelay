using MPI
using LinearAlgebra
using Random
using Statistics
using Printf
using CSV
using Dates
using TimerOutputs
using FFTW
using FLoops
using StableLinearAlgebra
using MuTuner
using JDQMCMeasurements
using JLD2

MPI.Init()

const _mattype = Float64
const _mattypeC = ComplexF64

include("initialize.jl")
include("montecarlo.jl")
include("measure.jl")
include("postprocess.jl")


println("--- Task $icpurank: start warm steps ---")
# wram steps
for step ∈ 1:WarmStep
  if step % 10 == 0
    print_process(icpurank, step, clock, 0)
  end
  if step % 2 == 1
    UpdateIsingKet!(+1, step, IsingField, GreenFunc, WeightSign, Accept_count, Avg_sign, eⱽ, e⁻ⱽ, Indices, MatDelay, MatMeasure, MatUneqMeasure, Fs, Ldr_ws, chemical_potential_tuner, FFT_workspace; warm=true)
  else
    UpdateIsingKet!(-1, step, IsingField, GreenFunc, WeightSign, Accept_count, Avg_sign, eⱽ, e⁻ⱽ, Indices, MatDelay, MatMeasure, MatUneqMeasure, Fs, Ldr_ws, chemical_potential_tuner, FFT_workspace; warm=true)
  end
end

println("--- Task $icpurank: start measure steps ---")
# measure steps
for step ∈ 1:MeasureStep
  if step % 10 == 0
    print_process(icpurank, step, clock, 1)
  end
  if step % 2 == 1
    UpdateIsingKet!(+1, step, IsingField, GreenFunc, WeightSign, Accept_count, Avg_sign, eⱽ, e⁻ⱽ, Indices, MatDelay, MatMeasure, MatUneqMeasure, Fs, Ldr_ws, chemical_potential_tuner, FFT_workspace, warm=false)
  else
    UpdateIsingKet!(-1, step, IsingField, GreenFunc, WeightSign, Accept_count, Avg_sign, eⱽ, e⁻ⱽ, Indices, MatDelay, MatMeasure, MatUneqMeasure, Fs, Ldr_ws, chemical_potential_tuner, FFT_workspace, warm=false)
  end
  # data as a function of the monte carlo step
  EnergyPath[step] = MatMeasure.E[3] / step
  SignPath[step] = Avg_sign[1] / step
end

# average and output

println("--- Task $icpurank: enter the post process ---")

OutputAverage(Avg_sign, Accept_count, EnergyPath, SignPath, MatMeasure, MatUneqMeasure)

println("--- Task $icpurank: finalize ---")

# end and output
print_timecount(icpurank)

MPI.Finalize()
exit()
