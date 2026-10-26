include("lattices.jl")
include("utilities.jl")

# 这两句命令似乎对计算效率没有影响
# BLAS.set_num_threads(1)
# FFTW.set_num_threads(1)

#########################
### system parameters ###
#########################

# MPI
const comm           = MPI.COMM_WORLD
const ncpurank       = MPI.Comm_size(comm)
const icpurank       = MPI.Comm_rank(comm)
const rootcpurank    = 0

# time
const timecount      = TimerOutput()
const clock          = [now()]

# system
parms = CSV.File("system0.csv", delim=" ", comment="#", transpose=true)
const L              = parms.l[1]
const Ns             = parms.ns[1]
const U              = parms.u[1]
const t0             = parms.t0[1]

# Chemical potential
const ρ              = parms.rho1[1] * 1.0 / parms.rho2[1]
const μ              = parms.mu[1]
const μμ             = Vector([μ, μ]) # μμ[1]存储当前的μ, μμ[2]存储上一个μ
const mutuner        = parms.mutuner[1]::Bool

# Monte Carlo basic
const WarmStep       = parms.warmstep[1]
const MeasureStep    = parms.measurestep[1]
const seed           = parms.seed[1]

# QMC basic
const β              = parms.beta[1]
const Interval       = parms.interval[1]
const P0             = parms.p0[1]
const latticetype    = parms.latticetype[1]

# equal time measurement
const Δτ             = β / Interval
const Nugroup        = round(Integer, Interval / P0)

# unequal time measurement
const ueqtimeMeasure = parms.uneq[1]::Bool

################
### Log File ###
################

if icpurank == rootcpurank
  # Log File, output measurements
  isdir("log") || mkdir("log")
  # Core file, output results per core before MPI_Avg
  isdir("core") || mkdir("core")
end
MPI.Barrier(comm)

if icpurank == rootcpurank
  ilogfile  = open("log-$icpurank.log", "w")
  redirect_stdout(ilogfile)
else
  # ilogfile = open("log/log-$icpurank.log", "w")
end
ierrfile = open("log/error-$icpurank.log", "w")
redirect_stderr(ierrfile)

println("Start the simulation on cpu $icpurank")
if icpurank == rootcpurank || icpurank == rootcpurank + 1
  println(parms[1])
end

##########################
### Initialize lattice ###
##########################

if     latticetype == 1
  const  K ,           BondGroup, SiteMap, Nlayer = MakeSquare()
  elseif latticetype == 2
  const  K ,           BondGroup, SiteMap, Nlayer = MakeTriangular()
  elseif latticetype == 3
  const  K ,           BondGroup, SiteMap, Nlayer = MakeHoneycomb()
  elseif latticetype == 4
  const  K ,           BondGroup, SiteMap, Nlayer = MakeKagome()
end


if latticetype in (2,3,4)
  # lattice vectors
  const  a1 = Array{Float64}([sqrt(3), 0])
  const  a2 = Array{Float64}([sqrt(3) / 2, 3 / 2])
  #const a3 = Array{Float64}([sqrt(3) / 2, 1 / 2]) # sublattice 1 -> 2
  const  a3 = Array{Float64}([0, 0])
  const  b1 = 2 * pi / 3 * Array{Float64}([sqrt(3), -1])
  const  b2 = 2 * pi / 3 * Array{Float64}([0, 2])
  # dirac K point
  const K_diarc = 4 * pi / 3 / sqrt(3) * Array{Float64}([1, 0])
  const K_diarc2 = 2 * pi / 3 * Array{Float64}([1/ sqrt(3), 1])
  # Van-Hove sigularity
  const m_1 = 2 * pi / sqrt(3) * Array{Float64}([+sqrt(3) / 2, 1 / 2])
  const m_2 = 2 * pi / sqrt(3) * Array{Float64}([0, -1])
  const m_3 = 2 * pi / sqrt(3) * Array{Float64}([-sqrt(3) / 2, 1 / 2])
  # vbs const parameter
  const ω  = exp(1im * 2 / 3 * pi)
  const ω2 = exp(1im * 4 / 3 * pi)
end


# K maticies exponential
println("eK")
const eK      = exp(-Δτ * K)
#println("eKHalf")
#const eKHalf  = exp(-Δτ * K / 2.0)
#println("eMK")
#const eMK     = exp(Δτ * K)
#println("eMKHalf")
#const eMKHalf = exp(Δτ * K / 2.0)

println("manual eKHalf")
Fhalf = eigen(Hermitian(K))
const eKHalf =
    Fhalf.vectors *
    Diagonal(exp.(-(Δτ/2) .* Fhalf.values)) *
    Fhalf.vectors'
println("manual eKHalf finished")

println("eMK")
F = eigen(Hermitian(K))
println("eigen finished")
tmp_eMK =
    F.vectors *
    Diagonal(exp.(Δτ .* F.values)) *
    F.vectors'
println("manual eMK finished")
const eMK = tmp_eMK
println("emkfinish")

tmp_eMKHalf=
    F.vectors *
    Diagonal(exp.(Δτ .* F.values ./2.0)) *
    F.vectors'
println("manual eMKHalf finished")
const eMKHalf = tmp_eMKHalf
println("eMKHalffinish")

println(typeof(K))
println(typeof(F.values))
println(typeof(F.vectors))
println(typeof(eK))
println(typeof(eKHalf))
println(typeof(eMK))
println(typeof(eMKHalf))

println("all exp finished")

# delay update steps
const nd = 2^(min(6, floor(Int, log(2, Nlayer / 20))))

if icpurank == rootcpurank
  println("True nd: ", nd)
end

############################
### Initialize HS fields ###
############################

# number of HS fields
const Nhs = 1
# local state in base-2 digits, ±1 for each group of HS field
const NIsingState = (4)^Nhs

# random seed
Random.seed!(seed + icpurank)
const η, γ, eVB, eMVB, IsingΔ = InitializeBondHS()
# IsingField is the intial HS field on the lattice bonds.
IsingField = InitializeIsing()

#######################
### Initialize expV ###
#######################

# identity matrices for constructing evg and emvg
const Iₗ = Diagonal(fill(1.0, L))
const Iₛ = Diagonal(fill(1.0, L * L))
const Iₙ = Diagonal(fill(1.0, Nlayer))

eⱽ  = Diagonal(zeros(_mattype, Nlayer))
e⁻ⱽ = Diagonal(zeros(_mattype, Nlayer))

#######################
### Initialize Sign ###
#######################

WeightSign = [1.0]

######################
### INITIALIZE FFT ###
######################

FFT_workspace = InitializeFFT_kagome(icpurank)

######################
### INITIALIZE LDR ###
######################
Fs, Ldr_ws = InitializeLDR()
CalculateUDV!(Fs, Ldr_ws, FFT_workspace, eⱽ)
if icpurank == rootcpurank
  println("--- the LDR groups are initialized---")
end

##########################
### Initialize MuTuner ###
##########################

chemical_potential_tuner = MuTunerLogger(n₀=ρ, β=β * 1.0, V=Nlayer, u₀=1.0, μ₀=μ, c=0.5)

###########################
### Initialize Matrices ###
###########################

# Green function
GreenFunc     = zeros(_mattype, Nlayer, Nlayer)
# Average sign
Avg_sign      = zeros(_mattype, 1)
# Acceptance ratio
Accept_count  = zeros(Integer, 1)

# measurement
MatMeasure    = InitializeMeasure_kagome(GreenFunc)

# unequal time measurement
MatUneqMeasure = InitializeUneqMeasure_kagome(GreenFunc)

# pre-allocated Temporary array
Indices       = Vector{Int}(undef, 1)
MatDelay      = InitializeDelay(GreenFunc)

# monte carlo path
EnergyPath    = zeros(_mattype, MeasureStep)
SignPath      = zeros(_mattype, MeasureStep)
