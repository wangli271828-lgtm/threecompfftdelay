include("2lattices.jl")
include("2utilities.jl")
include("2multiply.jl")
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
parms = CSV.File("system.csv", delim=" ", comment="#", transpose=true)
const L              = parms.l[1]
const Nflavor             = parms.ns[1]
const U1              = parms.u1[1]
const U2              = parms.u2[1]
const U3              = parms.u3[1]
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

# exact HS decomposition
const λ1 = acos(exp(Δτ * U1 / 2))
const λ2 = acos(exp(Δτ * U2 / 2))
const λ3 = acos(exp(Δτ * U3 / 2))

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
#bondGroup在on-site相互作用时维度是(nlayer,1)
if     latticetype == 1
  const K_layer, BondGroup, SiteMap, Nlayer = Makesquare()
  elseif latticetype == 2
  const K_layer, BondGroup, SiteMap, Nlayer = MakeTriangular()
  elseif latticetype == 3
  const K_layer, BondGroup, SiteMap, Nlayer = MakeHoneycomb()
  elseif latticetype == 4
  const K_layer, BondGroup, SiteMap, Nlayer = MakeKagome()
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


# Three-component kinetic matrix, matching Fortran init_expK().
# Basis: all sites of component 1, then component 2, then component 3.
# K_layer already includes -t0; no chemical-potential term is added here.
const Nsite = 3 * Nlayer
const K = kron(Diagonal(_mattype[1.0, 1.0, 1.0]), K_layer)

# Reuse one eigendecomposition for full-step and half-step propagators.
println("three-component kinetic propagators")
const eK, eKHalf, eMK, eMKHalf = let F = eigen(Hermitian(K))
  Q = F.vectors
  ε = F.values
  (
    Q * Diagonal(exp.(-Δτ .* ε)) * Q',
    Q * Diagonal(exp.(-(Δτ / 2) .* ε)) * Q',
    Q * Diagonal(exp.(Δτ .* ε)) * Q',
    Q * Diagonal(exp.((Δτ / 2) .* ε)) * Q',
  )
end
println("three-component kinetic propagators finished")



FFT_workspace = InitializeFFT_square(icpurank)

IsingField = InitializeIsing()

eVB, eMVB = InitializeThreeComponentHS(λ1, λ2, λ3)

###一定要使用相互作用矩阵的特殊结构！
eⱽ  = LocalHSMatrix(zeros(_mattype, 3, 3, Nlayer))
e⁻ⱽ = LocalHSMatrix(zeros(_mattype, 3, 3, Nlayer))

###Initialize LDR,给Fs，Ldr_ws正确的维度
"""
以下结构体来自StableLinearAlgebra包：
struct LDRWorkspace{T<:Number,E<:AbstractFloat}
    qr_ws::QRWorkspace{T,E}
    lu_ws::LUWorkspace{T}
    M::Matrix{T}
    M′::Matrix{T}
    M″::Matrix{T}
    v::Vector{E}
end
ldr_ws = ldr_workspace(G)，G多大，ldr_ws就多大
Fs创建了Ngroup个LDR对象，其属于AbstractVector{LDR{T,E}}
"""
Fs, Ldr_ws = InitializeLDR()
CalculateUDV!(Fs, Ldr_ws, FFT_workspace, eⱽ)


