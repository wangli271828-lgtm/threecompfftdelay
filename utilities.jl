include("multiply.jl")

# ----- Spin-1/2 -----
""" Initialize the bond HS decomposition
Δτ::AbstractFloat,
u::Array{<:AbstractFloat,1},
nisingstate::Integer
calculate on-site interaction term evb;
calculate isingΔ
"""

function InitializeBondHS()
  # HS parameters
  a = exp(-0.5 * U * Δτ)
  d = sqrt(8 + a^2 * (3 + a^2)^2)
  γi = [(a * (3 + a^2) + d) / d, (-a * (3 + a^2) + d) / d, (-a * (3 + a^2) + d) / d, (a * (3 + a^2) + d) / d] # i stands for ising=-2,-1,+1,+2 order may be important ！！！
  ηi = [-acos((a + 2 * a^3 + a^5 - (a^2 - 1) * d) / 4), -acos((a + 2 * a^3 + a^5 + (a^2 - 1) * d) / 4), acos((a + 2 * a^3 + a^5 + (a^2 - 1) * d) / 4), acos((a + 2 * a^3 + a^5 - (a^2 - 1) * d) / 4)]
  # --- γ phase and eVB ---
  #NIsingState local state in base-4 digits
  evb = zeros(_mattype, NIsingState)
  emvb = zeros(_mattype, NIsingState)
  """γ phase: γ({l}) = ∏γ(l)"""
  for ist ∈ 1:NIsingState
    #isingbit = digits(ist - 1, base = 4, pad = Nhs) .+ 1#从bond model中保留，用4进制生成所有的辅助场组合
    # singlet bond
    hsbd = ηi[ist] # 吸引势和排斥势分解方案的唯一不同就是此处没有 "i"
    ehsbd = exp(hsbd)
    # store exp(vmat).
    evb[ist] = ehsbd
    emvb[ist] = inv(ehsbd)
    # γ phase
    #γ[ist] = prod(γi[ist])
  end
  # Δ = eVi(new)/eVi(old) - 1 so that B(new)=(Δ+1)*B(old); τ=nΔτ
  isingΔ = zeros(_mattype, NIsingState, NIsingState)
  for newist ∈ 1:NIsingState
    for oldist ∈ 1:NIsingState
      # I for Diagonal(Vector{ComplexF64}([1,1,1,1]))
      isingΔ[newist, oldist] = evb[newist] * emvb[oldist] - 1.0
    end
  end
  return ηi, γi, evb, emvb, isingΔ
end

function CalculateIsing(oldist::Integer, newist::Integer)
  # Δ = eVi(new)/eVi(old) - 1 so that B(new)=(Δ+1)*B(old); τ=nΔτ
  # I for Diagonal(Vector{ComplexF64}([1,1,1,1]))
  isingΔ = eVB[newist] * eMVB[oldist] - 1.0
  return isingΔ
end


""" Initialize the Ising field on lattice bonds
bondgroup::Array{Tuple{Integer,Integer},2}
interval::Integer,
nisingstate::Integer
"""
function InitializeIsing()
  # field dimension
  nbond = size(BondGroup, 1)
  isingfield = rand(1:NIsingState, nbond, Interval)
  isingfield
end
# ----- -----

"""
Initialize FFT for  kagome lattice
"""
function InitializeFFT_kagome(cpurank::Integer)

  println("--- Task $cpurank: start FFT steps ---")

  # fix digital
  _C64 = ComplexF64

  # create submatrix A, Aᵀ
  Aᵀ = zeros(_C64, L, L)
  Aᵀ[1, L] = 1.0
  for i in 1:L-1
    Aᵀ[i+1, i] = 1.0
  end
  A = transpose(Aᵀ)

  # create submatrix kr, kb, kc
  k₁ = zeros(_C64, 3, 3)
  k₂ = zeros(_C64, 3, 3)
  k₃ = zeros(_C64, 3, 3)
  k₁[2, 1] = 1.0
  k₂[3, 1] = 1.0
  k₃[3, 2] = 1.0

  # create diagonal matrix
  Λₐ = Diagonal(exp_circulant(A[1, :]))
  Λₐᵀ = Diagonal(exp_circulant(Aᵀ[1, :]))

  Λₛ = kron(Iₗ, kron(Λₐ + Iₗ, k₁)) + kron(Λₐ + Iₗ, kron(Iₗ, k₂)) + kron(kron(Iₗ, Iₗ) + kron(Λₐ, Λₐᵀ), k₃)
  Λₛᵀ = kron(Iₗ, kron(Λₐᵀ + Iₗ, k₁')) + kron(Λₐᵀ + Iₗ, kron(Iₗ, k₂')) + kron(kron(Iₗ, Iₗ) + kron(Λₐᵀ, Λₐ), k₃')
  Λₖ = Δτ * t0 * (Λₛ + Λₛᵀ)

  Λ = _create_Kd_kagome(1.0, Λₖ)
  Λ⁻¹ = _create_Kd_kagome(-1.0, Λₖ)
  Λₕ = _create_Kd_kagome(1.0 / 2, Λₖ)
  Λₕ⁻¹ = _create_Kd_kagome(-1.0 / 2, Λₖ)

  DiagMat = FFTDiagonal(Λ, Λ⁻¹, Λₕ, Λₕ⁻¹)

  # create FFT plan
  m₁ = rand(_C64, L, L)
  m₂ = zeros(_C64, L, L)
  F = plan_fft(m₁, (1, 2))  # 创建 FFT 计划
  F⁻¹ = plan_ifft(m₁, (1, 2))  # 创建逆 FFT 计划
  Plan = FFTPlan(m₁, m₂, F, F⁻¹)

  # create temporary array
  M₁ = zeros(_C64, Nlayer, Nlayer)
  M₂ = zeros(_C64, Nlayer, Nlayer)
  Array_fft = FFTmats(M₁, M₂)


  println("--- Task $cpurank: finish FFT steps ---")


  return FFTWorkspace(DiagMat, Plan, Array_fft)
end


"""
Inner function to create and return a K_diagmatrices structure
for kagome lattice
"""
function _create_Kd_kagome(t::Float64, Λk::AbstractMatrix{T}) where {T<:Complex}
  Λk_vec = [exp(t * Λk[(i-1)*3+1:i*3, (i-1)*3+1:i*3]) for i ∈ 1:L*L]
  Λ = BlockDiagonal(Λk_vec)
  return Kpμ(Λ, t0 * Δτ * t)
end

function InitializeMeasure_kagome(g::Matrix{E}) where {E<:AbstractFloat}

  Dgfunc = zeros(E, size(g))
  Dgfunc2 = zeros(E, size(g))
  Energy = zeros(E, 3) # kinetic, tcur*3
  CorFuncDD = zeros(E, L * L, L * L)
  CorFuncPP = zeros(E, Nlayer, Nlayer)
  Matrix_pair = zeros(E, Nlayer, Nlayer)
  N_c = zeros(E, 1)
  P2 = zeros(E, 1) # double occupancy probability
  G_db = zeros(E, Nlayer) # histogram for double occupancy
  Hist_db = zeros(E, 20)
  MeanGreenFunc = zeros(E, size(g))
  EMS = zeros(E, 1) # equal time magnetic susceptibility
  ChargeSqr = zeros(E, Nlayer, Nlayer)
  Charge = zeros(E, 1) # total particle number, summed over sites and flavors

  return MeasureMats(Dgfunc, Dgfunc2, Energy, CorFuncDD, CorFuncPP, Matrix_pair, N_c, P2, G_db, Hist_db, MeanGreenFunc, EMS, ChargeSqr, Charge)
end

function InitializeUneqMeasure_kagome(g::Matrix{E}) where {E<:AbstractFloat}
  if ueqtimeMeasure == false
    return nothing
  elseif ueqtimeMeasure == true
    G1 = zeros(E, size(g))
    G2 = zeros(E, size(g))
    # G3 = zeros(E, size(G2))
    # G4 = zeros(E, size(G2))
    # G5 = zeros(E, size(G2))

    Gt0 = zeros(E, size(G2))
    G0t = zeros(E, size(G2))

    Gt0s = zeros(E, size(G2, 1), size(G2, 2), Interval + 1)
    G0ts = zeros(E, size(G2, 1), size(G2, 2), Interval + 1)

    UtSS = zeros(E, size(Gt0s, 3))
    MagSus = zeros(E, 1)
    GKt0s = zeros(_mattypeC, size(Gt0s, 3))
    QPW = zeros(E, 1)

    return UneqMats(G1, G2, Gt0, G0t, Gt0s, G0ts, UtSS, MagSus, GKt0s, QPW)
  end
end

@doc raw"""

Initialize the LDR matrices and vectors.

# Fields

Return ldr_ws, Fs
- `qr_ws::QRWorkspace{T,E}`: [`QRWorkspace`](@ref) for calculating column pivoted QR factorization without dynamic memory allocations.
- `Fs::Vector{LDR{T,E}}`: A vector of ``N_s`` LDR factorizations to represent the matrices ``B(0,\tau)`` and ``B(\tau,\beta)``.
"""
function InitializeLDR(
)
  # calculate scratch matrix
  G′ = Matrix{_mattype}(I, Nlayer, Nlayer)

  # construct vector of LDR factorization to represent B(τ,0) and B(β,τ) matrices
  ldr_ws = ldr_workspace(G′)
  Fs = ldrs(G′, Nugroup)
  return Fs, ldr_ws
end
# ----- -----

function InitializeDelay(g::Matrix{E}) where {E<:AbstractFloat}

  Idv = zeros(Integer, 1)
  Vec1 = zeros(E, Nlayer, nd)
  VecT1 = zeros(E, nd, Nlayer)

  return DelayMats(Idv, Vec1, VecT1)
end


"""
	Calculate the interaction term exp(Vg) and exp(-Vg) after HS decomposition.
	Return evg and emvg are τ-dependent matrix. size(eVg)=(nsite,nsite) to save the memory
	indices is a reused array.
	为了省内存, 每一个evg都是不含时的中间矩阵, 而含时的evb只储存了spin自由度的矩阵
"""
function CalculateExpVg!(evg::Diagonal{E,Vector{E}}, time::Integer, indices::Vector{<:Integer},) where {E<:AbstractFloat}
  # calculate the post-HS interaction term from scratch
  @timeit timecount label = "CalculateExpVg" begin
    @views begin
      evg .= Iₙ # start from an identity matrix
      nbond = size(BondGroup, 1)#BondGroup记录了所有(sitei,sitej)形式的bond
      for bi ∈ 1:nbond
        i, j = BondGroup[bi]
        #indices: [i,Nlayer+i]
        indices[1] = i #更改了indices的维度和等式，没改变功能
        evg[indices[1], indices[1]] = eVB[IsingField[bi, time]] #evbi[a,b]
      end
    end
  end

  return nothing
end

function CalculateExpMVg!(emvg::Diagonal{E,Vector{E}}, time::Integer, indices::Vector{<:Integer},) where {E<:AbstractFloat}
  # calculate the post-HS interaction term from scratch
  @timeit timecount label = "CalculateExpVg" begin
    @views begin
      emvg .= Iₙ # start from an identity matrix
      nbond = size(BondGroup, 1)
      for bi ∈ 1:nbond
        i, j = BondGroup[bi]
        indices[1] = i
        emvg[indices[1], indices[1]] = eMVB[IsingField[bi, time]]
      end
    end
  end

  return nothing
end

# --- Stabilization, Green function ---
"""
	The LDR decomposition for stabilizing the matrix multiplication
	F'=FU=[LDR]*eK*eV=(LL')D'R'
"""
function CalculateUDV!(
  Fs::AbstractVector{LDR{T,E}},
  ldr_ws::LDRWorkspace{T,E},
  fft_ws::FFTWorkspace{H,E},
  evg::Diagonal{T,Vector{T}}
) where {T<:Number,H<:Number,E<:AbstractFloat}

  indices = Vector{Int}(undef, 1)

  # iterating from l=Lτ => l=1
  for t ∈ Interval:-1:1
    n = ceil(Int, t / P0)
    l′ = mod1(t, P0)
    if n < Nugroup && l′ == P0
      copyto!(Fs[n], Fs[n+1])
    end
    B_βτ = Fs[n]::LDR{T,E}

    # B<*eK*eV
    rmul_kv!(t, B_βτ.R, ldr_ws, fft_ws, indices, evg)

    # LDR factorization
    if l′ == 1
      LDR_R!(B_βτ, ldr_ws)
    end
  end

  return nothing
end

"""
	Calculate the Green function from scratch using a stabilization method.
	The stabilized group will be updated.
	For updating forwards, only the uket group changes.
	For updating forwards, only the ubra group changes.
	We should count in the new isingfield i.e. new exp(V) within p0 time slice.
"""
function CalculateGreenFunc!(
  time::Integer, direction::Integer, indices::Array{<:Integer,1},
  greenfunc::Array{T,2}, weightsign::Array{T,1},
  evg::Diagonal{T,Vector{T}}, emvg::Diagonal{T,Vector{T}},
  Fs::AbstractVector{LDR{T,E}},
  ldr_ws::LDRWorkspace{T,E},
  fft_ws::FFTWorkspace{H,E},
  uneq_mats::Union{UneqMats{E},Nothing};
  warm::Bool,
) where {T<:Number,H<:Number,E<:AbstractFloat}
  @views begin


    if ueqtimeMeasure && !warm
      weight_sign = weightsign[1]
      G₂, Gₜₘ = uneq_mats.g₁, uneq_mats.g₂
      Gτ0, G0τ = uneq_mats.gt0, uneq_mats.g0t
      Gτ0s, G0τs = uneq_mats.gt0s, uneq_mats.g0ts
    end

    G = greenfunc
    n = ceil(Int, time / P0)

    # if iterating from l=1 => l=Lβ
    if direction == +1
      if time + 1 == 1
        # must be ``Fs[1]'' because we are using the last group that has been updated, but Fs[1] hasn't been updated along the other direction.
        copyto!(Fs[1], Fs[2])
        B_β0 = Fs[1]::LDR{T,E}

        for t ∈ P0:-1:1
          # R*eK*eV
          rmul_kv!(t, B_β0.R, ldr_ws, fft_ws, indices, evg)
        end

        # LDR factorization
        LDR_R!(B_β0, ldr_ws)

        # calcualte Green's function
        # G(0,0) = G(β,β) = [I + B(β,0)]⁻¹
        Inv_IpA!(G, B_β0, ldr_ws)

        if ueqtimeMeasure && !warm

          # G(β, 0) = I - G(0,0)
          Gτ0 .= Iₙ .- G
          # G(0, β) = -G(0,0)
          G0τ .= -G

          # store G(τ,0), G(0,τ)
          Trotter2nd!(G₂, Gτ0, fft_ws, Gₜₘ)
          Gτ0s[:, :, Interval+1] .+= G₂ .* weight_sign
          Trotter2nd!(G₂, G0τ, fft_ws, Gₜₘ)
          G0τs[:, :, Interval+1] .+= G₂ .* weight_sign


          # G(0,0) = G(β,β)
          Gτ0 .= G

          # G(0,0+) = −[I − G(0,0)]
          G0τ .= G .- Iₙ

          # store G(τ,0), G(0,τ)
          Trotter2nd!(G₂, Gτ0, fft_ws, Gₜₘ)
          Gτ0s[:, :, 1] .+= G₂ .* weight_sign
          Trotter2nd!(G₂, G0τ, fft_ws, Gₜₘ)
          G0τs[:, :, 1] .+= G₂ .* weight_sign

        end

      else
        if n == 1
          # initialize Fs[1]
          ldr!(Fs[n], I)
        elseif n > 1 && n < Nugroup
          copyto!(Fs[n], Fs[n-1])
        end

        B_βτ = Fs[n+1]::LDR{T,E}
        B_τ0 = Fs[n]::LDR{T,E}

        for t ∈ time-P0+1:time
          # eK*eV*L
          lmul_kv!(t, B_τ0.L, ldr_ws, fft_ws, indices, evg)
        end

        # LDR factorization
        LDR_L!(B_τ0, ldr_ws)

        # calcualte Green's function
        # calculate G(τ,τ) = [I + B(τ,0)⋅B(β,τ)]⁻¹
        Inv_IpUV!(G, B_τ0, B_βτ, ldr_ws)

        if ueqtimeMeasure && !warm

          # calculate G(τ,0) = [B⁻¹(τ,0) + B(β,τ)]⁻¹
          inv_invUpV!(Gτ0, B_τ0, B_βτ, ldr_ws)

          # calculate G(0,τ) = -[B⁻¹(β,τ) + B(τ,0)]⁻¹
          inv_invUpV!(G0τ, B_βτ, B_τ0, ldr_ws)
          G0τ .= -G0τ

          # store G(τ,0), G(0,τ)
          Trotter2nd!(G₂, Gτ0, fft_ws, Gₜₘ)
          Gτ0s[:, :, time+1] .+= G₂ .* weight_sign
          Trotter2nd!(G₂, G0τ, fft_ws, Gₜₘ)
          G0τs[:, :, time+1] .+= G₂ .* weight_sign

        end

      end

      if ueqtimeMeasure && !warm
        for t ∈ time+1:time+P0-1

          # G(l,0) = B[l]⋅G(l-1,0)
          lmul_kv!(t, Gτ0, ldr_ws, fft_ws, indices, evg)

          # G(0,l) = G(0,l-1)⋅B⁻¹[l]
          rmul_inv_kv!(t, G0τ, ldr_ws, fft_ws, indices, emvg)

          # store G(τ,0), G(0,τ)
          Trotter2nd!(G₂, Gτ0, fft_ws, Gₜₘ)
          Gτ0s[:, :, t+1] .+= G₂ .* weight_sign
          Trotter2nd!(G₂, G0τ, fft_ws, Gₜₘ)
          G0τs[:, :, t+1] .+= G₂ .* weight_sign

        end
      end

      # if iterating from l=Lτ => l=1
    elseif direction == -1
      if time == Interval
        # must be ``end-1'' because we are using the last group that has been updated, but Fs[end] hasn't been updated along the other direction.
        copyto!(Fs[end], Fs[end-1])
        B_β0 = Fs[end]::LDR{T,E}
        for t ∈ Interval-P0+1:Interval
          # eK*eV*L
          lmul_kv!(t, B_β0.L, ldr_ws, fft_ws, indices, evg)
        end
        # LDR factorization
        LDR_L!(B_β0, ldr_ws)
        # calcualte Green's function
        Inv_IpA!(G, B_β0, ldr_ws)

        if ueqtimeMeasure && !warm

          # G(0,0) = G(β,β)
          Gτ0 .= G

          # G(0,0+) = −[I − (β,β)]
          G0τ .= G .- Iₙ

          # store G(τ,0), G(0,τ)
          Trotter2nd!(G₂, Gτ0, fft_ws, Gₜₘ)
          Gτ0s[:, :, 1] .+= G₂ .* weight_sign
          Trotter2nd!(G₂, G0τ, fft_ws, Gₜₘ)
          G0τs[:, :, 1] .+= G₂ .* weight_sign


          # G(β, 0) = I - G(0,0)
          Gτ0 .= Iₙ - G
          # G(0, β) = -G(0,0)
          G0τ .= -G

          # store G(τ,0), G(0,τ)
          Trotter2nd!(G₂, Gτ0, fft_ws, Gₜₘ)
          Gτ0s[:, :, Interval+1] .+= G₂ .* weight_sign
          Trotter2nd!(G₂, G0τ, fft_ws, Gₜₘ)
          G0τs[:, :, Interval+1] .+= G₂ .* weight_sign

        end

      else
        if n == Nugroup - 1
          # initialize Fs[end]
          ldr!(Fs[end], I)
        elseif n < Nugroup - 1 && n >= 1
          copyto!(Fs[n+1], Fs[n+2])
        end
        B_βτ = Fs[n+1]::LDR{T,E}
        B_τ0 = Fs[n]::LDR{T,E}
        for t ∈ time+P0:-1:time+1
          # R*eK*eV
          rmul_kv!(t, B_βτ.R, ldr_ws, fft_ws, indices, evg)
        end
        # LDR factorization
        LDR_R!(B_βτ, ldr_ws)
        # calcualte Green's function

        Inv_IpUV!(G, B_τ0, B_βτ, ldr_ws)

        if ueqtimeMeasure && !warm

          # calculate G(τ,0) = [B⁻¹(τ,0) + B(β,τ)]⁻¹
          inv_invUpV!(Gτ0, B_τ0, B_βτ, ldr_ws)
          # calculate G(0,τ) = -[B⁻¹(β,τ) + B(τ,0)]⁻¹
          inv_invUpV!(G0τ, B_βτ, B_τ0, ldr_ws)
          G0τ .= -G0τ

          # store G(τ,0), G(0,τ)
          Trotter2nd!(G₂, Gτ0, fft_ws, Gₜₘ)
          Gτ0s[:, :, time+1] .+= G₂ .* weight_sign
          Trotter2nd!(G₂, G0τ, fft_ws, Gₜₘ)
          G0τs[:, :, time+1] .+= G₂ .* weight_sign

        end

      end

      if ueqtimeMeasure && !warm
        for t ∈ time:-1:time-P0+2

          # G(l-1, 0) = B⁻¹[l]⋅G(l,0)
          lmul_inv_kv!(t, Gτ0, ldr_ws, fft_ws, indices, emvg)

          # G(0, l-1) = G(0,l)⋅B[l]
          rmul_kv!(t, G0τ, ldr_ws, fft_ws, indices, evg)

          # store G(τ,0), G(0,τ)
          Trotter2nd!(G₂, Gτ0, fft_ws, Gₜₘ)
          Gτ0s[:, :, t] .+= G₂ .* weight_sign
          Trotter2nd!(G₂, G0τ, fft_ws, Gₜₘ)
          G0τs[:, :, t] .+= G₂ .* weight_sign

        end
      end

    else
      println("Wrong Direction, $direction")
    end
  end

  return nothing
end
# --- ---


@doc raw"""
Update the chemical potential ``\mu`` in the simulation to approach the target density/filling.
"""
function update_chemical_potential!(
  G::Matrix{T}; step::Integer,
  chemical_potential_tuner::MuTunerLogger{E,T}
) where {T<:Number,E<:AbstractFloat}

  @timeit timecount label = "update μ" begin
    # record the initial chemical potential
    # μμ[2] = μμ[1]

    # calculate average density
    n = 2 * measure_n(G)

    # calculate ⟨N²⟩
    Nsqrd = measure_Nsqrd(G, G)

    if icpurank == rootcpurank
      println(", Δρ=", n - ρ, ", μ=", μμ[1])
    end

    # update the chemical potential
    if abs(n - ρ) > 1e-3
      μμ[1] = MuTuner.update!(μtuner=chemical_potential_tuner, n=n, N²=Nsqrd, s=1.0)
    end
    # # update fermion path integrals
    # V = fermion_path_integral.V
    # @. V += -μμ[1] + μμ[2]

    # # update/calculate propagator matrices
    # calculate_propagators!(B, fermion_path_integral, calculate_exp_K=false, calculate_exp_V=true)

    # # update the Green's function
    # logdetG, sgndetG = calculate_equaltime_greens!(G, fermion_greens_calculator, B)
  end

  return nothing
end

function print_process(cpu_rank::Integer, step::Integer, clock::Vector{DateTime}, worm::Integer)
  if worm == 0
    st = "warm"
    TotalStep = WarmStep
  elseif worm == 1
    st = "mearsure"
    TotalStep = MeasureStep
  end
  if cpu_rank % 7 == 0
    print("Task $cpu_rank: $st step $step ", "process: ")
    @printf("%.1f", (step / TotalStep * 100))
    print("% ", now(), "  Memory usage: ")
    @printf("%.1f", ((1 - Sys.free_memory() / Sys.total_memory()) * 100))
    println("%")
    if step >= 2
      print("End time: ", now() + (now() - clock[1]) * (TotalStep - step))
    end
  else
    # logfile = open("log/log-$cpu_rank.log", "a")
    # print(logfile, "$st step $step ", "process: ")
    # @printf(logfile, "%.1f", (step / TotalStep * 100))
    # print(logfile, "% ", now())
    # if step >= 2
    #   println(logfile, ", End time: ", now() + (now() - clock[1]) * (TotalStep - step))
    # end
    # flush(logfile)
    # close(logfile)
  end
  clock[1] = now()

  return nothing
end

function print_timecount(cpu_rank::Integer,)
  if cpu_rank % 7 == 0
    println("End at: ", now())
  end
  if cpu_rank == rootcpurank
    Ωfile = open("result.txt", "a")
    show(Ωfile, timecount)
    flush(Ωfile)
    close(Ωfile)
  end

  return nothing
end
