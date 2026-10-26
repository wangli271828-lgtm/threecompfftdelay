include("Block.jl")

####################
#region Structure: μ
####################

struct Kpμ{T<:Number,E<:AbstractFloat}
  Kd::BlockDiagonal{T}
  t::E
end
######################
#region Structure: FFT
######################

"""
Diagonal matrices for FFT
"""
struct FFTDiagonal{T<:Number,E<:AbstractFloat}
  d::Kpμ{T,E}
  d⁻¹::Kpμ{T,E}
  dₕ::Kpμ{T,E}
  dₕ⁻¹::Kpμ{T,E}
end

"""
Prepare plan for FFT
"""
struct FFTPlan{T<:Number}
  m₁::Matrix{T}
  m₂::Matrix{T}
  f::FFTW.cFFTWPlan{T}
  f⁻¹::AbstractFFTs.ScaledPlan{T}
end

"""
pre allocated matrices for FFT
"""
struct FFTmats{T<:Number}
  M₁::Matrix{T}
  M₂::Matrix{T}
end

struct FFTWorkspace{T<:Number,E<:AbstractFloat}

  "Diagonal matrices for FFT"
  d::FFTDiagonal{T,E}

  "pregenerated FFT plan"
  p::FFTPlan{T}

  "pre allocated matrices for FFT"
  a::FFTmats{T}

end

#################
#region Nomal Mul
#################


function timed_mul!(y, a, b; label="mul normal")
  @timeit timecount label begin
    mul!(y, a, b)
  end
end

function timed_mul!(y, a, b, p, q; label="mul normal")
  @timeit timecount label begin
    mul!(y, a, b, p, q)
  end
end

function timed_mul_evg!(y, a, b; label="mul evg")
  @timeit timecount label begin
    mul!(y, a, b)
  end
end
########################
#region Structure: Delay
########################

struct DelayMats{E<:AbstractFloat}

  "count delay update steps"
  i::Vector{Integer}

  "temporary mats"
  v::Matrix{E}
  vᵗ::Matrix{E}

end

######################
#region Structure: ETM
######################

struct MeasureMats{E<:AbstractFloat}

  "tmp: gfunc1, gfunc2"
  g₁::Matrix{E}
  g₂::Matrix{E}


  "energy, CorDD, CorPP, Matrix_pair,condensate fraction, Double occupancy, Histogram of double occupancy, mean_gfunc, equal time magnetic susceptibility"
  E::Vector{E}
  dd::Matrix{E}
  pp::Matrix{E}
  mp::Matrix{E}
  nc::Vector{E}
  p2::Vector{E}
  g_db::Vector{E}
  hist_db::Vector{E}
  g̅::Matrix{E}
  ems::Vector{E}
  chargesqr::Matrix{E}
  charge::Vector{E}
  
end

#######################
#region Structure: UETM
#######################

struct UneqMats{E<:AbstractFloat}

  "gfunc temporary mats"
  g₁::Matrix{E}
  g₂::Matrix{E}
  # g₃::Matrix{E}
  # g₄::Matrix{E}
  # g₅::Matrix{E}

  gt0::Matrix{E}
  g0t::Matrix{E}

  "G(t,t₀) and, G(t₀,t)"
  gt0s::Array{E,3}
  g0ts::Array{E,3}

  "unequal time measurements"
  ut_ss::Vector{E}
  mag_sus::Vector{E}
  gkt0s::Vector{_mattypeC}
  qpw::Vector{E}

end

###########
#region LDR
###########

"""
Calculate the right product ``R := R * K * V``, where  ``R`` is
a [`LDR`](@ref) factorization and ``K V`` is eK and eV matrix .
"""
function rmul_kv!(
  t::Integer, A::Matrix{T}, ldr_ws::LDRWorkspace{T,E},
  fft_ws::FFTWorkspace{H,E}, indices::Array{<:Integer,1},
  evg::Diagonal{T,Vector{T}}
) where {T<:Number,H<:Number,E<:AbstractFloat}

  f_plan, f_diag = fft_ws.p, fft_ws.d
  f_array = fft_ws.a

  # R*eK
  # timed_mul!(ldr_ws.M, A, eK)
  rmulK_block!(ldr_ws.M, A, f_diag.d, f_plan, f_array)
  A .= ldr_ws.M

  # R*eK*eV
  CalculateExpVg!(evg, t, indices)
  rmul!(A, evg)#evg结构是Diagonal，Julia会自动派发到利用对角结构的专用实现

  return nothing
end

"""
Calculate the left product ``L := K * V * L``, where  ``L`` is
a [`LDR`](@ref) factorization and ``K V`` is eK and eV matrix .
"""
function lmul_kv!(
  t::Integer, A::Matrix{T}, ldr_ws::LDRWorkspace{T,E},
  fft_ws::FFTWorkspace{H,E}, indices::Array{<:Integer,1},
  evg::Diagonal{T,Vector{T}}
) where {T<:Number,H<:Number,E<:AbstractFloat}

  f_plan, f_diag = fft_ws.p, fft_ws.d
  f_array = fft_ws.a

  # eV*L
  CalculateExpVg!(evg, t, indices)
  lmul!(evg, A)

  # eK*eV*L
  # timed_mul!(ldr_ws.M, eK, A)
  lmulK_block!(ldr_ws.M, A, f_diag.d, f_plan, f_array)
  A .= ldr_ws.M

  return nothing
end

"""
Calculate the right product ``R := R * V⁻¹ * K⁻¹ ``, where  ``R`` is
a [`LDR`](@ref) factorization and `` V⁻¹ K⁻¹`` is emV and emK matrix .
"""
function rmul_inv_kv!(
  t::Integer, A::Matrix{T}, ldr_ws::LDRWorkspace{T,E},
  fft_ws::FFTWorkspace{H,E}, indices::Array{<:Integer,1},
  emvg::Diagonal{T,Vector{T}}
) where {T<:Number,H<:Number,E<:AbstractFloat}

  f_plan, f_diag = fft_ws.p, fft_ws.d
  f_array = fft_ws.a

  # R*emV
  CalculateExpMVg!(emvg, t, indices)
  rmul!(A, emvg)

  # R*emV*emK
  # timed_mul!(ldr_ws.M, A, eMK)
  rmulK_block!(ldr_ws.M, A, f_diag.d⁻¹, f_plan, f_array)
  A .= ldr_ws.M

  return nothing
end

"""
Calculate the right product ``L := V⁻¹ * K⁻¹ * L``, where  ``L`` is
a [`LDR`](@ref) factorization and `` V⁻¹ K⁻¹`` is emV and emK matrix .
"""
function lmul_inv_kv!(
  t::Integer, A::Matrix{T}, ldr_ws::LDRWorkspace{T,E},
  fft_ws::FFTWorkspace{H,E}, indices::Array{<:Integer,1},
  emvg::Diagonal{T,Vector{T}}
) where {T<:Number,H<:Number,E<:AbstractFloat}

  f_plan, f_diag = fft_ws.p, fft_ws.d
  f_array = fft_ws.a

  # emK*L
  # timed_mul!(ldr_ws.M, eMK, A)
  lmulK_block!(ldr_ws.M, A, f_diag.d⁻¹, f_plan, f_array)
  A .= ldr_ws.M

  # emV*emK*L
  CalculateExpMVg!(emvg, t, indices)
  lmul!(emvg, A)

  return nothing
end


""" LDR factorization
L[D(RKV)]->(LL')D'R'
"""
function LDR_R!(
  F::LDR{T,E}, ws::LDRWorkspace{T,E}
) where {T<:Number,E<:AbstractFloat}

  @timeit timecount label = "LDR" begin
    # record intial Lₐ
    Lᵤ = ws.M′
    copyto!(Lᵤ, F.L)
    # calculate Dᵤ⋅(Rᵤ⋅V)
    Dᵤ = Diagonal(F.d)
    mul!(F.L, Dᵤ, F.R)
    # calculate [L₀⋅D₀⋅R₀] = Dᵤ⋅(Rᵤ⋅V)
    ldr!(F, ws)
    # calculate L₁ = Lᵤ⋅L₀
    mul!(ws.M, Lᵤ, F.L)
    copyto!(F.L, ws.M)
  end

  return nothing
end

""" LDR factorization
[(KVL)D]R->L'D'(R'R)
"""
function LDR_L!(
  F::LDR{T,E}, ws::LDRWorkspace{T,E}
) where {T<:Number,E<:AbstractFloat}

  @timeit timecount label = "LDR" begin
    # record original Rᵥ matrix
    Rᵥ = ws.M′
    copyto!(Rᵥ, F.R)
    # calculate (U⋅Lᵥ)⋅Dᵥ
    Dᵥ = Diagonal(F.d)
    rmul!(F.L, Dᵥ) # (U⋅Lᵥ)⋅Dᵥ

    # calcualte [L₀⋅D₀⋅R₀] = U⋅Lᵥ⋅Dᵥ
    ldr!(F, ws)

    # calcualte R₁ = R₀⋅Rᵥ
    mul!(ws.M, F.R, Rᵥ)
    copyto!(F.R, ws.M)
  end

  return nothing
end

@doc raw"""
    Det_lu!(A::AbstractMatrix{T}, ws::LUWorkspace{T}) where {T}

Note that ``A`` is left modified by this function.
"""
function Det_lu!(A::AbstractMatrix{T}, ws::LUWorkspace{T}) where {T}

  # calculate LU factorization
  LAPACK.getrf!(A, ws)

  return nothing
end

@doc raw"""
    Inv_lu!(A::AbstractMatrix{T}, ws::LUWorkspace{T}) where {T}

Calculate the inverse of the matrix `A`, overwriting `A` in-place.
Also return ``\log(|\det A^{-1}|)`` and ``\textrm{sign}(\det A^{-1}).``
"""
function Inv_lu!(A::AbstractMatrix{T}, ws::LUWorkspace{T}) where {T}

  # calculate LU factorization of A and determinant at the same time
  Det_lu!(A, ws)

  # calculate matrix inverse
  LAPACK.getri!(A, ws)

  return nothing
end

@doc raw"""
    inv_IpA!(G::AbstractMatrix{T}, A::LDR{T,E}, ws::LDRWorkspace{T,E}) where {T,E}

Calculate the numerically stable inverse ``G := [I + A]^{-1},`` where ``G`` is a matrix,
and ``A`` is represented by a [`LDR`](@ref) factorization.

# Algorithm

The numerically stable inverse ``G := [I + A]^{-1}`` is calculated using the procedure
```math
\begin{align*}
G:= & [I+A]^{-1}\\
= & [I+L_{a}D_{a}R_{a}]^{-1}\\
= & [I+L_{a}D_{a,\min}D_{a,\max}R_{a}]^{-1}\\
= & [(R_{a}^{-1}D_{a,\max}^{-1}+L_{a}D_{a,\min})D_{a,\max}R_{a}]^{-1}\\
= & R_{a}^{-1}D_{a,\max}^{-1}[\overset{M}{\overbrace{R_{a}^{-1}D_{a,\max}^{-1}+L_{a}D_{a,\min}}}]^{-1}\\
= & R_{a}^{-1}D_{a,\max}^{-1}M^{-1},
\end{align*}
```
where ``D_{a,\min} = \min(D_a, 1)`` and ``D_{a,\max} = \max(D_a, 1).``
Intermediate matrix inversions and relevant determinant calculations are performed
via LU factorizations with partial pivoting.
"""
function Inv_IpA!(G::AbstractMatrix{T}, A::LDR{T,E}, ws::LDRWorkspace{T,E}) where {T,E}

  @timeit timecount label = "Inv_G" begin
    Lₐ = A.L
    dₐ = A.d
    Rₐ = A.R

    # calculate Rₐ⁻¹
    Rₐ⁻¹ = ws.M′
    copyto!(Rₐ⁻¹, Rₐ)
    Inv_lu!(Rₐ⁻¹, ws.lu_ws)

    # calculate D₋ = min(Dₐ, 1)
    d₋ = ws.v
    @. d₋ = min(dₐ, 1)

    # calculate Lₐ⋅D₋
    D₋ = Diagonal(d₋)
    mul!(ws.M, Lₐ, D₋)

    # calculate D₊ = max(Dₐ, 1)
    d₊ = ws.v
    @. d₊ = max(dₐ, 1)

    # calculate sign(det(D₊)) and log(|det(D₊)|)
    D₊ = Diagonal(d₊)

    # calculate Rₐ⁻¹⋅D₊⁻¹
    Rₐ⁻¹D₊ = Rₐ⁻¹
    rdiv!(Rₐ⁻¹D₊, D₊)

    # calculate M = Rₐ⁻¹⋅D₊⁻¹ + Lₐ⋅D₋
    axpy!(1.0, Rₐ⁻¹D₊, ws.M)

    # calculate M⁻¹ = [Rₐ⁻¹⋅D₊⁻¹ + Lₐ⋅D₋]⁻¹
    M⁻¹ = ws.M
    Inv_lu!(M⁻¹, ws.lu_ws)

    # calculate G = Rₐ⁻¹⋅D₊⁻¹⋅M⁻¹
    mul!(G, Rₐ⁻¹D₊, M⁻¹)
  end

  return nothing
end


@doc raw"""
    inv_IpUV!(G::AbstractMatrix{T}, U::LDR{T,E}, V::LDR{T,E}, ws::LDRWorkspace{T,E}) where {T,E}

Calculate the numerically stable inverse ``G := [I + UV]^{-1},`` where ``G`` is a matrix and
``U`` and ``V`` are represented by [`LDR`](@ref) factorizations.

# Algorithm

The numerically stable inverse ``G := [I + UV]^{-1}`` is calculated using the procedure
```math
\begin{align*}
G:= & [I+UV]^{-1}\\
= & [I+L_{u}D_{u}R_{u}L_{v}D_{v}R_{v}]^{-1}\\
= & [I+L_{u}D_{u,\max}D_{u,\min}R_{u}L_{v}D_{v,\min}D_{v,\max}R_{v}]^{-1}\\
= & [L_{u}D_{u,\max}(D_{u,\max}^{-1}L_{u}^{\dagger}R_{v}^{-1}D_{v,\max}^{-1}+D_{u,\min}R_{u}L_{v}D_{v,\min})D_{v,\max}R_{v}]^{-1}\\
= & R_{v}^{-1}D_{v,\max}^{-1}[\overset{M}{\overbrace{D_{u,\max}^{-1}L_{u}^{\dagger}R_{v}^{-1}D_{v,\max}^{-1}+D_{u,\min}R_{u}L_{v}D_{v,\min}}}]^{-1}D_{u,\max}^{-1}L_{u}^{\dagger}\\
= & R_{v}^{-1}D_{v,\max}^{-1}M^{-1}D_{u,\max}^{-1}L_{u}^{\dagger}
\end{align*}
```
Intermediate matrix inversions and relevant determinant calculations are performed
via LU factorizations with partial pivoting.
"""
function Inv_IpUV!(G::AbstractMatrix{T}, U::LDR{T,E}, V::LDR{T,E}, ws::LDRWorkspace{T,E}) where {T,E}

  @timeit timecount label = "Inv_G" begin
    Lᵤ = U.L
    dᵤ = U.d
    Rᵤ = U.R
    Lᵥ = V.L
    dᵥ = V.d
    Rᵥ = V.R

    # calculate sign(det(Lᵤ)) and log(|det(Lᵤ)|)
    copyto!(ws.M, Lᵤ)
    Det_lu!(ws.M, ws.lu_ws)

    # calculate Rᵥ⁻¹, sign(det(Rᵥ⁻¹)) and log(|det(Rᵥ⁻¹)|)
    Rᵥ⁻¹ = ws.M′
    copyto!(Rᵥ⁻¹, Rᵥ)
    Inv_lu!(Rᵥ⁻¹, ws.lu_ws)

    # calcuate Dᵥ₊ = max(Dᵥ, 1)
    dᵥ₊ = ws.v
    @. dᵥ₊ = max(dᵥ, 1)
    Dᵥ₊ = Diagonal(dᵥ₊)

    # calculate Rᵥ⁻¹⋅Dᵥ₊⁻¹
    rdiv!(Rᵥ⁻¹, Dᵥ₊)
    Rᵥ⁻¹Dᵥ₊⁻¹ = Rᵥ⁻¹

    # calcuate Dᵤ₊ = max(Dᵤ, 1)
    dᵤ₊ = ws.v
    @. dᵤ₊ = max(dᵤ, 1)
    Dᵤ₊ = Diagonal(dᵤ₊)


    # calcualte Dᵤ₊⁻¹⋅Lᵤᵀ
    adjoint!(ws.M, Lᵤ)
    ldiv!(Dᵤ₊, ws.M)
    Dᵤ₊⁻¹Lᵤᵀ = ws.M

    # calculate Dᵤ₋ = min(Dᵤ, 1)
    dᵤ₋ = ws.v
    @. dᵤ₋ = min(dᵤ, 1)
    Dᵤ₋ = Diagonal(dᵤ₋)

    # calculate Dᵤ₋⋅Rᵤ⋅Lᵥ
    mul!(G, Rᵤ, Lᵥ) # Rᵤ⋅Lᵥ
    lmul!(Dᵤ₋, G) # Dᵤ₋⋅Rᵤ⋅Lᵥ

    # calculate Dᵥ₋ = min(Dᵥ, 1)
    dᵥ₋ = ws.v
    @. dᵥ₋ = min(dᵥ, 1)
    Dᵥ₋ = Diagonal(dᵥ₋)

    # caluclate Dᵤ₋⋅Rᵤ⋅Lᵥ⋅Dᵥ₋
    rmul!(G, Dᵥ₋)

    # caluclate Dᵤ₊⁻¹⋅Lᵤᵀ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹
    mul!(ws.M″, Dᵤ₊⁻¹Lᵤᵀ, Rᵥ⁻¹Dᵥ₊⁻¹)

    # calculate M = Dᵤ₊⁻¹⋅Lᵤᵀ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹ + Dᵤ₋⋅Rᵤ⋅Lᵥ⋅Dᵥ₋
    M = G
    axpy!(1.0, ws.M″, M)

    # calculate M⁻¹, sign(det(M)) and log(|det(M)|)
    M⁻¹ = G
    Inv_lu!(M⁻¹, ws.lu_ws)

    # calculate G = Rᵥ⁻¹⋅Dᵥ₊⁻¹⋅M⁻¹⋅Dᵤ₊⁻¹⋅Lᵤᵀ
    mul!(ws.M″, M⁻¹, Dᵤ₊⁻¹Lᵤᵀ) # M⁻¹⋅Dᵤ₊⁻¹⋅Lᵤᵀ
    mul!(G, Rᵥ⁻¹Dᵥ₊⁻¹, ws.M″) # G = Rᵥ⁻¹⋅Dᵥ₊⁻¹⋅M⁻¹⋅Dᵤ₊⁻¹⋅Lᵤᵀ

  end

  return nothing
end

@doc raw"""
    inv_invUpV!(G::AbstractMatrix{T}, U::LDR{T,E}, V::LDR{T,E}, ws::LDRWorkspace{T,E}) where {T,E}

Calculate the numerically stable inverse ``G := [U^{-1}+V]^{-1},`` where ``G`` is a matrix and ``U``
and ``V`` are represented by [`LDR`](@ref) factorizations.
"""
function inv_invUpV!(G::AbstractMatrix{T}, U::LDR{T,E}, V::LDR{T,E}, ws::LDRWorkspace{T,E}) where {T,E}

  @timeit timecount label = "Inv_G_UT" begin
    Lᵤ = U.L
    dᵤ = U.d
    Rᵤ = U.R
    Lᵥ = V.L
    dᵥ = V.d
    Rᵥ = V.R

    # calcualte sign(det(Rᵤ)) and log(|det(Rᵤ)|)
    copyto!(ws.M′, Rᵤ)
    Det_lu!(ws.M′, ws.lu_ws)

    # calculate Dᵤ₋ = min(Dᵤ, 1)
    dᵤ₋ = ws.v
    @. dᵤ₋ = min(dᵤ, 1)
    Dᵤ₋ = Diagonal(dᵤ₋)

    # calculate Dᵤ₋⋅Rᵤ
    Dᵤ₋Rᵤ = ws.M′
    copyto!(Dᵤ₋Rᵤ, Rᵤ)
    lmul!(Dᵤ₋, Dᵤ₋Rᵤ)

    # calculate Rᵥ⁻¹, sign(det(Rᵥ⁻¹)) and log(|det(Rᵥ⁻¹)|)
    Rᵥ⁻¹ = ws.M″
    copyto!(Rᵥ⁻¹, Rᵥ)
    Inv_lu!(Rᵥ⁻¹, ws.lu_ws)

    # calculate Dᵥ₊ = max(Dᵥ, 1)
    dᵥ₊ = ws.v
    @. dᵥ₊ = max(dᵥ, 1)
    Dᵥ₊ = Diagonal(dᵥ₊)

    # calculate Rᵥ⁻¹⋅Dᵥ₊⁻¹
    Rᵥ⁻¹Dᵥ₊⁻¹ = Rᵥ⁻¹
    rdiv!(Rᵥ⁻¹Dᵥ₊⁻¹, Dᵥ₊)

    # calculate Dᵥ₋ = min(Dᵥ, 1)
    dᵥ₋ = ws.v
    @. dᵥ₋ = min(dᵥ, 1)
    Dᵥ₋ = Diagonal(dᵥ₋)

    # calculate Dᵤ₋⋅Rᵤ⋅Lᵥ⋅Dᵥ₋
    mul!(G, Dᵤ₋Rᵤ, Lᵥ) # Dᵤ₋⋅Rᵤ⋅Lᵥ
    rmul!(G, Dᵥ₋) # [Dᵤ₋⋅Rᵤ⋅Lᵥ]⋅Dᵥ₋

    # calculate Dᵤ₊ = max(Dᵤ, 1)
    dᵤ₊ = ws.v
    @. dᵤ₊ = max(dᵤ, 1)
    Dᵤ₊ = Diagonal(dᵤ₊)

    # calculate Dᵤ₊⁻¹⋅Lᵤᵀ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹
    Lᵤᵀ = adjoint(Lᵤ)
    mul!(ws.M, Lᵤᵀ, Rᵥ⁻¹Dᵥ₊⁻¹) # Lᵤᵀ⋅[Rᵥ⁻¹⋅Dᵥ₊⁻¹]
    ldiv!(Dᵤ₊, ws.M) # Dᵤ₊⁻¹⋅[Lᵤᵀ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹]

    # calculate Dᵤ₊⁻¹⋅Lᵤᵀ⋅Rᵥ⁻¹⋅Dᵥ₊⁻¹ + Dᵤ₋⋅Rᵤ⋅Lᵥ⋅Dᵥ₋
    axpy!(1.0, ws.M, G)

    # calculate M⁻¹, sign(det(M⁻¹)) and log(|det(M⁻¹)|)
    M⁻¹ = G
    Inv_lu!(M⁻¹, ws.lu_ws)

    # calculate G := Rᵥ⁻¹⋅Dᵥ₊⁻¹⋅M⁻¹⋅Dᵤ₋⋅Rᵤ
    mul!(ws.M, M⁻¹, Dᵤ₋Rᵤ) # M⁻¹⋅Dᵤ₋⋅Rᵤ
    mul!(G, Rᵥ⁻¹Dᵥ₊⁻¹, ws.M) # G := Rᵥ⁻¹⋅Dᵥ₊⁻¹⋅[M⁻¹⋅Dᵤ₋⋅Rᵤ]

  end

  return nothing
end

###############
#region FFT Mul
###############

"""
对轮换矩阵的第一行做 FFT, 生成轮换矩阵的对角阵
"""
function exp_circulant(c::AbstractVector)
  return fft(c)
end


"""
F = eK * A = FFT * Λ * (iFFT * A)
C = iFFT * A
"""
function liFFT!(C::Matrix{T}, A::Matrix{E},
  V_L_c::Matrix{T}, V_L_c2::Matrix{T}, iplan::AbstractFFTs.ScaledPlan{T}) where {T<:Number,E<:AbstractFloat}

  @views begin
    @inbounds for i in axes(A, 2)
      v_A_view = view(A, :, i)
      v_C_view = view(C, :, i)
      V_3_A_view = reshape(v_A_view, (3, L * L))
      V_3_C_view = reshape(v_C_view, (3, L * L))
      @inbounds for j ∈ 1:3
        V_L_c .= reshape(V_3_A_view[j, :], L, L) # 这里必须使用正常矩阵
        mul!(V_L_c2, iplan, V_L_c)
        @inbounds V_3_C_view[j, :] .= vec(V_L_c2)
      end
    end
  end

  return nothing
end

"""
F = eK * A = FFT * (Λ * iFFT * A)
F = FFT * C2
"""
function lFFT!(F::Matrix{E}, C::Matrix{T},
  V_L_c::Matrix{T}, V_L_c2::Matrix{T}, plan::FFTW.cFFTWPlan{T}) where {T<:Number,E<:AbstractFloat}

  @views begin
    @inbounds for i in axes(C, 2)
      v_C_view = view(C, :, i)
      v_F_view = view(F, :, i)
      V_3_C_view = reshape(v_C_view, (3, L * L))
      V_3_F_view = reshape(v_F_view, (3, L * L))
      @inbounds for j ∈ 1:3
        V_L_c .= reshape(V_3_C_view[j, :], L, L)
        mul!(V_L_c2, plan, V_L_c)
        @inbounds V_3_F_view[j, :] .= real.(vec(V_L_c2))
      end
    end
  end

  return nothing
end

"""
使用 FFT 计算 F = eK * A
  MatC1 = zeros(ComplexF64, Nlayer, Nlayer)
  MatC2 = zeros(ComplexF64, Nlayer, Nlayer)
  V_L_c = zeros(ComplexF64, L, L)
  V_L_c2 = zeros(ComplexF64, L, L)
"""
function lmulK_block!(F::Matrix{E}, A::Matrix{E},
  Λs::Kpμ{T,E}, P::FFTPlan{T}, V::FFTmats{T}) where {T<:Number,E<:AbstractFloat}
  @timeit timecount label = "FFT" begin
    @views begin

      F̃ = P.f
      F̃⁻¹ = P.f⁻¹
      MatC1, MatC2 = V.M₁, V.M₂
      V_L_c, V_L_c2 = P.m₁, P.m₂

      liFFT!(MatC1, A, V_L_c, V_L_c2, F̃⁻¹)
      @timeit timecount label = "FFT2" begin
        mul!(MatC2, Λs.Kd, MatC1)
        lmul!(exp(Λs.t * μμ[1]), MatC2)
      end
      lFFT!(F, MatC2, V_L_c, V_L_c2, F̃)
    end
  end

  return nothing
end


####################################################################################
"""
F = A * eK = (A * FFT) * Λ * iFFT
C = A * FFT
"""
function rFFT!(C::Matrix{T}, A::Matrix{E},
  V_L_c::Matrix{T}, V_L_c2::Matrix{T}, plan::FFTW.cFFTWPlan{T}) where {T<:Number,E<:AbstractFloat}

  @views begin
    @inbounds for i in axes(A, 1)
      v_A_view = view(A, i, :)
      v_C_view = view(C, i, :)
      V_3_A_view = reshape(v_A_view, (3, L * L))
      V_3_C_view = reshape(v_C_view, (3, L * L))
      @inbounds for j ∈ 1:3
        V_L_c .= reshape(V_3_A_view[j, :], L, L) # 这里必须使用正常矩阵

        mul!(V_L_c2, plan, V_L_c)

        @inbounds V_3_C_view[j, :] .= vec(V_L_c2)

      end
    end
  end

  return nothing
end

"""
F = A * eK = (A * FFT * Λ) * iFFT
F = C2 * iFFT
"""
function riFFT!(F::Matrix{E}, C::Matrix{T},
  V_L_c::Matrix{T}, V_L_c2::Matrix{T}, iplan::AbstractFFTs.ScaledPlan{T}) where {T<:Number,E<:AbstractFloat}

  @views begin
    @inbounds for i in axes(C, 1)
      v_C_view = view(C, i, :)
      v_F_view = view(F, i, :)
      V_3_C_view = reshape(v_C_view, (3, L * L))
      V_3_F_view = reshape(v_F_view, (3, L * L))
      @inbounds for j ∈ 1:3
        V_L_c .= reshape(V_3_C_view[j, :], L, L)
        mul!(V_L_c2, iplan, V_L_c)
        @inbounds V_3_F_view[j, :] .= real.(vec(V_L_c2))
      end
    end
  end

  return nothing
end


"""
使用 FFT 计算 F = A * eK
  MatC1 = zeros(ComplexF64, Nlayer, Nlayer)
  MatC2 = zeros(ComplexF64, Nlayer, Nlayer)
  V_L_c = zeros(ComplexF64, L, L)
  V_L_c2 = zeros(ComplexF64, L, L)
"""
function rmulK_block!(F::Matrix{E}, A::Matrix{E},
  Λs::Kpμ{T,E}, P::FFTPlan{T}, V::FFTmats{T}) where {T<:Number,E<:AbstractFloat}
  @timeit timecount label = "FFT" begin
    @views begin

      F̃ = P.f
      F̃⁻¹ = P.f⁻¹
      MatC1, MatC2 = V.M₁, V.M₂
      V_L_c, V_L_c2 = P.m₁, P.m₂

      rFFT!(MatC1, A, V_L_c, V_L_c2, F̃)
      @timeit timecount label = "FFT2" begin
        mul!(MatC2, MatC1, Λs.Kd)
        rmul!(MatC2, exp(Λs.t * μμ[1]))
      end
      riFFT!(F, MatC2, V_L_c, V_L_c2, F̃⁻¹)
    end
  end

  return nothing
end
