

function MeasureEqualTime!(
  gfunc::Matrix{E},
  weightsign::Vector{E},
  avg_sign::Vector{E},
  m_mats::MeasureMats{E},
  fft_ws::FFTWorkspace{H,E},
) where {H<:Number,E<:AbstractFloat}

  f_plan, f_diag = fft_ws.p, fft_ws.d
  f_array = fft_ws.a
  dgfunc, dgfunc2 = m_mats.g₁, m_mats.g₂
  energy, cordd, corpp, mat_pair, p2, n_c, g_db, hist_db, mgfunc, ems = m_mats.E, m_mats.dd, m_mats.pp, m_mats.mp, m_mats.p2, m_mats.nc, m_mats.g_db, m_mats.hist_db, m_mats.g̅, m_mats.ems
  chargesqr = m_mats.chargesqr
  charge = m_mats.charge

  weight_sign = weightsign[1]
  avg_sign[1] += weight_sign

  # second-order Trotter. exp(K)exp(V) -> exp(K/2)exp(V)exp(K/2)
  # timed_mul!(dgfunc, eMKHalf, gfunc)
  lmulK_block!(dgfunc, gfunc, f_diag.dₕ⁻¹, f_plan, f_array)
  # timed_mul!(dgfunc2, dgfunc, eKHalf)
  rmulK_block!(dgfunc2, dgfunc, f_diag.dₕ, f_plan, f_array)
  # measure
  mgfunc .+= dgfunc2 .* weight_sign

  @timeit timecount label = "m_E" begin
    MeasureEnergy!(dgfunc2, weight_sign, energy)
  end
  @timeit timecount label = "m_Double" begin
    MeasureDouble!(dgfunc2, weight_sign, p2, g_db, hist_db)
  end
  @timeit timecount label = "m_Pair" begin
    MeasurePair!(dgfunc2, weight_sign, corpp, mat_pair)
  end
  @timeit timecount label = "m_CDW" begin
    MeasureCDW!(dgfunc2, weight_sign, cordd)
  end
  @timeit timecount label = "m_EMS" begin
    MeasureEMS!(dgfunc2, weight_sign, dgfunc, ems)
  end
  @timeit timecount label = "m_chargesqr" begin
    Measurechargesqr!(dgfunc2, weight_sign, chargesqr)
  end
  @timeit timecount label = "m_charge" begin
    Measurecharge!(dgfunc2, weight_sign, charge)
  end

  return nothing
end

"""
unequal time, forward update direction
B(τ,τ-1)   = exp(K/2)exp(V)exp(K/2)
B(τ,τ-1)⁻¹ = exp(-K/2)exp(-V)exp(-K/2)
G(τ,0)    = ⟨c(τ)c⁺⟩  = B(τ,τ-1)G(τ-1,0)
G(0,τ)[i,j]    = -⟨cⱼ⁺(τ)cᵢ⟩ = G(0,τ-1)B(τ,τ-1)⁻¹
"""
function MeasureUnequalTime!(
  uneq_mats::UneqMats{E},
) where {E<:AbstractFloat}

  Gₜₘ = uneq_mats.g₁
  Gt0s, G0ts = uneq_mats.gt0s, uneq_mats.g0ts
  SS_ut_cor, Mag_sus, GKt0s, QPW = uneq_mats.ut_ss, uneq_mats.mag_sus, uneq_mats.gkt0s, uneq_mats.qpw

  MeasureUtMagSus!(Gt0s, G0ts, SS_ut_cor, Mag_sus, Gₜₘ)
  # MeasureUtQPW!(QPW, Gt0s, GKt0s, m_1)
  # MeasureUtQPW!(QPW, Gt0s, GKt0s, m_2)
  # MeasureUtQPW!(QPW, Gt0s, GKt0s, m_3)
  # QPW[1] /= 3.0
  MeasureUtQPW!(QPW, Gt0s, GKt0s, K_diarc)
  MeasureUtQPW!(QPW, Gt0s, GKt0s, K_diarc2)
  QPW[1] /= 2.0

  return nothing
end


# --- basic measurement functions ---
"""
	Ground state energy: kinetic energy, interaction parts
"""
function MeasureEnergy!(
  gfunc::Array{_mattype,2}, weight_sign::_mattype,
  energy::Array{_mattype,1},
)

  nbond = size(BondGroup, 1)
  # Kinetic energy
  for i ∈ 1:Nlayer
    for j ∈ 1:Nlayer
      energy[1] += -gfunc[j, i] * K[i, j] * weight_sign * 2
    end
  end

  # Interaction energy
  for i ∈ 1:Nlayer
    wickg = (1 - gfunc[i, i])^2
    energy[2] += U * weight_sign * wickg
  end


  #Total energy
  energy[3] = energy[1] + energy[2]

  return nothing
end

"""
	Histogram of double occupancy probability
"""
function MeasureDouble!(
  gfunc::Array{_mattype,2}, weight_sign::_mattype,
  p2::Array{_mattype,1}, G_db::Array{_mattype,1}, hist_db::Array{_mattype,1},
)
  for i in 1:Nlayer
    G_db[i] = (1 - gfunc[i, i])^2 * weight_sign
  end
  p2[1] += 1 / Nlayer * sum(G_db)
  bins = 0:0.05:1  # 分箱的边界
  hist_db .+= [sum(G_db .>= bins[i] .&& G_db .< bins[i+1]) for i in 1:length(bins)-1]

  return nothing
end


function MeasurePair!(
  gfunc::Array{_mattype,2}, weight_sign::_mattype,
  corpp::Array{_mattype,2}, mat_pair::Array{_mattype,2}
)
  for i ∈ 1:Nlayer
    for j ∈ 1:Nlayer
      # pair-pair correlation
      mat_pair[i, j] = Pair!(gfunc, weight_sign, i, j)
      corpp[i, j] += mat_pair[i, j]
    end
  end
  return nothing
end

"""Accumulate total particle number; Ns=2 includes both spin components."""
function Measurecharge!(
  gfunc::Matrix{E}, weight_sign::E,
  charge::Vector{E}
) where {E<:AbstractFloat}
  n_total = zero(E)
  for i ∈ 1:Nlayer
    n_total += Ns * (1 - gfunc[i, i])
  end
  charge[1] += n_total * weight_sign
  return nothing
end

function Measurechargesqr!(
  gfunc::Matrix{E}, weight_sign::E,
  chargesqr::Matrix{E}
) where {E<:AbstractFloat}
  for i ∈ 1:Nlayer
    for j ∈ 1:Nlayer
      chargesqr[i, j] += CDW!(gfunc, i, j) * weight_sign
    end
  end
  return nothing
end

function MeasureCDW!(
  gfunc::Matrix{E}, weight_sign::E,
  cordd::Matrix{E}
) where {E<:AbstractFloat}
  # density-density correlation
  cdwval = 0.0 + 0.0im
  @inbounds for x0 ∈ 1:L
    @inbounds for y0 ∈ 1:L
      # @timeit timecount label = "m_CDW1" begin
      i_id1 = SiteMap[x0, y0] # a
      i_id2 = i_id1 + 1 # b
      i_id3 = i_id1 + 2 # c
      i = (x0 - 1) * L + y0
      # end
      @inbounds for x1 ∈ 1:L
        @inbounds for y1 ∈ 1:L
          # @timeit timecount label = "m_CDW2" begin
          j_id1 = SiteMap[x1, y1] # a
          j_id2 = j_id1 + 1 # b
          j_id3 = j_id1 + 2 # c
          j = (x1 - 1) * L + y1
          # end
          cdwval = 0.0 + 0.0im
          # bond ordered wave
          # @timeit timecount label = "m_CDW3" begin
          cdwval += CDW!(gfunc, i_id1, j_id1)  # aa
          cdwval -= CDW!(gfunc, i_id1, j_id2) * 0.5  #ab
          cdwval -= CDW!(gfunc, i_id1, j_id3) * 0.5  #ac
          cdwval -= CDW!(gfunc, i_id2, j_id1) * 0.5  # ba
          cdwval += CDW!(gfunc, i_id2, j_id2)  # bb
          cdwval -= CDW!(gfunc, i_id2, j_id3) * 0.5  # bc
          cdwval -= CDW!(gfunc, i_id3, j_id1) * 0.5 # ca
          cdwval -= CDW!(gfunc, i_id3, j_id2) * 0.5  # cb
          cdwval += CDW!(gfunc, i_id3, j_id3)  # cc
          cordd[i, j] += 3 / 4 * cdwval * weight_sign
          # end
        end
      end
    end
  end

  return nothing
end

"""
Measure equal time uniform magnetic susceptibility:
Χₛ = 2β/3N ∑ᵢⱼ⟨Sᵢ⋅Sⱼ⟩
"""
function MeasureEMS!(
  gfunc::Matrix{E}, weight_sign::E,
  gₜₘ::Matrix{E}, ems::Vector{E},
) where {E<:AbstractFloat}

  @views begin
    # ∑⟨S⋅S⟩ = Tr[(I-G)G]
    timed_mul!(gₜₘ, gfunc, gfunc, -1.0, 0.0)
    gₜₘ .+= gfunc
    # 1/N * ∑⟨S(τ)⋅S(0)⟩ = -1/2N * Tr[G(τ,0)G(0,τ)]
    ems[1] += 3 * β / 2 / Nlayer * tr(gₜₘ) * weight_sign
  end

  return nothing
end

function Pair!(
  G::Array{_mattype,2}, W::_mattype, i::Integer, j::Integer,
)
  return 0.5 * ((G[i, j]^2 + (G[j, i] - δ(i, j))^2)) * W
end

function CDW!(
  G::Array{_mattype,2}, i::Integer, j::Integer,
)
  return Ns^2 * (1 - G[i, i]) * (1 - G[j, j]) + Ns * (δ(i, j) - G[j, i]) * G[i, j]
end

"""
Measure unequal time uniform magnetic susceptibility:
Χₛ = 1/N ∑ᵢⱼ∫dτ⟨Sᵢ(τ)⋅Sⱼ(0)⟩
"""
function MeasureUtMagSus!(
  gt0s::Array{E,3}, g0ts::Array{E,3},
  ss_ut_cor::Vector{E}, mag_sus::Vector{E},
  gₜₘ::Matrix{E}
) where {E<:AbstractFloat}

  @views begin

    for i in axes(gt0s, 3)
      timed_mul!(gₜₘ, gt0s[:, :, i], g0ts[:, :, i])
      # 1/N * ∑⟨S(τ)⋅S(0)⟩ = -1/2N * Tr[G(τ,0)G(0,τ)]
      ss_ut_cor[i] = -1 / (2 * Nlayer) * tr(gₜₘ)
    end
    mag_sus[1] = Rm(ss_ut_cor)
  end

  return nothing
end

"""
Calculate quasi-particle weight
"""
function MeasureUtQPW!(qpw::Vector{E}, gt0s::Array{E,3}, gkt0s::Vector{_mattypeC}, kf::Vector{E}) where {E<:AbstractFloat}


  @views begin

    for i in axes(gt0s, 3)
      gkt0s[i] = exp(1im * pi / β * (i - 1) * Δτ) * LatticeFourie_sublattice(gt0s[:, :, i], kf...)
    end
    qpw[1] += -inv(1.0 - imag(1im * pi / β - inv(Rm(gkt0s))) / (pi / β))

  end

  return nothing
end

"""
equal-time <c'c c'c>
"""
function wick(gfunc::Array{_mattype,2}, a::Integer, b::Integer, c::Integer, d::Integer)
  return Ns^2 * (δ(a, b) - gfunc[b, a]) * (δ(c, d) - gfunc[d, c]) + Ns * gfunc[b, c] * (δ(a, d) - gfunc[d, a])
end
"""
equal-time <c'c' cc>
"""
function wickpp(gfunc::Array{_mattype,2}, a::Integer, b::Integer, c::Integer, d::Integer)
  return Ns * (δ(a, d) - gfunc[d, a]) * (δ(b, c) - gfunc[c, b]) - Ns * (δ(a, c) - gfunc[c, a]) * (δ(b, d) - gfunc[d, b])
end

function δ(a::Integer, b::Integer)
  d = a == b ? 1 : 0
  d
end

function Rz(d::Float64, f1::E, f2::E) where {E<:Number}
  return 0.5 * d * (f1 + f2)
end


function Rm(matrix::AbstractVector{E}) where {E<:Number}
  n = length(matrix) - 1
  Rm = 0.0
  for i = 1:n
    Rm += Rz(Δτ, matrix[i], matrix[i+1])
  end
  return Rm
end
