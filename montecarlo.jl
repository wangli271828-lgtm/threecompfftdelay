
# --- Monte Carlo procedure ---
"""
	Update in the B> part
"""
function UpdateIsingKet!(
  direction::Integer, step::Integer,
  isingfield::Array{<:Integer,2},
  greenfunc::Array{T,2}, weightsign::Array{T,1},
  accept_count::Array{Integer,1}, avg_sign::Array{T,1},
  evg::Diagonal{T,Vector{T}}, emvg::Diagonal{T,Vector{T}},
  indices::Array{<:Integer,1},
  d_mats::DelayMats{E},
  m_mats::MeasureMats{E},
  uneq_mats::Union{UneqMats{E},Nothing},
  Fs::AbstractVector{LDR{T,E}}, ldr_ws::LDRWorkspace{T,E},
  chemical_potential_tuner::MuTunerLogger{E,T},
  fft_ws::FFTWorkspace{H,E};
  warm::Bool,
) where {T<:Number,H<:Number,E<:AbstractFloat}

  dgfunc = m_mats.g₁
  logerr = -100

  if direction == +1
    ta, tb, tc = 1, Interval, 1
  elseif direction == -1
    ta, tb, tc = Interval, 1, Interval
  else
    println("Wrong Direction")
  end
  for t ∈ ta:direction:tb

    # re-calculate the Green function of the last step from scratch
    if abs(t - tc) % P0 == 0
      dgfunc .= greenfunc
      direction == +1 && CalculateGreenFunc!(t - 1, direction, indices, greenfunc, weightsign, evg, emvg, Fs, ldr_ws, fft_ws, uneq_mats; warm=warm)
      direction == -1 && CalculateGreenFunc!(t, direction, indices, greenfunc, weightsign, evg, emvg, Fs, ldr_ws, fft_ws, uneq_mats; warm=warm)
      # record fast update error
      logerr = _logmax(logerr, dgfunc, greenfunc)
    end

    # record the Green function to an array
    if !warm
      MeasureEqualTime!(greenfunc, weightsign, avg_sign, m_mats, fft_ws)
    end

    # update the HS field on the bond
    UpdateBond!(t, direction, isingfield, greenfunc, weightsign, accept_count, evg, emvg, indices, dgfunc, d_mats, fft_ws; warm=warm)

  end

  if mutuner && warm && step <= round(Integer, WarmStep / 2)
    update_chemical_potential!(greenfunc, step=step, chemical_potential_tuner=chemical_potential_tuner)
  end

  icpurank == rootcpurank && println(" fast update error: ", logerr)

  return nothing
end



"""
   Go through the lattice bonds.
   0->beta, then beta->0
"""
function UpdateBond!(
  time::Integer, direction::Integer,
  isingfield::Matrix{<:Integer},
  greenfunc::Matrix{E},
  weightsign::Vector{E}, accept_count::Vector{Integer},
  evg::Diagonal{E,Vector{E}}, emvg::Diagonal{E,Vector{E}},
  indices::Vector{<:Integer},
  dgfunc::Matrix{E},
  d_mats::DelayMats{E},
  fft_ws::FFTWorkspace{H,E};
  warm::Bool,
) where {H<:Number,E<:AbstractFloat}

  idv, vec1, vecT1 = d_mats.i, d_mats.v, d_mats.vᵗ

  # Calculate eVg and eMVg from scratch
  CalculateExpVg!(evg, time, indices)
  CalculateExpMVg!(emvg, time, indices)

  WarpUpdateR2L!(direction, greenfunc, evg, emvg, dgfunc, fft_ws)

  @timeit timecount label = "delay_update" begin
    for bi ∈ axes(BondGroup, 1)

      i, j = BondGroup[bi]
      indices[1] = i
      oldist = isingfield[bi, time] # ising state
      #isingbit = digits(ist-1, base=4, pad=4) .+ 1
      newist = oldist
      while true
        newist = rand(1:NIsingState)
        newist != oldist && break
      end

      detR = CalculateMetroplis(idv[1], oldist, newist, indices, greenfunc, vec1, vecT1)
      detR = detR^Ns

      metroplis = (exp(-η[newist]) / exp(-η[oldist]))^(Ns / 2) * γ[newist] / γ[oldist] * detR # 吸引势中没有 i


      #metroplis = min(1, γ[newist]/γ[oldist] * real(detR))

      # --- update ---
      if icpurank == rootcpurank
        if imag(metroplis) > 1e-4
          println("Task $icpurank, complex sign: ", imag(metroplis))
        end
        if real(metroplis) < 0
          println("Task $icpurank, negative sign: ", real(metroplis))
        end
      end
      if abs(metroplis) >= rand()
        if !warm
          accept_count[1] += 1
        end
        if real(metroplis) < 0
          weightsign[1] *= -1#为啥是"*=-1"组会有过讨论
        end
        #println("accepted metroplis:", metroplis)
        isingfield[bi, time] = newist
        UpdateExpV!(newist, indices, evg, emvg)
        idv[1] += 1
        UpdateLocal!(idv[1], oldist, newist, indices, greenfunc, vec1, vecT1)
      end
      if bi == size(BondGroup, 1) || idv[1] == nd # 在site update结束时也要做local更新
        UpdateDelay!(idv[1], greenfunc, vec1, vecT1)
        idv[1] = 0 # vec id 归零
      end

    end
  end
  #println("imaginary part of accepted metroplis:", weightsign[1])
  WarpUpdateL2R!(direction, greenfunc, evg, emvg, dgfunc, fft_ws)

  return nothing
end



# --- Monte Carlo update ---
"""
	Calculating the metropolis probability by
	input: oldisingstate, newisingstate, and the indices of [i↑i↓j↑j↓]
	output: probability #prob .ie. det[1+Δ(1-G)]
	output: and inverse of matrix (1+Δ(1-G)) to be #prob_inv
"""
function CalculateMetroplis(
  id::Integer,# 标记vec, vecT的序号
  oldist::Integer,
  newist::Integer,
  indices::Vector{<:Integer},
  greenfunc::Matrix{E},
  vec1::Matrix{E},
  vecT1::Matrix{E},
) where {E<:AbstractFloat}
  i1 = indices[1]
  gi1 = greenfunc[i1, i1]
  if id >= 1
    for j in 1:id
      gi1 -= vec1[i1, j] * vecT1[j, i1]
    end
  end
  Δ1 = IsingΔ[newist, oldist] # CalculateIsing(oldist, newist) #
  return (1 + (1 - gi1) * Δ1)
  # return det(I + IsingΔ[:, :, newist, oldist] .* (I - gibtmp)) ## 去掉了中间量 Δ, R

end

"""
	Update the eVg and eMVg array after the update of Ising field
"""
function UpdateExpV!(
  newist::Integer,
  indices::Vector{<:Integer},
  evg::Diagonal{E,Vector{E}}, emvg::Diagonal{E,Vector{E}},
) where {E<:AbstractFloat}
  @views begin
    # exp(Vb)
    evg[indices[1], indices[1]] = eVB[newist]
    # exp(-Vb)
    emvg[indices[1], indices[1]] = eMVB[newist]
  end

  return nothing
end

"""
	Update the Green function G(τ)
	G = G - G*RD*(1-G)
"""
function UpdateLocal!(
  id::Integer,# 标记vec, vecT的序号
  oldist::Integer,
  newist::Integer,
  indices::Vector{<:Integer},
  greenfunc::Matrix{E},
  vec1::Matrix{E},
  vecT1::Matrix{E},
) where {E<:AbstractFloat}
  i1 = indices[1]
  @views begin

    for k in 1:Nlayer
      vec1[k, id] = greenfunc[k, i1]
      vecT1[id, k] = δf(i1, k) - greenfunc[i1, k]
      if id > 1
        for j in 1:id-1
          vec1[k, id] -= vec1[k, j] * vecT1[j, i1]
          vecT1[id, k] += vec1[i1, j] * vecT1[j, k]
        end
      end
    end
    gi1 = vec1[i1, id]
    Δ1 = IsingΔ[newist, oldist] # CalculateIsing(oldist, newist) #
    RΔ1 = inv(1 + (1 - gi1) * Δ1) * Δ1
    vecT1[id, :] .*= RΔ1
  end


  return nothing
end


"""
	Delayed update
	G = G0 - Σvec*vecT
"""
function UpdateDelay!(
  id::Integer,
  greenfunc::Matrix{E},
  vec1::Matrix{E},
  vecT1::Matrix{E},
) where {E<:AbstractFloat}
  @views begin
    mul!(greenfunc, vec1[:, 1:id], vecT1[1:id, :], -1.0, 1.0)
  end

  return nothing
end

"""
	Update the Green function G(τ) to G(τ+Δτ) where τ=nΔτ.
	Because in this problem the forward-backward schedule must be treated differently, we write one functions for each other. When adopting the forward schedule, the Green function G=1-B>(B<B>)^{-1}B< must be multiplied by exp(Vb) on the left hand side and exp(-Vb) on the right-hand side where Vb is one kind of lattice bond. In this way, we would be able to have the relation B>′=(Δ+1)B> when performing MC update. Additionally, the interaction part consists of exp(K)exp(V).
"""
function WarpUpdateR2L!(
  direction::Integer,
  greenfunc::Matrix{E},
  evg::Diagonal{E,Vector{E}}, emvg::Diagonal{E,Vector{E}},
  dgfunc::Matrix{E},
  fft_ws::FFTWorkspace{H,E}
) where {H<:Number,E<:AbstractFloat}
  f_plan, f_diag = fft_ws.p, fft_ws.d
  f_array = fft_ws.a
  @views begin
    if direction == +1
      # GreenFunc = evg * GreenFunc * emvg. (B<V)B> -> B<(VB>)
      timed_mul_evg!(dgfunc, evg, greenfunc)
      timed_mul_evg!(greenfunc, dgfunc, emvg)
    elseif direction == -1  # L2R
      # GreenFunc = eMK * GreenFunc * eK (given that time is not 1)  B<(KB>) -> (B<K)B>
      # timed_mul!(dgfunc, eMK, greenfunc)
      lmulK_block!(dgfunc, greenfunc, f_diag.d⁻¹, f_plan, f_array)
      # timed_mul!(greenfunc, dgfunc, eK)
      rmulK_block!(greenfunc, dgfunc, f_diag.d, f_plan, f_array)
    else
      println("Wrong Direction")
    end
  end

  return nothing
end

function WarpUpdateL2R!(
  direction::Integer,
  greenfunc::Matrix{E},
  evg::Diagonal{E,Vector{E}}, emvg::Diagonal{E,Vector{E}},
  dgfunc::Matrix{E},
  fft_ws::FFTWorkspace{H,E}
) where {H<:Number,E<:AbstractFloat}
  f_plan, f_diag = fft_ws.p, fft_ws.d
  f_array = fft_ws.a
  @views begin
    if direction == +1 # R2L
      # GreenFunc = eK * GreenFunc * eMK. (B<K)B> -> B<(KB>)
      # timed_mul!(dgfunc, eK, greenfunc)
      lmulK_block!(dgfunc, greenfunc, f_diag.d, f_plan, f_array)
      # timed_mul!(greenfunc, dgfunc, eMK)
      rmulK_block!(greenfunc, dgfunc, f_diag.d⁻¹, f_plan, f_array)
    elseif direction == -1
      timed_mul_evg!(dgfunc, emvg, greenfunc)
      timed_mul_evg!(greenfunc, dgfunc, evg)
    else
      println("Wrong Direction")
    end
  end

  return nothing
end

"""
  Second-order Trotter, exp(K)exp(V) -> exp(K/2)exp(V)exp(K/2).
  G =  exp(-K/2) * G * exp(K/2).
  For equal and unequal time Greens' functions.
"""
function Trotter2nd!(
  F::AbstractMatrix{T}, A::AbstractMatrix{T},
  fft_ws::FFTWorkspace{H,E}, M::Matrix{T},
) where {T<:Number,H<:Number,E<:AbstractFloat}
  f_plan, f_diag = fft_ws.p, fft_ws.d
  f_array = fft_ws.a

  # A =  exp(-K/2) * A * exp(K/2)
  lmulK_block!(M, A, f_diag.dₕ⁻¹, f_plan, f_array)
  rmulK_block!(F, M, f_diag.dₕ, f_plan, f_array)

  return nothing
end


function _logmax(logerr::Integer, dgfunc::Matrix{E}, gfunc::Matrix{E}) where {E<:AbstractFloat}
  @timeit timecount label = "fast_err" begin
    @fastmath begin
      log10_inv = 1 / log(10)
      return max(logerr, ceil(Integer, log(maximum(abs.(dgfunc .- gfunc))) * log10_inv))
    end
  end
end

function δf(a::Integer, b::Integer)
  d = a == b ? 1.0 + 0im : 0im
  d
end
