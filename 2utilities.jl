include("2multiply.jl")
"""
ldr_workspace与ldrs均来自StableLinearAlgebra包.
Fs创建了Nugroup个LDR对象
"""
function InitializeLDR(
)
  # calculate scratch matrix
  G′ = Matrix{_mattype}(I, Nsite, Nsite)
  # construct vector of LDR factorization to represent B(τ,0) and B(β,τ) matrices
  ldr_ws = ldr_workspace(G′)
  Fs = ldrs(G′, Nugroup)
  return Fs, ldr_ws
end





######搞出eK等的对角形式
"""
Initialize FFT for three identical square-lattice color sectors.
"""
function InitializeFFT_square(cpurank::Integer)

  println("--- Task $cpurank: start FFT steps ---")

  _C64 = ComplexF64

  # Momentum eigenvalues of the periodic single-color hopping matrix.
  # The reshape order matches rFFT! and riFFT!.
  K_row = reshape(_C64.(K_layer[1, :]), L, L) #把lattice.jl中算出来的k_layer第一行拿出来
  εk = vec(fft(K_row))#对K_row做二维fft之后向量化，这就是k_layer的全部本征值

  # Store the exponent -Δτ * εk for one color.
  # K_layer already includes the hopping amplitude and its sign.
  Λk = -Δτ .* εk

  Λ = _create_Kd_square(1.0, Λk)
  Λ⁻¹ = _create_Kd_square(-1.0, Λk)
  Λₕ = _create_Kd_square(0.5, Λk)
  Λₕ⁻¹ = _create_Kd_square(-0.5, Λk)

  DiagMat = FFTDiagonal(Λ, Λ⁻¹, Λₕ, Λₕ⁻¹)#结构体，在multiply中，把这四套数据放在一起，之后用 .d、.d⁻¹、.dₕ、.dₕ⁻¹ 选择

  # Create FFT plans shared by all three colors.
  m₁ = zeros(_C64, L, L)
  m₂ = zeros(_C64, L, L)
  F = plan_fft(m₁, (1, 2))
  F⁻¹ = plan_ifft(m₁, (1, 2))
  Plan = FFTPlan(m₁, m₂, F, F⁻¹)

  # Full three-color matrix workspaces.
  M₁ = zeros(_C64, 3 * Nlayer, 3 * Nlayer)
  M₂ = zeros(_C64, 3 * Nlayer, 3 * Nlayer)
  Array_fft = FFTmats(M₁, M₂)

  println("--- Task $cpurank: finish FFT steps ---")

  return FFTWorkspace(DiagMat, Plan, Array_fft)
end

"""
Repeat the single-color momentum propagator for three color sectors.
"""
function _create_Kd_square(
  t::Float64, Λk::AbstractVector{T},
) where {T<:Complex}

  λ0 = exp.(t .* Λk)#能量本征值构成的向量求exp
  Λ = Diagonal(repeat(λ0, 3))#在对角位置重复三遍

  # Retained for interface compatibility; no chemical-potential factor.
  return Kpμ(Λ, Δτ * t)#Kpμ第一个槽返回3Nlayer维对角阵
end







####算相互作用项相关
#三分量的辅助场是(格点数，时间切片数，色数)
function InitializeIsing()
  # Three HS channels per site and time slice.
  nbond = size(BondGroup, 1)
  isingfield = rand((-1, 1), nbond, Interval, Nflavor)
  #isingfield = ones(Int, nbond, Interval, Nflavor)
  isingfield
end

#一个格点上的3*3HS矩阵,共8种
function InitializeThreeComponentHS(
    λ1::T, λ2::T, λ3::T,
) where {T<:AbstractFloat}

    eVB = Array{T}(undef, 3, 3, 8)
    eMVB = similar(eVB)
    z, o = zero(T), one(T)

    for s1 in (-1, 1), s2 in (-1, 1), s3 in (-1, 1)
        state = 2 * (s1 + 1) + (s2 + 1) + (s3 + 1) ÷ 2 + 1

        a = λ1 * s3
        b = λ2 * s2
        c = λ3 * s1

        R12 = T[cos(c) sin(c) z;
                -sin(c) cos(c) z;
                z z o]

        R13 = T[cos(b) z sin(b);
                z o z;
                -sin(b) z cos(b)]

        R23 = T[o z z;
                z cos(a) sin(a);
                z -sin(a) cos(a)]

        R = R12 * R13 * R23
        eVB[:, :, state] .= R
        eMVB[:, :, state] .= transpose(R)
    end

    return eVB, eMVB
end

#用一个格点上的HS矩阵来构造整体的相互作用矩阵，使用方法：
#CalculateExpV!(eⱽ,  eVB,  IsingField, time)
#CalculateExpV!(e⁻ⱽ, eMVB, IsingField, time)
function CalculateExpV!(
    V::LocalHSMatrix{T},
    table::Array{T,3},
    isingfield::AbstractArray{<:Integer,3},
    time::Integer,
) where {T<:AbstractFloat}

    n = size(isingfield, 1)

    size(V.data) == (3, 3, n) ||
        throw(DimensionMismatch("V.data must have size (3, 3, n)."))

    for i in 1:n
        s1 = isingfield[i, time, 1]
        s2 = isingfield[i, time, 2]
        s3 = isingfield[i, time, 3]

        state = 2 * (s1 + 1) + (s2 + 1) + (s3 + 1) ÷ 2 + 1

        for b in 1:3, a in 1:3
            V.data[a, b, i] = table[a, b, state]
        end
    end

    return nothing
end







"""
初始化LDR或者说UDV矩阵
Fs[3]= B_6B_5
Fs[2]= B_6B_5B_4B_3
Fs[1]= B_6B_5B_4B_3B_2B_1
"""
function CalculateUDV!(
    Fs::AbstractVector{LDR{T,E}},
    ldr_ws::LDRWorkspace{T,E},
    fft_ws::FFTWorkspace{H,E},
    evg::LocalHSMatrix{T},
) where {T<:AbstractFloat,H<:Number,E<:AbstractFloat}

    for t in Interval:-1:1
        n = ceil(Int, t / P0)
        l′ = mod1(t, P0)

        if n < Nugroup && l′ == P0
            copyto!(Fs[n], Fs[n+1])
        end

        B_βτ = Fs[n]::LDR{T,E}

        # Update the right factor: R ← R * eK * eV.
        rmul_kv!(t, B_βτ.R, ldr_ws, fft_ws, evg)

        # Stabilize after completing each time-slice group.
        if l′ == 1
            LDR_R!(B_βτ, ldr_ws)
        end
    end

    return nothing
end








########计算格林函数

function CalculateGreenFunc!(
  time::Integer, direction::Integer,
  greenfunc::Array{T,2}, weightsign::Array{T,1},
  evg::LocalHSMatrix{T}, emvg::LocalHSMatrix{T},
  Fs::AbstractVector{LDR{T,E}},
  ldr_ws::LDRWorkspace{T,E},
  fft_ws::FFTWorkspace{H,E},
  uneq_mats;
  warm::Bool,
) where {T<:AbstractFloat,H<:Number,E<:AbstractFloat}
  @views begin


    if ueqtimeMeasure && !warm
      # Match the identity to the full three-color single-particle space.
      Iₙ = Diagonal(fill(one(T), size(greenfunc, 1)))
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
          rmul_kv!(t, B_β0.R, ldr_ws, fft_ws, evg)
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
          lmul_kv!(t, B_τ0.L, ldr_ws, fft_ws, evg)
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
          lmul_kv!(t, Gτ0, ldr_ws, fft_ws, evg)

          # G(0,l) = G(0,l-1)⋅B⁻¹[l]
          rmul_inv_kv!(t, G0τ, ldr_ws, fft_ws, emvg)

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
          lmul_kv!(t, B_β0.L, ldr_ws, fft_ws, evg)
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
          rmul_kv!(t, B_βτ.R, ldr_ws, fft_ws, evg)
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
          lmul_inv_kv!(t, Gτ0, ldr_ws, fft_ws, emvg)

          # G(0, l-1) = G(0,l)⋅B[l]
          rmul_kv!(t, G0τ, ldr_ws, fft_ws, evg)

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
