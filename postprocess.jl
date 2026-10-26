

"""
	Average and Error over the MPI processes/ Return the mean values and the variance
"""
function MPIAverage(vals::Array{<:Number,N}) where {N}
  vsize = size(vals)
  MPI.Barrier(comm)
  newvals = MPI.Gather(vals, rootcpurank, comm)
  if icpurank == rootcpurank
    newvals = transpose(reshape(newvals, (prod(vsize), ncpurank)))

    mvals = mean(newvals, dims=1)
    vvals = std(newvals, mean=mvals, dims=1)

    mvals = reshape(mvals, vsize)
    vvals = reshape(vvals, vsize)

    return mvals, vvals
  else
    nothing, nothing
  end
end

"""
	Lattice Fourie transformation. Calculate the structure factors from the correlation function.
"""
function LatticeFourie(corfunc::Array{<:Number,2}, kx::Real, ky::Real)
  strucfactor = 0.0 + 0im
  for x0 ∈ 1:L
    for y0 ∈ 1:L
      i = (x0 - 1) * L + y0
      for x1 ∈ 1:L
        for y1 ∈ 1:L
          j = (x1 - 1) * L + y1
          rij = (x0 - x1) * a1 .+ (y0 - y1) * a2
          strucfactor += corfunc[i, j] * exp(1im * (rij[1] * kx + rij[2] * ky)) # exp(ir·k)
        end
      end
    end
  end
  strucfactor ./ Nlayer
end
"""
	Lattice Fourie transformation for sublattice. Calculate the structure factors from the correlation function.
"""
function LatticeFourie_sublattice(corfunc::AbstractMatrix{<:Number}, kx::Real, ky::Real)
  strucfactor = 0.0 + 0im
  for x ∈ 1:L
    for y ∈ 1:L
      for s ∈ 0:2 # sublattice
        i = SiteMap[x, y] + s
        for xx ∈ 1:L
          for yy ∈ 1:L
            for ss ∈ 0:2
              j = SiteMap[xx, yy] + ss
              rij = (x - xx) * a1 .+ (y - yy) * a2 .+ (s - ss) * a3
              strucfactor += corfunc[i, j] * exp(1im * (rij[1] * kx + rij[2] * ky)) # exp(ir·k)
            end
          end
        end
      end
    end
  end
  strucfactor ./ Nlayer
end

"""
	Output data on rootcpurank
"""
function OutputAverage(
  avg_sign::Vector{E}, accept_count::Array{Integer,1},
  energypath::Vector{E},
  signpath::Vector{E},
  m_mats::MeasureMats{E},
  uneq_mats::Union{UneqMats{E},Nothing}
) where {E<:AbstractFloat}
  @timeit timecount label = "postprocess" begin
    energy, cordd, corpp, p2, n_c, hist_db, mgfunc, ems = m_mats.E, m_mats.dd, m_mats.pp, m_mats.p2, m_mats.nc, m_mats.hist_db, m_mats.g̅, m_mats.ems
    chargesqr = m_mats.chargesqr
    charge = m_mats.charge
    if ueqtimeMeasure
      Gt0, G0t = uneq_mats.gt0s, uneq_mats.g0ts
      SS_ut_cor, Mag_sus, Gkt0s, QPW = uneq_mats.ut_ss, uneq_mats.mag_sus, uneq_mats.gkt0s, uneq_mats.qpw
      Gt0_K = zeros(size(Gt0, 3))
      Gt0_0 = zeros(size(Gt0, 3))
      Gt0_c = zeros(size(Gt0, 3))
      Gtau = zeros(size(Gt0, 3))
    end

    nbond = size(BondGroup, 1)
    # average over the Monte simulation and lattice bonds
    oavg_sign = avg_sign ./ (MeasureStep * Interval)
    accept_ratio = zeros(_mattype, 1)
    accept_ratio[1] = accept_count[1] ./ (MeasureStep * Interval * Nlayer)
    tmp_oavg_sign = zeros(_mattype, 1)
    tmp_oavg_sign[1] = abs(real(oavg_sign[1]))
    oenergy = energy ./ (MeasureStep * Interval * (nbond)) ./ Ns
    ohist_db = hist_db ./ sum(hist_db)
    ocordd = cordd ./ (MeasureStep * Interval)
    ocorpp = corpp ./ (MeasureStep * Interval)
    p2 = p2 ./ (MeasureStep * Interval)
    rescaled_p2 = (p2[1] - (ρ / 2)^2) / (ρ / 2 - (ρ / 2)^2)
    # n_c = n_c ./ (MeasureStep * Interval)
    oenergypath = energypath ./ Interval / nbond ./ Ns
    osignpath = signpath ./ Interval
    ogfunc = mgfunc ./ (MeasureStep * Interval)
    ems = ems ./ (MeasureStep * Interval)
    ochargesqr = chargesqr ./ (MeasureStep * Interval) ./ oavg_sign[1]
    ocharge = charge ./ (MeasureStep * Interval) ./ oavg_sign[1]
    # Compute each rank's compressibility from its own Monte Carlo moments.
    ocompressibility = [β / Nlayer * (sum(ochargesqr) - ocharge[1]^2)]

    # unequal time
    if ueqtimeMeasure

      Gt0 ./= MeasureStep
      G0t ./= MeasureStep

      MeasureUnequalTime!(uneq_mats)
    end





    odensity1 = [abs(2 * measure_n(ogfunc) - ρ)]

    # Pair correlation function vs distance r
    # along the diagonal direction Gᴾ(r)
    G_pr = zeros(_mattype, div(L, 2) + 1)
    for i in 1:(div(L, 2)+1)
      G_pr[i] = log(10, ocorpp[1, SiteMap[i, i]])
    end

    # c-CDW at M point
    c_cdw_m1 = sqrt(LatticeFourie(ocordd, m_1...) / Nlayer)
    c_cdw_m2 = sqrt(LatticeFourie(ocordd, m_2...) / Nlayer)
    c_cdw_m3 = sqrt(LatticeFourie(ocordd, m_3...) / Nlayer)
    c_cdw_m = (c_cdw_m1 + c_cdw_m2 + c_cdw_m3) / 3
    # c-CDW at K point
    c_cdw_k = LatticeFourie(ocordd, K_diarc...)
    c_cdw_k = sqrt(c_cdw_k / Nlayer)
    # s-CDW at Γ point
    s_cdw = LatticeFourie(ocordd, 0, 0)
    s_cdw = sqrt(s_cdw / Nlayer)
    # Paring order at Γ point
    pairing2 = LatticeFourie_sublattice(ocorpp, 0, 0)
    pairing = sqrt(pairing2 / Nlayer * 2)

    # momentum space Green's function
    G_k = zeros(_mattype, L, L)
    n_k = zeros(_mattype, L, L)
    M_k = zeros(_mattype, L * L, L * L)
    for i = 1:L
      for j = 1:L
        vec_k = (i - 1) / L * b1 .+ (j - 1) / L * b2
        G_k[i, j] = real(LatticeFourie_sublattice(ogfunc, vec_k...))
        n_k[i, j] = 1.0 - G_k[i, j]
      end
    end
    # for i = 1:L
    #   for j = 1:L
    #     if i == 1
    #       ii = 1
    #     else
    #       ii = L + 2 - i
    #     end
    #     if j == 1
    #       jj = 1
    #     else
    #       jj = L + 2 - j
    #     end
    #     k1 = (i - 1) * L + j # index of K point
    #     k2 = (ii - 1) * L + jj
    #     # M_k[k1, k1] = (1 - G_k[i, j]) * (1 - G_k[ii, jj])
    #     M_k[k1, k2] = (1 - G_k[i, j]) * (1 - G_k[ii, jj])
    #   end
    # end

    # condensate fraction
    # n_c[1] = tr(M_k) / (Nlayer * ρ / 2)

    # Ratio
    # Paring
    pairk = LatticeFourie_sublattice(ocorpp, b1 * 1 / L...) + LatticeFourie_sublattice(ocorpp, b2 * 1 / L...) + LatticeFourie_sublattice(ocorpp, (-b1 * 1 / L - b2 * 1 / L)...) + LatticeFourie_sublattice(ocorpp, -b1 * 1 / L...) + LatticeFourie_sublattice(ocorpp, -b2 * 1 / L...) + LatticeFourie_sublattice(ocorpp, (b1 * 1 / L + b2 * 1 / L)...)
    Rpair = 1.0 - 1 / 6 * pairk / pairing2


    # Histogram

    # results of per core
    # 这里还没有除以平均符号, 只能先假设不存在符号问题了
    # 单独输出之后的平均只能在任务结束之后做了, 因为每个核都可能会崩溃
    Cfile = open("core/core-$icpurank.txt", "w")
    println(Cfile, "# Accepted ratio")
    println(Cfile, "$(real(accept_ratio))"[2:end-1])
    println(Cfile, "# Density 1")
    println(Cfile, "$(real(odensity1))"[2:end-1])
    println(Cfile, "# Energy: kinetic, onsite_U, total")
    println(Cfile, "$(real(oenergy))"[2:end-1])
    println(Cfile, "# Order: c-cdw at M1, M2, M3")
    println(Cfile, "$(real([c_cdw_m1, c_cdw_m2, c_cdw_m3]./ oavg_sign[1]))"[2:end-1])
    println(Cfile, "# Order: c-cdw-m, c-cdw-k, s-cdw, pairing")
    println(Cfile, "$(real([c_cdw_m, c_cdw_k, s_cdw, pairing]./ oavg_sign[1]))"[2:end-1])
    println(Cfile, "# Binder Ratio: Pair")
    println(Cfile, "$(real(Rpair))"[2:end-1])
    # println(Cfile, "# Condensate fraction")
    # println(Cfile, "$(real(n_c))"[2:end-1])
    println(Cfile, "# Double occupancy probability P2")
    println(Cfile, "$(real(p2))"[2:end-1])
    println(Cfile, "# Rescaled P2")
    println(Cfile, "$(real(rescaled_p2))"[2:end-1])
    println(Cfile, "# Equal time magnetic susceptibility")
    println(Cfile, "$(real(ems))"[2:end-1])
    println(Cfile, "# Total particle number")
    println(Cfile, ocharge[1])
    println(Cfile, "# Compressibility")
    println(Cfile, ocompressibility[1])
    flush(Cfile)
    close(Cfile)

    # MPI Average
    # Accepted ratio
    oaccept, accepterr = MPIAverage(accept_ratio)
    # Average sign
    avg_signm, avg_signerr = MPIAverage(tmp_oavg_sign)
    # Energy
    energym, energyerr = MPIAverage(oenergy ./ oavg_sign[1])
    # Double occupancy
    hist_dbm, hist_dberr = MPIAverage(ohist_db ./ oavg_sign[1])
    # Pair correlation function
    oG_pr, G_prerr = MPIAverage(G_pr ./ oavg_sign[1])

    # order parameters (site)
    oc_cdw_ms, oc_cdw_merrs = MPIAverage([c_cdw_m1, c_cdw_m2, c_cdw_m3] ./ oavg_sign[1])
    ocdws, ocdwerrs = MPIAverage([c_cdw_m, c_cdw_k, s_cdw, pairing] ./ oavg_sign[1])

    # Ratio
    oRatio, oRatioerr = MPIAverage([Rpair] ./ oavg_sign[1])

    # Density
    density1m, density1err = MPIAverage(odensity1 ./ oavg_sign[1])

    # green function in K space
    oG_k, oG_kerr = MPIAverage(G_k ./ oavg_sign[1])
    on_k, on_kerr = MPIAverage(n_k ./ oavg_sign[1])

    # condensate fraction
    # on_c, on_c_err = MPIAverage(n_c ./ oavg_sign[1])

    # Double occupancy probability
    op2, op2_err = MPIAverage(p2 ./ oavg_sign[1])
    orescaled_p2, orescaled_p2_err = MPIAverage([rescaled_p2] ./ oavg_sign[1])

    # green function
    ogfuncm, ogfuncerr = MPIAverage(ogfunc ./ oavg_sign[1])

    # Equal time magnetic susceptibility
    oems, oemserr = MPIAverage(ems ./ oavg_sign[1])

    # data vs path
    oepath, oepatherr = MPIAverage(oenergypath ./ oavg_sign[1])
    ospath, ospatherr = MPIAverage(osignpath ./ sign(real(oavg_sign[1])))

    #chargesqr
    ochargesqrm, ochargesqrerr = MPIAverage(ochargesqr)
    ochargem, ochargeerr = MPIAverage(ocharge)
    ocompressibilitym, ocompressibilityerr = MPIAverage(ocompressibility)
    
    # unequal time
    if ueqtimeMeasure
      Gt0 ./= oavg_sign[1]
      for t in axes(Gt0, 3)
        @views Gt0_view = Gt0[:, :, t]
        # Gt0_K[t] = log.(abs(real(LatticeFourie_sublattice(Gt0_view, m_1...))))
        Gt0_K[t] = log.(abs(real(LatticeFourie_sublattice(Gt0_view, K_diarc...))))
        Gt0_0[t] = log.(abs(real(LatticeFourie_sublattice(Gt0_view, 0, 0))))
        Gt0_c[t] = log.(abs(real(tr(Gt0_view)) ./ Nlayer))
        Gtau[t] = tr(Gt0_view) ./ Nlayer
      end
      oGt0_K, oGt0_K_err = MPIAverage(Gt0_K)
      oGt0_0, oGt0_0_err = MPIAverage(Gt0_0)
      oGt0_c, oGt0_c_err = MPIAverage(Gt0_c)
      oGtau, oGtau_err = MPIAverage(Gtau)
      # oG₀, oG₀_err = MPIAverage(G₀)

      oSS_ut_cor, oSS_ut_cor_err = MPIAverage(SS_ut_cor)
      oMag_sus, oMag_sus_err = MPIAverage(Mag_sus)
      oGkt0s, oGkt0s_err = MPIAverage(imag(Gkt0s))
      oQPW, oQPW_err = MPIAverage(QPW)
    end

    if icpurank == rootcpurank
      mean_N = ochargem[1]
      compressibility = ocompressibilitym[1]

      # Log File, output measurements
      Ωfile = open("result.txt", "w")
      # System
      println(Ωfile, "# Lattice")
      println(Ωfile, "L, $L")
      println(Ωfile, "Ns, $Ns")
      println(Ωfile, "U, " * "$U")
      println(Ωfile, "tuner μ, $mutuner")
      println(Ωfile, "μ, $(μμ[1])")
      println(Ωfile, "ρ, $ρ")
      println(Ωfile, "# Monte Carlo")
      println(Ωfile, "warmstep, $WarmStep")
      println(Ωfile, "measurestep, $MeasureStep")
      println(Ωfile, "randomseed, $seed")
      println(Ωfile, "β, $β")
      println(Ωfile, "T, $(1/β)")
      println(Ωfile, "interval, $Interval")
      println(Ωfile, "p0, $P0")
      println(Ωfile, "Δτ, $Δτ")
      println(Ωfile, "ncpu, $ncpurank")
      println(Ωfile, "nd, $nd")

      println(Ωfile, "# Accepted ratio")
      println(Ωfile, "$(real(oaccept))"[2:end-1])
      println(Ωfile, "# Accepted ratio Error")
      println(Ωfile, "$(real(accepterr))"[2:end-1])
      println(Ωfile, "# Density 1")
      println(Ωfile, "$(real(density1m))"[2:end-1])
      println(Ωfile, "# Density 1 Error")
      println(Ωfile, "$(real(density1err))"[2:end-1])
      println(Ωfile, "# Energy: kinetic, onsite_U, total")
      println(Ωfile, "$(real(energym))"[2:end-1])
      println(Ωfile, "# Energy Error")
      println(Ωfile, "$(real(energyerr))"[2:end-1])
      println(Ωfile, "# Order: c-cdw at M1, M2, M3")
      println(Ωfile, "$(real(oc_cdw_ms))"[2:end-1])
      println(Ωfile, "# Order: c-cdw-m, c-cdw-k, s-cdw, pairing")
      println(Ωfile, "$(real(ocdws))"[2:end-1])
      println(Ωfile, "# Order Error c-cdw-m, c-cdw-k, s-cdw, pairing")
      println(Ωfile, "$(real(ocdwerrs))"[2:end-1])
      println(Ωfile, "# Binder Ratio: Pair")
      println(Ωfile, "$(real(oRatio))"[2:end-1])
      println(Ωfile, "# Binder Ratio Error")
      println(Ωfile, "$(real(oRatioerr))"[2:end-1])
      if ueqtimeMeasure
        println(Ωfile, "# Unequal time magnetic susceptibility")
        println(Ωfile, "$(real(oMag_sus))"[2:end-1])
        println(Ωfile, "# Susceptibility Error")
        println(Ωfile, "$(real(oMag_sus_err))"[2:end-1])
        println(Ωfile, "# Quasi-particle weight")
        println(Ωfile, "$(real(oQPW))"[2:end-1])
        println(Ωfile, "# Quasi-particle weight Error")
        println(Ωfile, "$(real(oQPW_err))"[2:end-1])
      end


      println(Ωfile, "#Average minus sign")
      println(Ωfile, "$(real(avg_signm))"[2:end-1])
      println(Ωfile, "#Average minus sign Error")
      println(Ωfile, "$(real(avg_signerr))"[2:end-1])

      # println(Ωfile, "#Condensate fraction")
      # println(Ωfile, "$(real(on_c))"[2:end-1])
      # println(Ωfile, "#Condensate fraction Error")
      # println(Ωfile, "$(real(on_c_err))"[2:end-1])

      println(Ωfile, "#Double occupancy probability")
      println(Ωfile, "$(real(op2))"[2:end-1])
      println(Ωfile, "#Double occupancy probability Error")
      println(Ωfile, "$(real(op2_err))"[2:end-1])
      println(Ωfile, "# Rescaled P2")
      println(Ωfile, "$(real(orescaled_p2))"[2:end-1])
      println(Ωfile, "# Rescaled P2 Error")
      println(Ωfile, "$(real(orescaled_p2_err))"[2:end-1])
      println(Ωfile, "# Equal time magnetic susceptibility")
      println(Ωfile, "$(real(oems))"[2:end-1])
      println(Ωfile, "# Equal time magnetic susceptibility Error")
      println(Ωfile, "$(real(oemserr))"[2:end-1])
      println(Ωfile, "# Mean particle number per site")
      println(Ωfile, ochargem[1] / Nlayer)
      println(Ωfile, "# Mean particle number per site Error")
      println(Ωfile, ochargeerr[1] / Nlayer)
      println(Ωfile, "# Compressibility")
      println(Ωfile, compressibility)
      println(Ωfile, "# Compressibility Error")
      println(Ωfile, ocompressibilityerr[1])
      flush(Ωfile)
      close(Ωfile)

      # # Correlation File. Greenfunc, bond, site
      JLD2.save("data.jld2", Dict("G_mean" => ogfuncm, "G_mean_err" => ogfuncerr, "chargesqr" => ochargesqrm, "chargesqr_err" => ochargesqrerr, "charge" => ochargem, "charge_err" => ochargeerr, "C_pair_r" => oG_pr, "C_pair_r_err" => G_prerr, "Hist_db" => hist_dbm, "Hist_db_err" => hist_dberr, "n(K)" => on_k, "n(K)err" => on_kerr))

      # unequal time matrix
      if ueqtimeMeasure == true
        JLD2.save("dataUEQ.jld2", Dict("Gt0_K" => oGt0_K, "Gt0_K_err" => oGt0_K_err, "Gt0_0" => oGt0_0, "Gt0_0_err" => oGt0_0_err, "Gt0_c" => oGt0_c, "Gt0_c_err" => oGt0_c_err,"Gtau" => oGtau, "Gtau_err" => oGtau_err, "SS_ut_cor" => oSS_ut_cor, "SS_ut_cor_err" => oSS_ut_cor_err, "Gkt0s" => oGkt0s, "Gkt0s_err" => oGkt0s_err))
      end

      #data vs path
      # EPfile = open("epath.txt", "w")
      # for i ∈ eachindex(oepath)
      #   @printf(EPfile, "%i, %16.10f, %16.10f, %16.10f, %16.10f\n", i,
      #     real(oepath[i]), imag(oepath[i]), real(oepatherr[i]), imag(oepatherr[i]))
      # end
      # flush(EPfile)
      # close(EPfile)
      # SPfile = open("spath.txt", "w")
      # for i ∈ eachindex(ospath)
      #   @printf(SPfile, "%i, %16.10f, %16.10f, %16.10f, %16.10f\n", i,
      #     real(ospath[i]), imag(ospath[i]), real(ospatherr[i]), imag(ospatherr[i]))
      # end
      # flush(SPfile)
      # close(SPfile)

    else
      nothing
    end
  end

  return nothing
end
