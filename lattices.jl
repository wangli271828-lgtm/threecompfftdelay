function MakeKagome()
	nlayer = 3 * L * L
	# idx: sublattice type
	xy2siteidx(idx::Integer, x::Integer, y::Integer) = idx + (y - 1) * 3 + (x - 1) * 3 * L
	# _mattype = Float64
	sitemap = zeros(Integer, L, L)
	hopping = zeros(_mattype, nlayer, nlayer)#nnhopping
	bg1 = Array{Tuple{Integer, Integer}, 1}()#只保留一个bond用来等效成Hubbard U
	for x ∈ 1:L
		for y ∈ 1:L
			i = xy2siteidx(1, x, y)
			sitemap[x, y] = i
			j = xy2siteidx(2, x, y)
			k = xy2siteidx(3, x, y)
			bg1 = [bg1; (i, i); (j, j); (k, k)]
			""" NN hopping """
			# hopping in the same Bravais lattice between A, B and C
			hopping[i, j] = -t0
			hopping[j, i] = -t0
			hopping[i, k] = -t0
			hopping[k, i] = -t0
			hopping[k, j] = -t0
			hopping[j, k] = -t0
			# hopping between different Bravais lattice in L
			# periodic boundary condition
			yy = y == 1 ? L : y - 1
			jj = xy2siteidx(2, x, yy)
			hopping[i, jj] = -t0
			hopping[jj, i] = -t0
			# hopping between different Bravais lattice in L
			xx = x == L ? 1 : x + 1
			ii = xy2siteidx(1, xx, y)
			hopping[k, ii] = -t0
			hopping[ii, k] = -t0
			# hopping between different Bravais lattice, to x-1,y+1
			xx = x == 1 ? L : x - 1
			yy = y == L ? 1 : y + 1
			kk = xy2siteidx(3, xx, yy)
			hopping[j, kk] = -t0
			hopping[kk, j] = -t0
		end
	end
	bondgroup = bg1
	kmat = zeros(_mattype, nlayer, nlayer)
	# add chemical potiential
	kmat = hopping - I * μ
	return kmat, bondgroup, sitemap, nlayer
end


"""Honeycomb latiice"""
function MakeHoneycomb()
	nlayer = 2 * L * L
	# idx: sublattice type
	xy2siteidx(idx::Integer, x::Integer, y::Integer) = idx + (y - 1) * 2 + (x - 1) * 2 * L
	# _mattype = ComplexF64
	sitemap = zeros(Integer, L, L)
	hopping = zeros(_mattype, nlayer, nlayer)#nnhopping
	nhopping = zeros(_mattype, nlayer, nlayer)#next-nnhopping
	bg1 = Array{Tuple{Integer, Integer}, 1}()#只保留一个bond用来等效成Hubbard U
	nnnbg1 = Array{Tuple{Integer, Integer}, 1}() #储存 nnn bond
	nnnbg2 = Array{Tuple{Integer, Integer}, 1}()
	nnnbg3 = Array{Tuple{Integer, Integer}, 1}()
	nnnbg4 = Array{Tuple{Integer, Integer}, 1}()
	nnnbg5 = Array{Tuple{Integer, Integer}, 1}()
	nnnbg6 = Array{Tuple{Integer, Integer}, 1}()
	for x ∈ 1:L
		for y ∈ 1:L
			i = xy2siteidx(1, x, y)
			sitemap[x, y] = i
			j = xy2siteidx(2, x, y)
			k = j #k reserves value of j
			bg1 = [bg1; (i, i + nlayer); (j, j + nlayer)]
			""" NN hopping """
			# hopping in the same Bravais lattice between A and B
			hopping[i, j] = -t0
			hopping[j, i] = -t0
			# hopping between different Bravais lattice in L
			# periodic boundary condition
			yy = y == 1 ? L : y - 1
			j = xy2siteidx(2, x, yy)
			hopping[i, j] = -t0
			hopping[j, i] = -t0
			# hopping between different Bravais lattice in L
			# hopping between different Bravais lattice, to x-1
			xx = x == 1 ? L : x - 1
			j = xy2siteidx(2, xx, y)
			hopping[i, j] = -t0
			hopping[j, i] = -t0

			""" NNN hopping between (x,y) and (x,y+1)/ (x-1,y)/ (x+1,y-1) """
			""" hopping between A-A (flow in: +) """
			""" (x,y+1) bond 1 """
			xxx = x
			# periodic boundary condition
			yyy = y == L ? 1 : y + 1
			j = xy2siteidx(1, xxx, yyy)
			nhopping[i, j] = -λ
			nhopping[j, i] = +λ
			nnnbg1 = [nnnbg1; (i, j)]
			""" (x-1,y) bond 2 """
			# periodic boundary condition
			xxx = x == 1 ? L : x - 1
			yyy = y
			j = xy2siteidx(1, xxx, yyy)
			nhopping[i, j] = -λ
			nhopping[j, i] = +λ
			nnnbg2 = [nnnbg2; (i, j)]
			""" (x+1,y-1) bond 3 """
			# periodic boundary condition
			xxx = x == L ? 1 : x + 1
			# periodic boundary condition
			yyy = y == 1 ? L : y - 1
			# open boundary condition
			j = xy2siteidx(1, xxx, yyy)
			nhopping[i, j] = -λ
			nhopping[j, i] = +λ
			nnnbg3 = [nnnbg3; (i, j)]
			""" hopping between B-B (flow out: +) """
			""" (x,y+1) bond 4 """
			xxx = x
			# periodic boundary condition
			yyy = y == L ? 1 : y + 1
			j = xy2siteidx(2, xxx, yyy)
			nhopping[k, j] = +λ
			nhopping[j, k] = -λ
			nnnbg4 = [nnnbg4; (j, k)]
			""" (x-1,y) bond 5 """
			xxx = x == 1 ? L : x - 1
			yyy = y
			j = xy2siteidx(2, xxx, yyy)
			nhopping[k, j] = +λ
			nhopping[j, k] = -λ
			nnnbg5 = [nnnbg5; (j, k)]
			""" (x+1,y-1) bond 6 """
			xxx = x == L ? 1 : x + 1
			# periodic boundary condition
			yyy = y == 1 ? L : y - 1
			j = xy2siteidx(2, xxx, yyy)
			nhopping[k, j] = +λ
			nhopping[j, k] = -λ
			nnnbg6 = [nnnbg6; (j, k)]
		end
	end
	bondgroup = bg1
	nnnbg = [nnnbg1 nnnbg2 nnnbg3 nnnbg4 nnnbg5 nnnbg6]
	kmat = zeros(_mattype, nlayer, nlayer)

	# add chemical potiential
	kmat = hopping - I * μ
	return kmat, bondgroup, nnnbg, sitemap, nlayer
end



"""Square lattice"""
function MakeSquare()
	nlayer = L * L
	xy2siteidx(x::Integer, y::Integer) = (x - 1) * L + y
	# _mattype = ComplexF64
	sitemap = zeros(Integer, L, L)
	hopping = zeros(_mattype, nlayer, nlayer)
	bg1 = Array{Tuple{Integer, Integer}, 1}()
	bg2 = Array{Tuple{Integer, Integer}, 1}()
	bg3 = Array{Tuple{Integer, Integer}, 1}()
	bg4 = Array{Tuple{Integer, Integer}, 1}()
	for x ∈ 1:L
		for y ∈ 1:L
			i = xy2siteidx(x, y)
			sitemap[x, y] = i # store site(x,y) = idx
			xx = x == L ? 1 : x + 1
			yy = y == L ? 1 : y + 1
			# hopping to x+1
			j1 = xy2siteidx(xx, y)
			hopping[j1, i] = -t0
			hopping[i, j1] = -t0
			# hopping to y+1
			j2 = xy2siteidx(x, yy)
			hopping[j2, i] = -t0
			hopping[i, j2] = -t0
			# bond in 4 group
			if (-1)^(x + y) == 1 # group 1(x),2(y)
				bg1 = i < j1 ? [bg1; (i, j1)] : [bg1; (j1, i)]
				bg2 = i < j2 ? [bg2; (i, j2)] : [bg2; (j2, i)]
			else # group 3(x),4(y)
				bg3 = i < j1 ? [bg3; (i, j1)] : [bg3; (j1, i)]
				bg4 = i < j2 ? [bg4; (i, j2)] : [bg4; (j2, i)]
			end
		end
	end
	bondgroup = [bg1 bg2 bg3 bg4]
	# add flavor N
	kmat = zeros(_mattype, nlayer * N, nlayer * N)
	for a ∈ 1:N
		kmat[(a-1)*nlayer+1:a*nlayer, (a-1)*nlayer+1:a*nlayer] = t2^(a - 1) * hopping
	end
	# add chemical potiential
	kmat = kmat - I * μ
	vals, vecs = eigen(kmat)
	#L>=4 && println("Kmat eigenvals: ", vals)
	return kmat, bondgroup, sitemap, nlayer * N, nlayer
end




""" Triangular lattice """
function MakeTriangular()
	nlayer = L * L
	xy2siteidx(x::Integer, y::Integer) = (x - 1) * L + y
	# _mattype = ComplexF64
	sitemap = zeros(Integer, L, L)
	hopping = zeros(_mattype, nlayer, nlayer)
	bg1 = Array{Tuple{Integer, Integer}, 1}()
	bg2 = Array{Tuple{Integer, Integer}, 1}()
	bg3 = Array{Tuple{Integer, Integer}, 1}()
	bg4 = Array{Tuple{Integer, Integer}, 1}()
	bg5 = Array{Tuple{Integer, Integer}, 1}()
	bg6 = Array{Tuple{Integer, Integer}, 1}()
	for x ∈ 1:L
		for y ∈ 1:L
			i = xy2siteidx(x, y)
			sitemap[x, y] = i # store site(x,y) = idx
			xx = x == L ? 1 : x + 1
			yy = y == L ? 1 : y + 1
			# hopping to x+1
			j1 = xy2siteidx(xx, y)
			hopping[j1, i] = -t0
			hopping[i, j1] = -t0
			# hopping to y+1
			j2 = xy2siteidx(x, yy)
			hopping[j2, i] = -t0
			hopping[i, j2] = -t0
			# hopping to x+1,y+1 (Triangular hopping)
			j3 = xy2siteidx(xx, yy)
			hopping[j3, i] = -t0
			hopping[i, j3] = -t0
			# NN bond in 4 group
			if (-1)^(x + y) == 1 # group 1(x),2(y)
				bg1 = i < j1 ? [bg1; (i, j1)] : [bg1; (j1, i)]
				bg2 = i < j2 ? [bg2; (i, j2)] : [bg2; (j2, i)]
			else # group 3(x),4(y)
				bg3 = i < j1 ? [bg3; (i, j1)] : [bg3; (j1, i)]
				bg4 = i < j2 ? [bg4; (i, j2)] : [bg4; (j2, i)]
			end
			# NNN bond in 2 group
			if (-1)^(y) == 1 # group 5
				bg5 = i < j3 ? [bg5; (i, j3)] : [bg5; (j3, i)]
			else # group 6
				bg6 = i < j3 ? [bg6; (i, j3)] : [bg6; (j3, i)]
			end
		end
	end
	bondgroup = [bg1 bg2 bg3 bg4 bg5 bg6]
	# add flavor N
	kmat = zeros(_mattype, nlayer * N, nlayer * N)
	for a ∈ 1:N
		kmat[(a-1)*nlayer+1:a*nlayer, (a-1)*nlayer+1:a*nlayer] = hopping
	end
	# add chemical potiential
	kmat = kmat - I * μ
	vals, vecs = eigen(kmat)
	#L>=4 && println("Kmat eigenvals: ", vals)
	return kmat, bondgroup, sitemap, nlayer * N, nlayer
end





