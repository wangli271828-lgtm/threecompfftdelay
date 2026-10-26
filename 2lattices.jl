"""Square lattice with periodic boundaries and no chemical-potential term."""
function Makesquare()
	nlayer = L * L
	xy2siteidx(x::Integer, y::Integer) = (x - 1) * L + y
	sitemap = zeros(Integer, L, L)
	hopping = zeros(_mattype, nlayer, nlayer)
	bg1 = Array{Tuple{Integer, Integer}, 1}()
	for x in 1:L
		for y in 1:L
			i = xy2siteidx(x, y)
			sitemap[x, y] = i
			push!(bg1, (i, i))
			# Store each distinct neighbor once and exclude self-connections.
			xx = x == L ? 1 : x + 1
			yy = y == L ? 1 : y + 1
			j = xy2siteidx(xx, y)
			if i != j
				hopping[i, j] = -t0
				hopping[j, i] = -t0
			end
			j = xy2siteidx(x, yy)
			if i != j
				hopping[i, j] = -t0
				hopping[j, i] = -t0
			end
		end
	end
	bondgroup = bg1
	kmat = hopping
	return kmat, bondgroup, sitemap, nlayer
end
