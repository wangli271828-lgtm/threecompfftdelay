#data = zeros(Float64, 3, 3, n)
#data[:, :, 1] .= R1
#data[:, :, 2] .= R2
#expV = LocalHSMatrix(data)
#expV.data[:, :, 1]  # first site's 3×3 matrix
#expV.data[:, :, 2]  # second site's 3×3 matrix
#目标：仅存非零元素
struct LocalHSMatrix{T<:AbstractFloat}
    data::Array{T,3}
end

#F=A*V
function LinearAlgebra.mul!(
    F::Matrix{T},
    A::Matrix{T},
    V::LocalHSMatrix{T},
) where {T<:AbstractFloat}

    R = V.data
    n = size(R, 3)

    size(R, 1) == size(R, 2) == 3 || #短路或，条件为假，执行下面的
        throw(DimensionMismatch("Local matrices must be 3×3."))

    size(A, 2) == 3 * n ||
        throw(DimensionMismatch("A must have 3n columns."))

    size(F) == size(A) ||
        throw(DimensionMismatch("F and A must have the same size."))

    @inbounds for i in 1:n
        i1 = i
        i2 = i + n
        i3 = i + 2 * n

        r11, r12, r13 = R[1, 1, i], R[1, 2, i], R[1, 3, i]
        r21, r22, r23 = R[2, 1, i], R[2, 2, i], R[2, 3, i]
        r31, r32, r33 = R[3, 1, i], R[3, 2, i], R[3, 3, i]

        for row in axes(A, 1)
            a1 = A[row, i1]
            a2 = A[row, i2]
            a3 = A[row, i3]

            F[row, i1] = a1 * r11 + a2 * r21 + a3 * r31
            F[row, i2] = a1 * r12 + a2 * r22 + a3 * r32
            F[row, i3] = a1 * r13 + a2 * r23 + a3 * r33
        end
    end

    return F
end

# F = V * A
function LinearAlgebra.mul!(
    F::Matrix{T},
    V::LocalHSMatrix{T},
    A::Matrix{T},
) where {T<:AbstractFloat}

    R = V.data
    n = size(R, 3)

    size(R, 1) == size(R, 2) == 3 ||
        throw(DimensionMismatch("Local matrices must be 3×3."))

    size(A, 1) == 3 * n ||
        throw(DimensionMismatch("A must have 3n rows."))

    size(F) == size(A) ||
        throw(DimensionMismatch("F and A must have the same size."))

    @inbounds for col in axes(A, 2)
        for i in 1:n
            i1 = i
            i2 = i + n
            i3 = i + 2 * n

            a1 = A[i1, col]
            a2 = A[i2, col]
            a3 = A[i3, col]

            F[i1, col] =
                R[1, 1, i] * a1 + R[1, 2, i] * a2 + R[1, 3, i] * a3
            F[i2, col] =
                R[2, 1, i] * a1 + R[2, 2, i] * a2 + R[2, 3, i] * a3
            F[i3, col] =
                R[3, 1, i] * a1 + R[3, 2, i] * a2 + R[3, 3, i] * a3
        end
    end

    return F
end