abstract type AbstractBlockDiagonal{T} <: AbstractMatrix{T} end
blocks(B::AbstractBlockDiagonal) = B.blocks

_is_square(A::AbstractMatrix) = size(A, 1) == size(A, 2)

"""
	BlockDiagonal

A block-diagonal matrix.
Contains the blocks (`blocks`), their global position in the matrix (`block_row_indices`, `block_col_indices`), and information about if all blocks are square (`is_block_square`).
"""
struct BlockDiagonal{T, V <: AbstractMatrix{T}} <: AbstractBlockDiagonal{T}
	blocks::Vector{V}
	block_row_indices::Vector{Int}
	block_col_indices::Vector{Int}
	is_block_square::Bool
end

"""
	cumulative_square_indices(V;dim=1)

Computes the comulative indicies of the blocks in `V` in the `dim` direction.
The vector start at zero, so the block have indicies from (index[i] + 1):index[i + 1]
"""
function cumulative_square_indices(V; dim = 1)
	block_dim_size = size.(V, dim)
	return [0; cumsum(block_dim_size)]
end

"""
	BlockDiagonal(blocks)

Creates a block-diagonal matrix given a vector of the `blocks`.
"""
function BlockDiagonal(blocks::Vector{V}) where {T, V <: AbstractMatrix{T}}
	block_row_indices = cumulative_square_indices(blocks)
	block_col_indices = cumulative_square_indices(blocks, dim = 2)
	is_block_square = all(block -> _is_square(block), blocks)
	return BlockDiagonal{T, V}(blocks, block_row_indices, block_col_indices, is_block_square)
end

function LinearAlgebra.mul!(y::AbstractVecOrMat{T}, B::BlockDiagonal{T, V}, x::AbstractVecOrMat{T}) where {T, V}
	@floop @inbounds for (block_id, block) in enumerate(blocks(B))
		block_row_start = B.block_row_indices[block_id] + 1
		block_row_end = B.block_row_indices[block_id+1]
		block_col_start = B.block_col_indices[block_id] + 1
		block_col_end = B.block_col_indices[block_id+1]
		@views mul!(y[block_row_start:block_row_end, :], block, x[block_col_start:block_col_end, :])
	end
end

function LinearAlgebra.mul!(y::AbstractVecOrMat{T}, x::AbstractVecOrMat{T}, B::BlockDiagonal{T, V}) where {T, V}
	@floop @inbounds for (block_id, block) in enumerate(blocks(B))
		block_row_start = B.block_row_indices[block_id] + 1
		block_row_end = B.block_row_indices[block_id+1]
		block_col_start = B.block_col_indices[block_id] + 1
		block_col_end = B.block_col_indices[block_id+1]
		@views mul!(y[:, block_col_start:block_col_end], x[:, block_row_start:block_row_end], block)
	end
end
