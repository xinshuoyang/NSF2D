module NSF2D

using SparseArrays
using LinearAlgebra
using JLD

import Base: copy, transpose

export CanTenDec
export NSFVector
export NSFMatrix
export to_dense

include("utils.jl")
include("wavelets.jl")
include("CanTenDec.jl")
include("NSFVector.jl")
include("NSFMatrix.jl")
include("DNSFMatrix.jl")

end
