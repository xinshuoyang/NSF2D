using SparseArrays
using LinearAlgebra

# """
# Removes stored values from A whose absolute value is less than or equal  
# """
# function dropband!(A::SparseMatrixCSC{Float64,Int},atol::Float64)
#     m,n = size(A)
#     if m != n
#         throw(DimensionMismatch("expected a square matrix, got a matrix of size $m by $n"))
#     end
    
#     # Determine the bandwidth using the first column
#     bw = 0
#     while bw <= n-2 && abs(A[bw+2,1]) >= atol/1e1
#         bw += 1
#     end
    
#     # if bw >= Int(ceil(m/2))
#     #     @warn "Bandwidth $bw is greater than half of the matrix size, retaining all entries!"
#     # end

#     # # println("dense inverse is ",Matrix(A))
#     # bw = m
        
#     # Remove nonzeros outside the band
#     rows,columns,vals = findnz(A)
#     for j in 1:n
#         for i in nzrange(A,j)
#             if rows[i] != j
#                 if abs(j-rows[i])<=bw || abs(abs(j-rows[i])-m)<=bw
#                     continue
#                 else
#                     vals[i] = 0.
#                 end
#             end
#         end
#     end
#     A.nzval .= vals
#     dropzeros!(A)
# end

"""
Removes stored values from A which are outside a given bandwidth  
"""
function dropband!(A::SparseMatrixCSC{Float64,Int},bw::Int64)
    m,n = size(A)
    if m != n
        throw(DimensionMismatch("expected a square matrix, got a matrix of size $m by $n"))
    end

    if bw <= 0
        throw("invliad bandwidth $bw")
    end
        
    # Remove nonzeros outside the band
    rows,columns,vals = findnz(A)
    for j in 1:n
        for i in nzrange(A,j)
            if rows[i] != j
                if abs(j-rows[i])<=bw || abs(abs(j-rows[i])-m)<=bw
                    continue
                else
                    vals[i] = 0.
                end
            end
        end
    end
    A.nzval .= vals
    dropzeros!(A)
end

"""
Inner product (element-wise product) between two matrices
"""
function innerprod(A::AbstractMatrix{Float64},B::AbstractMatrix{Float64})::Float64
    if size(A) != size(B)
        throw(DimensionMismatch("size of A not equal to size of B"))
    end
    return sum(A.*B)/size(A)[1]
end

"""
Return a copy of the off-diagonal part of matrix
"""
function get_offdiag(A::AbstractMatrix{Float64})::AbstractMatrix{Float64}
    return A-Diagonal(A)
end

"""
Create center finite difference matrix
"""
function get_cfd_mat(ord::Int,n::Int)::SparseMatrixCSC{Float64,Int}
    if ord == 2
        arr = circshift(vcat(Vector{Float64}([1,-2,1]),zeros(Float64,n-3)),-1)
    elseif ord == 4
        arr = circshift(vcat(Vector{Float64}([-1/12,4/3,-5/2,4/3,-1/12]),zeros(Float64,n-5)),-2)
    elseif ord == 8
        arr = circshift(vcat(Vector{Float64}([1/90,-3/20,3/2,-49/18,3/2,-3/20,1/90]),zeros(Float64,n-7)),-3)
    else
        throw("unimplemented")
    end

    A = Matrix{Float64}(undef,n,n)
    for i in 1:n
        A[i,:] = arr
        arr = circshift(arr,1)
    end

    return A
end

function get_stagger_grids(ord::Int,n::Int,a::Vector{Float64})::Matrix{Float64}
    if length(a) != n
        throw("size of vector a not equal to n")
    end

    if ord == 2
        coefs = Float64[  1  0;
                         -1 -1;
                          0  1;]

        h = 1.0/n
        A = zeros(Float64,n,n)

        for i in 1:n
            arr = circshift(a,2-i)[1:2]
            A[i,:] = circshift(vcat(coefs*arr,zeros(Float64,n-3)),i-2)
        end 

    elseif ord == 4        
        coefs = Float64[ 1/576      0      0      0;
                         -3/64  -3/64      0      0;
                          3/64  81/64   3/64      0;
                        -1/576 -81/64 -81/64 -1/576;
                             0   3/64  81/64   3/64;
                             0      0  -3/64  -3/64;
                             0      0      0  1/576;]

        h = 1.0/n
        A = zeros(Float64,n,n)

        for i in 1:n
            arr = circshift(a,3-i)[1:4]
            A[i,:] = circshift(vcat(coefs*arr,zeros(Float64,n-7)),i-4)
        end 

    elseif ord == 6
        coefs = Float64[ 9/409600           0          0          0           0        0;
                         -5/16384    -5/16384          0          0           0        0;
                          45/8192  625/147456    45/8192          0           0        0;
                         -45/8192   -625/8192  -625/8192   -45/8192           0        0;
                          5/16384    625/8192  5625/4096   625/8192     5/16384        0;
                        -9/409600 -625/147456 -5625/4096 -5625/4096 -625/147456 -9/409600;
                                0     5/16384   625/8192  5625/4096    625/8192   5/16384;
                                0           0   -45/8192  -625/8192   -625/8192  -45/8192;
                                0           0          0    45/8192  625/147456   45/8192;
                                0           0          0          0    -5/16384  -5/16384;
                                0           0          0          0           0  9/409600;]

        h = 1.0/n
        A = zeros(Float64,n,n)

        for i in 1:n
            arr = circshift(a,4-i)[1:6]
            A[i,:] = circshift(vcat(coefs*arr,zeros(Float64,n-11)),i-6)
        end 
    else
        throw("unimplemented")
    end
    return A
end

"""
Check if a matrix is strictly diagonally dominant
"""
function is_diagally_dominant(A::AbstractMatrix{Float64})
    diff = abs.(diag(A)).-collect(Iterators.flatten(sum(abs.(get_offdiag(A)),dims=1)))
    return minimum(diff)
end


"""
Get 2D meshgrid
"""
function meshgrid(x,y)
    return (reshape(repeat(x,outer=length(y)),(length(x),length(y))),
        reshape(repeat(y,inner=length(x)),(length(x),length(y))))
end


"""
Get index for repartition of girds 
"""
function get_repartiton_index(n::Int)::Vector{Int}
    idx = transpose(reshape(collect([1:n*n;]),(n,n)))
    return vcat(collect(Iterators.flatten(idx[1:n÷2,1:n÷2])),
        collect(Iterators.flatten(idx[n÷2+1:n,1:n÷2])),
        collect(Iterators.flatten(idx[1:n÷2,n÷2+1:n])),
        collect(Iterators.flatten(idx[n÷2+1:n,n÷2+1:n]))
    )
end