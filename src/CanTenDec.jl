"""
Canonical Tensor Decompositions
"""
mutable struct CanTenDec
    size::Int
    capacity::Int
    rank::Int
    sig::Vector{Float64}
    u::Vector{SparseMatrixCSC{Float64,Int}}
    v::Vector{SparseMatrixCSC{Float64,Int}}
end

"""
Make a copy of CanTenDec
"""
copy(A::CanTenDec) = CanTenDec(A.size,A.capacity,A.rank,copy(A.sig),copy(A.u),copy(A.v))

"""
Populate s-values in descending order
"""
function get_svals(A::CanTenDec)::Vector{Float64}
    return sort(A.sig[1:A.rank],rev=true)
end

"""
Get the largest s-value
"""
function get_max_sval(A::CanTenDec)::Float64
    if A.rank == 0
        throw("attempt to retrive max s-value of an empty CanTenDec")
    end

    return maximum(A.sig[1:A.rank])
end


"""
Create a zero CanTenDec of the same size as A and with a given (optional) capacity
"""
create_zeros(size::Int,capacity::Int=0) = CanTenDec(size,capacity,0,
    zeros(Float64,capacity),[spzeros(size) for i in 1:capacity],[spzeros(size) for i in 1:capacity])

"""
Create a identity CanTenDec of the same size as A and with a given (optoinal) capacity
"""
function create_identity(size::Int,capacity::Int=1)
    u = [spzeros(size,size) for i in 1:capacity]
    v = [spzeros(size,size) for i in 1:capacity]

    sig[1] = 1.0
    u[1] = sparse(Matrix(1.0I,size,size))
    v[1] = sparse(Matrix(1.0I,size,size))

    return CanTenDec(size,1,capacity,sig,u,v)
end

"""
Increase the capacity of CanTenDec
"""
function resize(A::CanTenDec,capacity::Int)::Nothing

    capacity = capacity > 2*A.capacity ? capacity : 2*A.capacity

    sig = Vector{Float64}(undef,capacity)
    u = Vector{SparseMatrixCSC{Float64,Int}}(undef,capacity)
    v = Vector{SparseMatrixCSC{Float64,Int}}(undef,capacity)

    for i in 1:A.rank
        sig[i] = A.sig[i]
        u[i] = copy(A.u[i])
        v[i] = copy(A.v[i])
    end

    A.sig = sig
    A.u = u
    A.v = v
    A.capacity = capacity

    return
end

"""
Extract the diagonal vector
"""
function diagvec(A::CanTenDec)::Vector{Float64}
    return sum(A.sig[j]*kron(Vector(diag(A.u[j])),Vector(diag(A.v[j]))) for j in 1:A.rank)
end


"""
Get a copy of the diagonals of A
"""
function get_diag(A::CanTenDec)::CanTenDec
    sig = Vector{Float64}(undef,A.rank)
    u = Vector{SparseMatrixCSC{Float64,Int}}(undef,A.rank)
    v = Vector{SparseMatrixCSC{Float64,Int}}(undef,A.rank)

    for i in 1:A.rank
        sig[i] = A.sig[i]
        u[i] = spdiagm(0=>diag(A.u[i]))
        v[i] = spdiagm(0=>diag(A.v[i]))
    end
    return CanTenDec(A.size,A.rank,A.rank,sig,u,v)
end

"""
Get a copy of the off-diagonals of A
"""
# function get_offdiag(A::CanTenDec,atol::Float64=1e-15)::CanTenDec
#     max_rank = 2*A.rank
#     sig = Vector{Float64}(undef,max_rank)
#     u = Vector{SparseMatrixCSC{Float64,Int}}(undef,max_rank)
#     v = Vector{SparseMatrixCSC{Float64,Int}}(undef,max_rank)

#     for i in 1:A.rank
#         sig[i] = A.sig[i]
#         u[i] = copy(A.u[i])
#         v[i] = copy(A.v[i])
#     end

#     rank = A.rank
#     for i in 1:A.rank
#         sig[rank+1] = A.sig[i]
#         u[rank+1] = spdiagm(0=>-diag(A.u[i]))
#         v[rank+1] = spdiagm(0=>diag(A.v[i]))
#         uu = sqrt(innerprod(u[rank+1],u[rank+1]))
#         vv = sqrt(innerprod(v[rank+1],v[rank+1]))

#         if sig[rank+1]*uu*vv > atol
#             sig[rank+1] *= uu*vv
#             u[rank+1] /= uu
#             v[rank+1] /= vv
#             rank += 1
#         end
#     end

#     B = CanTenDec(A.size,rank,rank,sig[1:rank],u[1:rank],v[1:rank])
    
#     reduce(B,atol)
#     # fastchol(B,atol=atol)

#     return B
# end

"""
Normalize CanTenDec
"""
function normalize(A::CanTenDec,atol::Float64)::Nothing
    atol2 = atol*atol
    rank = 0
    for i in 1:A.rank
        uu = sqrt(innerprod(A.u[i],A.u[i]))
        vv = sqrt(innerprod(A.v[i],A.v[i]))

        if A.sig[i]*uu*vv > atol2
            rank += 1
            A.sig[rank] = A.sig[i]*uu*vv
            A.u[rank] = A.u[i]/uu
            A.v[rank] = A.v[i]/vv
        end
    end
    A.rank = rank

    return
end


"""
Convert CanTenDec to dense matrix
"""
function to_dense(A::CanTenDec)::Matrix{Float64}
    B = zeros(Float64,A.size^2,A.size^2)
    for i in 1:A.rank
        B += A.sig[i]*kron(A.u[i],A.v[i])
    end
    return B
end

"""
Plus a constant times another CanTenDec
Y <- alpha * X + Y

:prarm alpha: constant
:type alpha: Float64
:param X: canonical tensor decomposition
:type X: CanTenDec
:param Y: canonical tensor decomposition
:type Y: CanTenDec
"""
function axpy(alpha::Float64,
        X::CanTenDec,
        Y::CanTenDec,
        atol::Float64;
        rel::Bool=true,
    )::Nothing

    if X.size != Y.size
        throw(DimensionMismatch("size of X not equal to size of Y"))
    end

    if Y.capacity < X.rank+Y.rank
        resize(Y,X.rank+Y.rank)
    end

    if alpha > 0.0
        for i in 1:X.rank
            Y.sig[Y.rank+i] = alpha*X.sig[i]
            Y.u[Y.rank+i] = copy(X.u[i])
            Y.v[Y.rank+i] = copy(X.v[i])
        end
    else
        for i in 1:X.rank
            Y.sig[Y.rank+i] = -alpha*X.sig[i]
            Y.u[Y.rank+i] = -X.u[i]
            Y.v[Y.rank+i] = copy(X.v[i])
        end
    end
    Y.rank += X.rank

    # reduce(Y,atol,rel=rel)

    return
end

"""
X <- alpha * I + X
"""
function aipy(alpha::Float64,X::CanTenDec)::Nothing
    if X.capacity < X.rank+1
        resize(X,X.rank+1)
    end
    if alpha > 0.0
        X.sig[X.rank+1] = alpha
        X.u[X.rank+1] = sparse(I,X.size,X.size)
        X.v[X.rank+1] = sparse(I,X.size,X.size)
    else
        X.sig[X.rank+1] = -alpha
        X.u[X.rank+1] = -sparse(I,X.size,X.size)
        X.v[X.rank+1] = sparse(I,X.size,X.size)
    end
    X.rank += 1
    return
end


"""
Perform multiplication between two CanTenDec A and B
C <- A * B
"""
function mul(A::CanTenDec,
    B::CanTenDec,
    atol::Float64;
    )::CanTenDec

    if !(A.size == B.size)
        throw(DimensionMismatch("sizes of A not equal to size of B"))
    end

    capacity = A.rank*B.rank
    C = CanTenDec(A.size,capacity,0,
        Vector{Float64}(undef,capacity),
        Vector{SparseMatrixCSC{Float64,Int}}(undef,capacity),
        Vector{SparseMatrixCSC{Float64,Int}}(undef,capacity),
    )
    
    atol2 = atol*atol
    for i in 1:A.rank, j in 1:B.rank
        C.sig[C.rank+1] = A.sig[i]*B.sig[j]
        C.u[C.rank+1] = A.u[i]*B.u[j]
        C.v[C.rank+1] = A.v[i]*B.v[j]
        
        uu = sqrt(innerprod(C.u[C.rank+1],C.u[C.rank+1]))
        vv = sqrt(innerprod(C.v[C.rank+1],C.v[C.rank+1]))

        if C.sig[C.rank+1]*uu*vv > atol2
            C.rank += 1

            C.sig[C.rank] *= uu*vv
            C.u[C.rank] /= uu
            C.v[C.rank] /= vv
        end
    end

    # reduce(C,atol,rel=rel)
    # fastchol(C,atol=atol)

    return C
end


"""
Compute the diagonals of A*B
"""
function diagmul(A::CanTenDec,B::CanTenDec)::Vector{Float64}
    size = A.size
    if size != B.size
        DimensionMismatch("first CanTenDec has size $(A.size) which does not match the size of the second, $(B.size)")
    end
    d = zeros(Float64,size^2)
    for i in 1:A.rank, j in 1:B.rank
        d += A.sig[i]*B.sig[j]*kron([dot(A.u[i][k,:],B.u[j][:,k]) for k in 1:size],
            [dot(A.v[i][k,:],B.v[j][:,k]) for k in 1:size])
    end
    return d
end

"""
Scale CanTenDec by constant
"""
function scale(A::CanTenDec,alpha::Float64)::Nothing
    # if isapprox(alpha,0.0,atol=atol)
    if alpha == 0.0
        A.rank = 0
    elseif alpha > 0.
        A.sig[1:A.rank] *= alpha
    else
        A.sig[1:A.rank] *= -alpha
        for i in 1:A.rank
            A.u[i] *= -1.
        end
    end
    return
end

"""
Compute inverse of a diagonal CanTenDec
"""
# function get_inv_diag(A::CanTenDec,atol::Float64;max_iter=20)::CanTenDec

#     atol2 = atol/1e2

#     # Get diagonals of A and select alpha
#     diags = sum(A.sig[k]*kron(Array(diag(A.u[k])),Array(diag(A.v[k]))) for k in 1:A.rank)
#     sigs = diags.^2
#     alpha = 2.0/(maximum(sigs)+minimum(sigs))

#     println("alpha = ",alpha)
#     DA = get_diag(A)

#     G0 = copy(DA)
#     scale(G0,alpha)

    
#     for i in 1:max_iter
#         G1 = copy(G0)
#         scale(G1,2.)

#         # T = mul(mul(G0,DA,atol2,rel=false),G0,atol2,rel=false)
#         # axpy(-1.,T,G1,atol2,rel=false)
#         # reduce(G1,atol2,rel=false)

#         T = mul(mul(G0,DA,atol2),G0,atol2)
#         axpy(-1.,T,G1,atol2)
#         reduce(G1,atol2)
#         # fastchol(G1,atol=atol2)

#         G0 = copy(G1)

#         println("iteration = ")
#         println(i)
#         println("rank of G0")
#         println(G0.rank)
#         println("sig of G0")
#         println(sort(G0.sig[1:G0.rank],rev=true))
#         println("inverse error")
#         err = norm(ones(Float64,A.size^2).-sum(G0.sig[k]*kron(Array(diag(G0.u[k])),Array(diag(G0.v[k]))) for k in 1:G0.rank).*diags,Inf)
#         # println(norm(diag(to_dense(G0)).*diag(to_dense(DA))-ones(Float64,A.size^2),Inf))
#         println(err)

#         if err < atol
#             @goto converge
#         end
#     end
#     @warn "Schulz iteration does not converge with error tolerance $(atol) after $(max_iter) iterations"
#     @label converge
#     return G0
# end


"""
Compute inverse square root of tensor product sum
     n
A = sum u_j ⊗ v_j
    j=1
where u_j and v_j are diagonal matrices.
"""
function invsqrt(A::CanTenDec,atol::Float64;n_iter::Int=20)::CanTenDec
    # Compute norm of A
    Anorm = max(sum(kron(diag(A.u[j]),diag(A.v[j])) for j in 1:A.rank))

    # Get initial guess
    X0 = CanTenDec(A.size,1,1,[1.0],1,1)

    # Iteration
    for _ in 1:n_iter
        T = mul(mul(X0,A,atol),X0,atol)
        scale(T,-0.5/Anorm)
        aipy(1.5,T)
        X1 = mul(X0,T,atol)
        reduce(X1,atol)
        # fastchol(X1,atol=atol)
        X0 = X1

        # Compute error
        T = (mul(X0,A,atol),X0,atol)
        err = max(sum(T.sig[j]*kron(diag(T.u[j]),diag(T.v[j])) for j in 1:T.rank)-Anorm*ones(Float64,A*size^2))

        if err < atol2
            @goto converge
        end
    end
    @warn "Schulz iteration does not converge with error tolerance $(atol2) after $(max_iter) iterations"
    @label converge
    scale(X0,Anorm)
    return X0
end

"""
Apply (multiply) CanTenDec to dense matrix
"""
function apply(A::CanTenDec,X::Matrix{Float64};TRANS::String="N")::Matrix{Float64}
    # Check inputs
    if size(X) != (A.size,A.size)
        throw(DimensionMismatch("size of A not compatible with size of X"))
    end

    if A.rank > 0
        if TRANS == "N"
            return sum(A.sig[i]*A.v[i]*X*transpose(A.u[i]) for i in 1:A.rank)
        elseif TRANS == "T"
            return sum(A.sig[i]*transpose(A.v[i])*X*A.u[i] for i in 1:A.rank)
        else
            throw("invalid input for TRANS")
        end
    else
        return zeros(Float64,size(X))
    end
end

"""
Solve A * x = b via SVD
"""
function solve(A::CanTenDec,
    b::Matrix{Float64},
    atol::Float64,
    )::Matrix{Float64}

    if size(b) != (A.size,A.size)
        throw(DimensionMismatch("size of A not equal to size of b"))
    end

    n = A.size*A.size

    F = LinearAlgebra.svd(to_dense(A))
    sig = 1.0./F.S
    sig[length(sig)] = 0.


    # sig = Vector{Float64}(undef,n)
    # for i in 1:n
    #     if F.S[i] > atol
    #     # if F.S[i] > 10e-7
    #         sig[i] = 1.0/F.S[i]
    #     else
    #         sig[i] = 0.0
    #     end
    # end
    # println("coarsest scale condition number $(F.S[1]/F.S[n-1]) and $(F.S[n])")

    x = reshape(transpose(F.Vt)*(sig.*(transpose(F.U)*reshape(b,n,1))),size(b))

    return x
end


"""
Reduce the rank of CanTenDec via Gram-Schmidt Orthogonalization

A --- 
atol ---
maxsig ---
maxiter --- 
if maxsig<=0, truncate the rest s-values wrt the largest s-value before reduction
otherwise truncate all s-values wrt maxsig.
"""
function reduce(A::CanTenDec,atol::Float64;maxiter::Int=1)::Nothing
    
    # Quick return if possible
    if A.rank == 0
        return
    end

    # println("Start reduction")
    
    # atol2 = atol/1e2
    atol2 = atol
    rank = A.rank

    # Initialize pivot vector
    ipivot = collect(1:rank)

    for _ in 1:maxiter

        # println("iter $(iter)")

        rankini = rank

        # println("initial rank is $(rankini)")

        # Orthogonalize u-vectors
        for j in 1:rank
            # Pivoting: find index for the largest sig
            smax = A.sig[ipivot[j]]
            kmax = j
            for k in j+1:rank
                if smax < A.sig[ipivot[k]]
                    kmax = k
                    smax = A.sig[ipivot[k]]
                end
            end

            # Swap into the leading position
            ipivot[j],ipivot[kmax] = ipivot[kmax],ipivot[j]

            mm = ipivot[j]

            # Check if sig are large enough:
            # if they are smaller than a threshold
            # then reset the rank (run out of ipvots)
            # if A.sig[mm]/maxsig <= atol2
            if A.sig[mm] <= atol2
                rank = j-1
                break
            end

            for l in j+1:rank

                ll = ipivot[l]

                # If sig is smaller than a threshold,
                # set sig to be ignored from now on
                # if A.sig[ll]/maxsig <= atol2
                if A.sig[ll] <= atol2
                    A.sig[ll] = 0.
                    @goto next_uvec
                end

                # Modify u-vectors
                dalpha = innerprod(A.u[mm],A.u[ll])
                A.u[ll] -= dalpha*A.u[mm]
                fact1 = sqrt(innerprod(A.u[ll],A.u[ll]))
                if fact1*A.sig[ll] > atol2
                    A.u[ll] /= fact1
                end

                # Modify v-vectors
                dalpha *= A.sig[ll]/A.sig[mm]
                A.v[mm] += dalpha*A.v[ll]

                # Adjust sig
                A.sig[ll] *= fact1

                @label next_uvec
            end

            # Adjust sig of the pivot vector
            fact2 = sqrt(innerprod(A.v[mm],A.v[mm]))
            if fact2*A.sig[mm] > atol2
                A.v[mm] /= fact2
            end
            A.sig[mm] *= fact2
        end

        # println("sweeping of u-vectors reduce to $(rank)")

        # Orthogonalize v-vectors
        for j in 1:rank

            # Pivoting: find index for the largest sig
            smax = A.sig[ipivot[j]]
            kmax = j
            for k in j+1:rank
                if smax < A.sig[ipivot[k]]
                    kmax = k
                    smax = A.sig[ipivot[k]]
                end
            end

            # Swap into the leading position
            ipivot[j],ipivot[kmax] = ipivot[kmax],ipivot[j]

            mm = ipivot[j]

            # Check if sig are large enough
            # if they are smaller than a threshold,
            # then reset to the rank (run out of pivots)
            # if A.sig[mm]/maxsig <= atol2
            if A.sig[mm] <= atol2
                rank = j-1
                break
            end

            for l in j+1:rank

                ll = ipivot[l]

                # if sig is smaller than a threshold, set sig to zero
                # to be ignored from now on
                # if A.sig[ll]/maxsig <= atol2
                if A.sig[ll] <= atol2
                    A.sig[ll] = 0.
                    @goto next_vvec
                end

                # Modify v-vectors
                dalpha = innerprod(A.v[mm],A.v[ll])
                A.v[ll] -= dalpha*A.v[mm]
                fact1 = sqrt(innerprod(A.v[ll],A.v[ll]))
                if fact1*A.sig[ll] > atol2
                    A.v[ll] /= fact1
                end

                # Modify u-vectors
                dalpha *= A.sig[ll]/A.sig[mm]
                A.u[mm] += dalpha*A.u[ll]

                # Adjust sig
                A.sig[ll] *= fact1

                @label next_vvec
            end

            # Adjust sig of the pivot vector
            fact2 = sqrt(innerprod(A.u[mm],A.u[mm]))
            if fact2*A.sig[mm] > atol2
                A.u[mm] /= fact2
            end
            A.sig[mm] *= fact2
        end

        # println("reduce to $(rank)")

        if rank == rankini
            break
        end

    end


    # Fill in gaps, if any
    # Find all pivots less than the new rank with actual location beyond rank
    # these vectors and sig need to be swapped to location within the range 1<=...<=rank
    iempty = Vector{Int}(undef,A.rank)

    icount = 1
    for l in rank+1:A.rank
        if ipivot[l] <= rank
            iempty[icount] = ipivot[l]
            icount += 1
        end
    end

    # Find all pivots larger than the new rank, These lications are overwritten
    # by the vectors and sig values within the range 1<=..<=rank found above
    icount = 1
    for l = 1:rank
        if ipivot[l] > rank
            A.u[iempty[icount]] = copy(A.u[ipivot[l]])
            A.v[iempty[icount]] = copy(A.v[ipivot[l]])
            A.sig[iempty[icount]] = A.sig[ipivot[l]]
            ipivot[l] = iempty[icount]
            icount += 1
        end
    end

    # Reset the rank
    A.rank = rank

    # Remove small values
    for i in 1:A.rank

        droptol!(A.u[i],atol2/A.sig[i])
        droptol!(A.v[i],atol2/A.sig[i])

        # dropband!(A.u[i],atol2/A.sig[i])
        # dropband!(A.v[i],atol2/A.sig[i])
    end

    # # Remove stored values outside bandwidth
    # bw = 30
    # for i in 1:A.rank
    #     dropband!(A.u[i],bw)
    #     dropband!(A.v[i],bw)
    # end

    # normalize(A,atol2)

    
    # println("Done reduction")
    return
end


# function reduce(A::CanTenDec,atol::Float64)::Nothing
#     U = Matrix{Float64}(undef,A.size^2,A.rank)
#     V = Matrix{Float64}(undef,A.size^2,A.rank)
#     for i in 1:A.rank
#         U[:,i] = vec(A.u[i])
#         V[:,i] = vec(A.v[i])
#     end
#     D = U*diagm(A.sig[1:A.rank])*transpose(V)

#     F = svd(D)
#     for i in 1:size(D)[1]
#         if F.S[i]/F.S[1] < atol
#             A.rank = i-1
#             break
#         else
#             A.sig[i] = F.S[i]
#             A.u[i] = reshape(F.U[:,i],A.size,A.size)
#             A.v[i] = reshape(F.V[:,i],A.size,A.size)
#         end
#     end
# end

# function reduce_abs(A::CanTenDec,atol::Float64;maxiter::Int=10)::Nothing
    
#     # Quick return if possible
#     if A.rank == 0
#         return
#     end

#     println("Start reduction")
    
#     atol2 = atol/1e2
#     # atol2 = atol
#     rank = A.rank

#     # Initialize pivot vector
#     ipivot = collect(1:rank)

#     for iter in 1:maxiter

#         println("iter $(iter)")

#         rankini = rank

#         println("initial rank is $(rankini)")

#         # Orthogonalize u-vectors
#         for j in 1:rank
#             # Pivoting: find index for the largest sig
#             smax = A.sig[ipivot[j]]
#             kmax = j
#             for k in j+1:rank
#                 if smax < A.sig[ipivot[k]]
#                     kmax = k
#                     smax = A.sig[ipivot[k]]
#                 end
#             end

#             # Swap into the leading position
#             ipivot[j],ipivot[kmax] = ipivot[kmax],ipivot[j]

#             mm = ipivot[j]

#             # Check if sig are large enough:
#             # if they are smaller than a threshold
#             # then reset the rank (run out of ipvots)
#             if A.sig[mm] <= atol2
#                 rank = j-1
#                 break
#             end

#             for l in j+1:rank

#                 ll = ipivot[l]

#                 # If sig is smaller than a threshold,
#                 # set sig to be ignored from now on
#                 if A.sig[ll] <= atol2
#                     A.sig[ll] = 0.
#                     break
#                 end

#                 # Modify u-vectors
#                 dalpha = innerprod(A.u[mm],A.u[ll])
#                 A.u[ll] -= dalpha*A.u[mm]
#                 fact1 = sqrt(innerprod(A.u[ll],A.u[ll]))
#                 if fact1*A.sig[ll] > atol2
#                     A.u[ll] /= fact1
#                 end

#                 # Modify v-vectors
#                 dalpha *= A.sig[ll]/A.sig[mm]
#                 A.v[mm] += dalpha*A.v[ll]

#                 # Adjust sig
#                 A.sig[ll] *= fact1

#             end

#             # Adjust sig of the pivot vector
#             fact2 = sqrt(innerprod(A.v[mm],A.v[mm]))
#             if fact2*A.sig[mm] > atol2
#                 A.v[mm] /= fact2
#             end
#             A.sig[mm] *= fact2
#         end

#         # Orthogonalize v-vectors
#         for j in 1:rank

#             # Pivoting: find index for the largest sig
#             smax = A.sig[ipivot[j]]
#             kmax = j
#             for k in j+1:rank
#                 if smax < A.sig[ipivot[k]]
#                     kmax = k
#                     smax = A.sig[ipivot[k]]
#                 end
#             end

#             # Swap into the leading position
#             ipivot[j],ipivot[kmax] = ipivot[kmax],ipivot[j]

#             mm = ipivot[j]

#             # Check if sig are large enough
#             # if they are smaller than a threshold,
#             # then reset to the rank (run out of pivots)
#             if A.sig[mm] <= atol2
#                 rank = j-1
#                 break
#             end

#             for l in j+1:rank

#                 ll = ipivot[l]

#                 # if sig is smaller than a threshold, set sig to zero
#                 # to be ignored from now on
#                 if A.sig[ll] <= atol2
#                     A.sig[ll] = 0.
#                     break
#                 end

#                 # Modify v-vectors
#                 dalpha = innerprod(A.v[mm],A.v[ll])
#                 A.v[ll] -= dalpha*A.v[mm]
#                 fact1 = sqrt(innerprod(A.v[ll],A.v[ll]))
#                 if fact1*A.sig[ll] > atol2
#                     A.v[ll] /= fact1
#                 end

#                 # Modify u-vectors
#                 dalpha *= A.sig[ll]/A.sig[mm]
#                 A.u[mm] += dalpha*A.u[ll]

#                 # Adjust sig
#                 A.sig[ll] *= fact1

#             end

#             # Adjust sig of the pivot vector
#             fact2 = sqrt(innerprod(A.u[mm],A.u[mm]))
#             if fact2*A.sig[mm] > atol2
#                 A.u[mm] /= fact2
#             end
#             A.sig[mm] *= fact2
#         end

#         println("reduce to $(rank)")

#         if rank == rankini
#             break
#         end

#     end


#     # Fill in gaps, if any
#     # Find all pivots less than the new rank with actual location beyond rank
#     # these vectors and sig need to be swapped to location within the range 1<=...<=rank
#     iempty = Vector{Int}(undef,A.rank)

#     icount = 1
#     for l in rank+1:A.rank
#         if ipivot[l] <= rank
#             iempty[icount] = ipivot[l]
#             icount += 1
#         end
#     end

#     # Find all pivots larger than the new rank, These lications are overwritten
#     # by the vectors and sig values within the range 1<=..<=rank found above
#     icount = 1
#     for l = 1:rank
#         if ipivot[l] > rank
#             A.u[iempty[icount]] = copy(A.u[ipivot[l]])
#             A.v[iempty[icount]] = copy(A.v[ipivot[l]])
#             A.sig[iempty[icount]] = A.sig[ipivot[l]]
#             ipivot[l] = iempty[icount]
#             icount += 1
#         end
#     end

#     # Reset the rank
#     A.rank = rank

#     # Remove small values
#     for i in 1:A.rank

#         droptol!(A.u[i],atol2/A.sig[i])
#         droptol!(A.v[i],atol2/A.sig[i])

#         # dropband!(A.u[i],atol2/A.sig[i])
#         # dropband!(A.v[i],atol2/A.sig[i])
#     end
#     normalize(A,atol2)

    
#     println("Done reduction")
#     return
# end


# """
# Reduce the number of terms in CanTenDec using Cholesky decomposition.
# """
# function fastchol(A::CanTenDec;rankmax::Union{Int,Nothing}=nothing,atol::Float64=1e-7)::Nothing
#     println("initial number of terms $(A.rank)")

#     if A.rank == 0
#         return
#     end

#     if rankmax === nothing || rankmax > A.rank
#         rankmax = A.rank
#     end

#     atol2 = atol*atol
#     if atol2 < 1e-15
#         atol2 = 1e-15
#         @warn "atol2 has been reset to 1e-15"
#     end

#     diag = ones(Float64,A.rank)
#     ipivot = collect(1:A.rank)
#     L = Matrix{Float64}(undef,rankmax,A.rank)

#     rank = 0
#     for i in 1:rankmax
        
#         # find the largest diagonal
#         dmax = diag[ipivot[i]]
#         imax = i
#         for j in i+1:A.rank
#             if dmax < diag[ipivot[j]]
#                 dmax = diag[ipivot[j]]
#                 imax = j
#             end
#         end

#         # swap to the leading position
#         ipivot[i],ipivot[imax] = ipivot[imax],ipivot[i]

#         # check if diagonal is large enough
#         if diag[ipivot[i]] < atol2
#             break
#         end

#         L[i,ipivot[i]] = sqrt(diag[ipivot[i]])
#         for j in i+1:A.rank
#             t1 = dot(L[1:i-1,ipivot[i]],L[1:i-1,ipivot[j]])
#             t2 = innerprod(A.u[ipivot[i]],A.u[ipivot[j]])*
#                 innerprod(A.v[ipivot[i]],A.v[ipivot[j]])
#             L[i,ipivot[j]] = (t2-t1)/L[i,ipivot[i]]
#             diag[ipivot[j]] = diag[ipivot[j]]-L[i,ipivot[j]]^2
#         end # j loop

#         rank += 1

#     end # i loop

#     # form rhs for solving new s-values
#     newsig = zeros(Float64,rank)
#     for i in 1:rank
#         for j in rank+1:A.rank
#             newsig[i] += A.sig[ipivot[j]]*dot(L[1:i,ipivot[i]],L[1:i,ipivot[j]])
#         end
#     end

#     # forward substitution
#     for i in 1:rank
#         t = 0.
#         for j in 1:i-1
#             t += L[j,ipivot[i]]*newsig[j]
#         end
#         newsig[i] = (newsig[i]-t)/L[i,ipivot[i]]
#     end

#     # backward substitution
#     for i in rank:-1:1
#         t = 0.
#         for j in i+1:rank
#             t += L[i,ipivot[j]]*newsig[j]
#         end
#         newsig[i] = (newsig[i]-t)/L[i,ipivot[i]]
#     end

#     # add `skeleton` coefficients
#     for i in 1:rank
#         newsig[i] += A.sig[ipivot[i]]
#     end
    
#     newu = Vector{SparseMatrixCSC{Float64,Int}}(undef,rank)
#     newv = Vector{SparseMatrixCSC{Float64,Int}}(undef,rank)
    
#     # copy data, modify u-vectors to keep sig positive
#     for i in 1:rank
#         if newsig[i] > 0.
#             newsig[i] = newsig[i]
#             newu[i] = A.u[ipivot[i]]
#             newv[i] = A.v[ipivot[i]]
#         else
#             newsig[i] = -newsig[i]
#             newu[i] = -A.u[ipivot[i]]
#             newv[i] = A.v[ipivot[i]]
#         end
#     end
    
#     A.capacity = rank
#     A.rank = rank
#     A.sig = newsig[1:rank]
#     A.u = newu
#     A.v = newv

#     println("reduced number of terms is $(A.rank)")
    
#     # Success
#     return
# end

"""
Reduce the rank of A by keeping the largest ``rank`` s-values
"""
function droprank(A::CanTenDec,rank::Int64)::Nothing

    if A.rank <= rank
        return
    end

    ipivot = sortperm(A.sig[1:A.rank],rev=true)[1:rank]
    sig = Vector{Float64}(undef,rank)
    u = Vector{SparseMatrixCSC{Float64,Int}}(undef,rank)
    v = Vector{SparseMatrixCSC{Float64,Int}}(undef,rank)
    for i in 1:rank
        sig[i] = A.sig[ipivot[i]]
        u[i] = copy(A.u[ipivot[i]])
        v[i] = copy(A.v[ipivot[i]])
    end

    A.sig = sig
    A.u = u
    A.v = v
    A.rank = rank
    A.capacity = rank

    return
end

function droptol(A::CanTenDec,atol::Float64)::Nothing

    ipivot = sortperm(A.sig[1:A.rank],rev=true)
    sig = Vector{Float64}(undef,A.rank)
    u = Vector{SparseMatrixCSC{Float64,Int}}(undef,A.rank)
    v = Vector{SparseMatrixCSC{Float64,Int}}(undef,A.rank)

    rank = 0
    for i in 1:A.rank
        if A.sig[ipivot[i]] >= atol
            sig[i] = A.sig[ipivot[i]]
            u[i] = copy(A.u[ipivot[i]])
            v[i] = copy(A.v[ipivot[i]])
            rank += 1
        else
            break
        end
    end

    A.sig = sig
    A.u = u
    A.v = v
    A.rank = rank
    A.capacity = A.rank

    return
end