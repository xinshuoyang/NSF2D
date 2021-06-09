mutable struct MatA
    size::Int
    values::Matrix{CanTenDec}
end
copy(A::MatA) = MatA(A.size,reshape([copy(A.values[j,i]) for i in 1:3 for j in 1:3],(3,3)))


function to_dense(A::MatA)::Matrix{Float64}
    d = Matrix{Float64}(undef,3*A.size^2,3*A.size^2)
    for i in 1:3, j in 1:3
        d[(j-1)*A.size^2+1:j*A.size^2,(i-1)*A.size^2+1:i*A.size^2] = to_dense(A.values[j,i])
    end
    return d
end

mutable struct MatB
    size::Int
    values::Vector{CanTenDec}
end
copy(B::MatB) = MatB(B.size,[copy(B.values[i]) for i in 1:3])

function to_dense(B::MatB)::Matrix{Float64}
    d = Matrix{Float64}(undef,B.size^2,3*B.size^2)
    for i in 1:3
        d[1:B.size^2,(i-1)*B.size^2+1:i*B.size^2] = to_dense(B.values[i])
    end
    return d
end

mutable struct MatC
    size::Int
    values::Vector{CanTenDec}
end
copy(C::MatC) = MatC(C.size,[copy(C.values[i]) for i in 1:3])

function to_dense(C::MatC)::Matrix{Float64}
    d = Matrix{Float64}(undef,3*C.size^2,C.size^2)
    for i in 1:3
        d[(i-1)*C.size^2+1:i*C.size^2,1:C.size^2] = to_dense(C.values[i])
    end
    return d
end

function mul(A::MatA,B::MatA,atol::Float64)::MatA
    if A.size != B.size
        throw("sizes of A and B not compatible")
    end

    C = MatA(A.size,Matrix{CanTenDec}(undef,3,3))

    # Distribute error threshold
    atol2 = atol/9.0
    # atol2 = atol

    for i in 1:3, j in 1:3
        C.values[i,j] = create_zeros(A.size)
        for k in 1:3
            axpy(1.,mul(A.values[i,k],B.values[k,j],atol2),C.values[i,j],atol2)
            reduce(C.values[i,j],atol2)
            # reduce_abs(C.values[i,j],atol2)
        end
    end
    return C
end

function mul(A::MatA,B::MatC,atol::Float64)::MatC
    if A.size != B.size
        throw("sizes of A and B not compatible")
    end

    C = MatC(A.size,Vector{CanTenDec}(undef,3))

    # Distribute error threshold
    atol2 = atol/3.0
    # atol2 = atol

    for i in 1:3
        C.values[i] = create_zeros(A.size)
    end

    for i in 1:3
        for j in 1:3
            axpy(1.,mul(A.values[i,j],B.values[j],atol2),C.values[i],atol2)
            reduce(C.values[i],atol2)
            # reduce_abs(C.values[i],atol2)
        end
    end
    return C
end

function mul(A::MatB,B::MatA,atol::Float64)::MatB
    if A.size != B.size
        throw("sizes of A and B not compatible")
    end

    C = MatB(A.size,Vector{CanTenDec}(undef,3))

    # Distribute error threshold
    atol2 = atol/3.0
    # atol2 = atol

    for i in 1:3
        C.values[i] = create_zeros(A.size)
    end

    for i in 1:3
        for j in 1:3
            axpy(1.,mul(A.values[j],B.values[j,i],atol2),C.values[i],atol2)
            reduce(C.values[i],atol2)
            # reduce_abs(C.values[i],atol2)
        end
    end

    return C
end

function mul(A::MatB,B::MatC,atol::Float64)::CanTenDec
    if A.size != B.size
        throw("sizes of A and B not compatible")
    end

    C = create_zeros(A.size)
    for i in 1:3
        axpy(1.,mul(A.values[i],B.values[i],atol),C,atol)
        reduce(C,atol)
        # reduce_abs(C,atol)
    end

    return C
end


mutable struct NSFMatCoefs
    size::Int
    A::MatA
    B::MatB
    C::MatC
    T::CanTenDec
    NSFMatCoefs(size) = new(size)
    NSFMatCoefs(size,A,B,C,T) = new(size,A,B,C,T)
end

function to_dense(X::NSFMatCoefs)::Matrix{Float64}
    d = Matrix{Float64}(undef,4*X.size^2,4*X.size^2)
    d[1:3*X.size^2,1:3*X.size^2] = to_dense(X.A)
    d[3*X.size^2+1:4*X.size^2,1:3*X.size^2] = to_dense(X.B)
    d[1:3*X.size^2,3*X.size^2+1:4*X.size^2] = to_dense(X.C)
    d[3*X.size^2+1:4*X.size^2,3*X.size^2+1:4*X.size^2] = to_dense(X.T)
    return d
end

function get_rank(X::NSFMatCoefs)::Matrix{Int}
    ranks = Matrix{Int}(undef,4,4)
    for i in 1:3, j in 1:3
        ranks[j,i] = X.A.values[j,i].rank
    end
    for i in 1:3
        ranks[4,i] = X.B.values[i].rank
        ranks[i,4] = X.C.values[i].rank
    end
    ranks[4,4] = X.T.rank
    return ranks
end


function get_NSFMatCoefs(X::CanTenDec,m::Int,H::Vector{Float64},G::Vector{Float64},atol::Float64)::NSFMatCoefs

    # Distribute error threshold
    # atol2 = atol/16.0
    atol2 = atol/1e1
    println("atol2 = $(atol2)")

    ut = [dwtmat(X.size,X.u[i],m,H,G) for i in 1:X.rank]
    vt = [dwtmat(X.size,X.v[i],m,H,G) for i in 1:X.rank]

    index = Dict((1,1)=>1,(2,1)=>2,(1,2)=>3,(2,2)=>4)

    # Create 4-by-4 block matrix
    mat = Matrix{CanTenDec}(undef,4,4)
    for i in 1:2, j in 1:2, k in 1:2, l in 1:2
        mat[2*(k-1)+l,2*(i-1)+j] = CanTenDec(X.size÷2,X.rank,X.rank,copy(X.sig),
            [ut[p][index[(k,i)]] for p in 1:X.rank],[vt[p][index[(l,j)]] for p in 1:X.rank])
        normalize(mat[2*(k-1)+l,2*(i-1)+j],atol2)
    end

    # # Reduce rank of mat[1,1], then reduce ranks in other blocks wrt the largest s-value of mat[1,1]
    # reduce(mat[1,1],atol2)
    # # reduce_abs(mat[1,1],atol2)
    # maxsval = maximum(mat[1,1].sig[1:mat[1,1].rank])
    # for i in 1:4, j in 1:4
    #     if i != 1 || j != 1
    #         reduce(mat[i,j],atol2,maxsig=maxsval)
    #         # reduce_abs(mat[i,j],atol2)
    #     end
    # end

    A = MatA(X.size÷2,Matrix{CanTenDec}(undef,3,3))
    for i in 1:3, j in 1:3
        A.values[j,i] = mat[j,i]
    end
    B = MatB(X.size÷2,[mat[4,i] for i in 1:3])
    C = MatC(X.size÷2,[mat[i,4] for i in 1:3])
    T = mat[4,4]

    Y = NSFMatCoefs(X.size÷2,A,B,C,T)
end

mutable struct NSFMatrix
    size::Int
    moments::Int
    scales::Int
    coefs::Vector{NSFMatCoefs}
end


# """
# Compute inverse of MatA
# """
# function get_inv_MatA(A::MatA,atol::Float64;max_iter=10)::MatA

#     # Adense = to_dense(A)
#     # println("is A is_diagally_dominant? ",is_diagally_dominant(Adense))

#     atol2 = atol/9
#     # atol2 = atol
#     # DA =  [get_inv_diag(A.values[i,i],atol2) for i in 1:3]
#     DA =  [get_inv_diag(A.values[i,i],1e-5) for i in 1:3]

#     atol2 = atol/9
#     # atol2 = atol
#     T = MatA(A.size,Matrix{CanTenDec}(undef,3,3))
#     for i in 1:3, j in 1:3
#         if i == j
#             T.values[j,i] = mul(DA[j],get_offdiag(A.values[j,i]),atol2)
#             scale(T.values[j,i],-1.)
#         else
#             T.values[j,i] = mul(DA[j],A.values[j,i],atol2)
#             scale(T.values[j,i],-1.)
#         end
#     end

#     B = copy(T)
#     for i in 1:3
#         aipy(1.,B.values[i,i])
#     end

#     T = mul(T,T,atol)
#     est = sum(sum(T.values[j,i].sig[1:T.values[j,i].rank]) for i in 1:3, j in 1:3)
#     println("est = ", est)
#     for _ in 1:max_iter
#         if est < atol
#             @goto converged
#         else
#             S = copy(T)
#             for i in 1:3
#                 aipy(1.0,S.values[i,i])
#             end
#             B = mul(B,S,atol2)
#             T = mul(T,T,atol2)
#             est = sum(sum(T.values[j,i].sig[1:T.values[j,i].rank]) for i in 1:3, j in 1:3)
#             println("est = ",est)
#         end
#     end
#     @warn "maximum iterations exceeded!"
#     @label converged

#     for i in 1:3, j in 1:3
#         B.values[j,i] = mul(B.values[j,i],DA[i],atol2)
#         reduce(B.values[j,i],atol2)
#         # fastchol(B.values[j,i],atol=atol2)
#     end

#     return B
# end


"""
Compute inverse of MatA via Schulz iterations
"""
function invMatA(A::NSF2D.MatA,atol::Float64;max_iter=50)::Union{Nothing,NSF2D.MatA}
    # Select α
    # atol2 = min(1e-3,atol*1e2)
    atol2 = 1e-4
    α = 1/snorm(A,atol2)

    # Initialize 
    G0 = copy(A)
    for i in 1:3, j in 1:3
        scale(G0.values[i,j],α)
    end 

    atol2 = 1e-1
    err = 1.0

    # Main iteration
    for _ in 1:max_iter
        
        println("atol2 = $(atol2)")

        # Check for termination
        # err_ = maximum(norm(diagvec(GA.values[j,j]).-1.0,Inf) for j in 1:3)
        err_ = max(maximum([norm(sum(diagmul(G0.values[i,j],A.values[j,i]) for j in 1:3).-1.0,Inf) for i in 1:3]),
            maximum([norm(sum(diagmul(A.values[i,j],G0.values[j,i]) for j in 1:3).-1.0,Inf) for i in 1:3]))
        # err_ = opnorm(to_dense(GA)-Matrix(1.0I,3*A.size^2,3*A.size^2),2)
        
        println("invMatA err is $(err_)")
        
        if err_ < atol
            println("reach asked error threshold $(atol), return")
            return G0
        end
            
        println("err reduction is $(abs(err_-err)/err)")
        atol2 = min(atol2,err_/10)
        if err_ >= err || abs(err_-err) <= 0.5*err
            atol2 /= 2
        end
        if err_ < 1e-3
            atol2 = max(err_*err_,atol/1e2)
        end

        err = err_
        
        # 
        B = mul(mul(G0,A,atol2),G0,atol2)

        for i in 1:3, j in 1:3
            scale(G0.values[i,j],2.0)
        end

        for i in 1:3, j in 1:3
            axpy(-1.0,B.values[i,j],G0.values[i,j],atol2)
            reduce(G0.values[i,j],atol2)
        end
    end

    # No convergence
    @warn("invMatA maximum number of iteration reached")
end

function invMatAr(A::NSF2D.MatA,atol::Float64,rank::Matrix{Int64};max_iter=50)::Union{Nothing,NSF2D.MatA}
    if size(rank) != (3,3)
        throw("expected a rank matrix of size (3,3), got $(size(rank))")
    end

    # Select α
    # atol2 = min(1e-3,atol*1e2)
    atol2 = 1e-4
    α = 1/snorm(A,atol2)

    # Initialize 
    G0 = copy(A)
    for i in 1:3, j in 1:3
        scale(G0.values[i,j],α)
    end 

    atol2 = 1e-1
    err = 1.0

    # Main iteration
    for _ in 1:max_iter
        
        println("atol2 = $(atol2)")

        # Check for termination
        # Compute maximum diagonal of I-G0*A and check for termination
        err_ = max(maximum([norm(sum(diagmul(G0.values[i,j],A.values[j,i]) for j in 1:3).-1.0,Inf) for i in 1:3]),
            maximum([norm(sum(diagmul(A.values[i,j],G0.values[j,i]) for j in 1:3).-1.0,Inf) for i in 1:3]))
        # err_ = maximum(norm(diagvec(GA.values[j,j]).-1.0,Inf) for j in 1:3)
        # err_ = opnorm(to_dense(GA)-Matrix(1.0I,3*A.size^2,3*A.size^2),2)
        
        println("invMatA err is $(err_)")
        
        if err_ < atol
            println("reach asked error threshold $(atol), return")
            return G0
        end
            
        println("err reduction is $(abs(err_-err)/err)")
        atol2 = min(atol2,err_/2)
        if err_ >= err || abs(err_-err) <= 0.5*err
            atol2 /= 2
        end
        err = err_
        
        # 
        B = mul(mul(G0,A,atol2),G0,atol2)

        for i in 1:3, j in 1:3
            scale(G0.values[i,j],2.0)
        end

        for i in 1:3, j in 1:3
            axpy(-1.0,B.values[i,j],G0.values[i,j],atol2)
            reduce(G0.values[i,j],atol2)
            droprank(G0.values[i,j],rank[i,j])
        end
    end

    # No convergence
    @warn("invMatAr maximum number of iteration reached")
end

"""
Second version if invMatAr
"""
function invMatAr2(A::NSF2D.MatA,atol::Float64;max_iter=50)::Union{Nothing,NSF2D.MatA}
    # Select α
    # atol2 = min(1e-3,atol*1e2)
    atol2 = 1e-4
    α = 1/snorm(A,atol2)

    # Initialize 
    G0 = copy(A)
    for i in 1:3, j in 1:3
        scale(G0.values[i,j],α)
    end 

    atol2 = 1e-1
    err = 1.0

    # Main iteration
    for _ in 1:max_iter
        
        println("atol2 = $(atol2)")

        # Compute maximum diagonal of I-G0*A and check for termination
        @time err_ = max(maximum([norm(sum(diagmul(G0.values[i,j],A.values[j,i]) for j in 1:3).-1.0,Inf) for i in 1:3]),
            maximum([norm(sum(diagmul(A.values[i,j],G0.values[j,i]) for j in 1:3).-1.0,Inf) for i in 1:3]))
        # err_ = opnorm(to_dense(GA)-Matrix(1.0I,3*A.size^2,3*A.size^2),2)
        
        println("invMatA err is $(err_)")
        
        if err_ < atol
            println("reach asked error threshold $(atol), return")
            return G0
        end
            
        println("err reduction is $(abs(err_-err)/err)")
        atol2 = min(atol2,err_/2)
        if err_ >= err || abs(err_-err) <= 0.5*err
            atol2 /= 2
        end
        err = err_
        
        # Compute diagonal blocks of 2*G0-G0*A*G0
        B = MatA(A.size,Matrix{CanTenDec}(undef,3,3))
        for i in 1:3
            B.values[i,i] = copy(G0.values[i,i])
            scale(B.values[i,i],2.0)
            for k in 1:3
                for l in 1:3
                    axpy(-1.0,mul(mul(G0.values[i,k],A.values[k,l],atol2),G0.values[l,i],atol2),B.values[i,i],atol2)
                end
            end
            reduce(B.values[i,i],atol2)
        end

        # Compute off-diagonal blocks of G0*A*G0
        # Use maximum diagonal rank for truncation
        rank = maximum([B.values[i,i].rank for i in 1:3])
        println("invMatA: maximim rank is $(rank)")
        for i in 1:3
            for j in 1:3
                B.values[i,j] = copy(G0.values[i,j])
                scale(B.values[i,j],2.0)
                for k in 1:3
                    for l in 1:3
                        axpy(-1.0,mul(mul(G0.values[i,k],A.values[k,l],atol2),G0.values[l,j],atol2),B.values[i,j],atol2)
                    end
                end
                reduce(B.values[i,j],atol2)
                droprank(B.values[i,j],rank)
            end
        end

        for i in 1:3
            for j in 1:3
                G0.values[i,j] = copy(B.values[i,j])
            end
        end
    end

    # No convergence
    @warn("invMatAr2 maximum number of iteration reached")
end

# """
# Estimate the spectral norm (square root of the largest eigenvalue of A^H*A) of MatA using 
# random power iteration
# """
function snorm(A::MatA,atol::Float64;max_iter::Int=500)::Union{Float64,Nothing}
    x = [rand(A.size,A.size),rand(A.size,A.size),rand(A.size,A.size)]
    xnorm = norm(x)
    s = xnorm
    
    for _ in 1:max_iter
        x /= xnorm
        s_ = s
        
        x = [sum(apply(A.values[i,j],x[j]) for j in 1:3) for i in 1:3]
        x = [sum(apply(A.values[i,j],x[j],TRANS="T") for j in 1:3) for i in 1:3]
        
        xnorm = norm(x)
        s = xnorm
                
        if xnorm <= atol || abs(s-s_) <= s_*atol
            return s
        end

    end
        
    # No convergence
    @warn("snorm maximum number of iteration reached")
end


function factorize(X::NSFMatCoefs,atol::Float64)::NSFMatCoefs
    Y = NSFMatCoefs(X.size)

    # atol2 = 9/16*atol
    atol2 = atol
    # Y.A = get_inv_MatA(X.A,atol2)
    Y.A = invMatA(X.A,atol2)

    Y.B = copy(X.B)

    # atol2 = 3/16*atol
    atol2 = atol
    Y.C = mul(Y.A,X.C,atol2)
    # for j in 1:3
    #     reduce(Y.C.values[j],atol2)
    # end

    Y.T = copy(X.T)
    # atol2 = atol/16
    atol2 = atol
    axpy(-1.,mul(X.B,Y.C,atol2),Y.T,atol2)
    reduce(Y.T,atol2)
    # reduce_abs(Y.T,atol2)

    return Y
end


function factorize2(X::NSFMatCoefs,atol::Float64,rank::Matrix{Int64})::NSFMatCoefs
    Y = NSFMatCoefs(X.size)

    # atol2 = 9/16*atol
    atol2 = atol
    # Y.A = get_inv_MatA(X.A,atol2)
    # Y.A = invMatA(X.A,atol2)
    Y.A = invMatAr(X.A,atol2,3*rank[1:3,1:3])
    # Y.A = invMatAr2(X.A,atol2)
    # for i in 1:3
    #     for j in 1:3
    #         droprank(Y.A.values[i,j],rank[i,j])
    #     end
    # end

    Y.B = copy(X.B)
    # for i in 1:3
    #     droprank(Y.B.values[i],rank[4,i])
    # end

    # atol2 = 3/16*atol
    atol2 = atol
    Y.C = mul(Y.A,X.C,atol2)
    # for i in 1:3
    #     droprank(Y.C.values[i],rank[i,4])
    # end

    Y.T = copy(X.T)
    # atol2 = atol/16
    atol2 = atol
    axpy(-1.,mul(X.B,Y.C,atol2),Y.T,atol2)
    reduce(Y.T,atol2)
    # droprank(Y.T,rank[4,4])

    return Y
end


function get_NSFMatrix(X::CanTenDec,
        moments::Int,
        scales::Int,
        atol::Float64,
        )::NSFMatrix

    atol2 = atol

    H,G = get_qmf(moments)
    Y = NSFMatrix(X.size,moments,scales,Vector{NSFMatCoefs}(undef,scales))
    T = X
    for j in 1:scales
        Y.coefs[j] = get_NSFMatCoefs(T,moments,H,G,atol2)
        T = Y.coefs[j].T
        atol2 /= 16.
    end
    return Y
end

# function get_NSFMatrix_decomp(X::CanTenDec,
#         moments::Int,
#         scales::Int,
#         atol::Float64,
#         )::NSFMatrix
#     atol2 = atol
#     H,G = get_qmf(moments)
#     Y = NSFMatrix(X.size,moments,scales,Vector{NSFMatCoefs}(undef,scales))
#     T = X

#     # At first scale compute decomposition with truncation of s-values
#     println("construct decomposition of scale 1")
#     println("rank of T is $(T.rank)")
#     println("getting NSFMatCoefs ")
#     @time coefs = get_NSFMatCoefs(T,moments,H,G,atol2)
#     println("done")
#     println("compute factorization")
#     @time Y.coefs[1] = factorize(coefs,atol2)
#     println("done")
#     T = Y.coefs[1].T

#     rank = get_rank(Y.coefs[1])
#     atol2 *= 2

#     # Start from second scale, compute decomposition with truncation of ranks
#     for j in 2:scales
#         println("construct decomposition of scale $(j)")
#         println("rank of T is $(T.rank)")
#         println("getting NSFMatCoefs ")
#         @time coefs = get_NSFMatCoefs(T,moments,H,G,atol2)
#         println("done")

#         println("compute factorization")
#         @time Y.coefs[j] = factorize2(coefs,atol2,rank)
#         println("done")
#         T = Y.coefs[j].T
#         # atol2 *= 2
#         # atol2 /= 16.
#         # atol2 /= 4
        
#     end
#     return Y
# end


function get_NSFMatrix_decomp(X::CanTenDec,
        moments::Int,
        scales::Int,
        atol::Float64,
        )::NSFMatrix
    atol2 = atol
    H,G = get_qmf(moments)
    Y = NSFMatrix(X.size,moments,scales,Vector{NSFMatCoefs}(undef,scales))
    T = X

    # Start from second scale, compute decomposition with truncation of ranks
    for j in 1:scales

        # if j==scales
        #     atol2 /= 100.
        # end

        println("construct decomposition of scale $(j)")
        println("rank of T is $(T.rank)")
        println("getting NSFMatCoefs ")
        @time coefs = get_NSFMatCoefs(T,moments,H,G,atol2)
        println("done")

        println("compute factorization")
        @time Y.coefs[j] = factorize(coefs,atol2)
        println("done")
        T = Y.coefs[j].T
        # atol2 *= 2
        # atol2 /= 2.
        atol2 /= 4.

    end
    return Y
end

"""
Apply
"""
function apply(A::NSFMatrix,x::NSFVector)::NSFVector
    throw("unimplemented")
end

"""
Forward-backward solve M * x = b
"""
function fbsub(M::NSFMatrix,
    b::NSFVector,
    atol::Float64,
    )::NSFVector

    println("fbsub using atol = $(atol)")

    if M.size != b.size || M.moments != b.moments || M.scales != b.scales
        throw(DimensionMismatch("size of M not compatible with size of b"))
    end

    n = M.size
    moments = M.moments
    scales = M.scales

    H,G = get_qmf(moments)

    println("forward sub....")

    # Forward substitution L * y = b
    y = NSFVector(n,moments,scales,Vector{NSFVecCoefs}(undef,scales))
    n ÷= 2
    proj = NSFVecCoefs(zeros(Float64,n,n),
        zeros(Float64,n,n),
        zeros(Float64,n,n),
        zeros(Float64,n,n))

    for j in 1:scales
        
        y.coefs[j] = NSFVecCoefs()

        rhs = [b.coefs[j].A-proj.A,b.coefs[j].B-proj.B,b.coefs[j].C-proj.C]
        
        y.coefs[j].A = apply(M.coefs[j].A.values[1,1],rhs[1])+
            apply(M.coefs[j].A.values[1,2],rhs[2])+
            apply(M.coefs[j].A.values[1,3],rhs[3])

        y.coefs[j].B = apply(M.coefs[j].A.values[2,1],rhs[1])+
            apply(M.coefs[j].A.values[2,2],rhs[2])+
            apply(M.coefs[j].A.values[2,3],rhs[3])

        y.coefs[j].C = apply(M.coefs[j].A.values[3,1],rhs[1])+
            apply(M.coefs[j].A.values[3,2],rhs[2])+
            apply(M.coefs[j].A.values[3,3],rhs[3])

        if j < scales
            coefs = apply(M.coefs[j].B.values[1],y.coefs[j].A)+
                    apply(M.coefs[j].B.values[2],y.coefs[j].B)+
                    apply(M.coefs[j].B.values[3],y.coefs[j].C)+
                    proj.T
            A,B,C,T = dwtmat(n,coefs,moments,H,G)
            proj = NSFVecCoefs(A,B,C,T)
            n ÷= 2
        end
    end

    # Solve the last scale
    rhs = b.coefs[scales].T-proj.T-
        apply(M.coefs[scales].B.values[1],y.coefs[scales].A)-
        apply(M.coefs[scales].B.values[2],y.coefs[scales].B)-
        apply(M.coefs[scales].B.values[3],y.coefs[scales].C)
    y.coefs[scales].T = solve(M.coefs[scales].T,rhs,atol)

    println("backward sub....")

    # Backward substitution U * x = y
    x = NSFVector(M.size,moments,scales,Vector{NSFVecCoefs}(undef,scales))
    x.coefs[scales] = NSFVecCoefs()
    x.coefs[scales].T = y.coefs[scales].T
    for j in scales:-1:1
        x.coefs[j].A = y.coefs[j].A-apply(M.coefs[j].C.values[1],x.coefs[j].T)
        x.coefs[j].B = y.coefs[j].B-apply(M.coefs[j].C.values[2],x.coefs[j].T)
        x.coefs[j].C = y.coefs[j].C-apply(M.coefs[j].C.values[3],x.coefs[j].T)

        if j > 1
            x.coefs[j-1] = NSFVecCoefs()
            x.coefs[j-1].T = idwtmat(n,moments,H,G,x.coefs[j].A,x.coefs[j].B,x.coefs[j].C,x.coefs[j].T)
            n *= 2
        end
    end

    return x
end


