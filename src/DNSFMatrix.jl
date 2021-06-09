mutable struct DNSFMatCoefs
    size::Int
    A::Matrix{Float64}
    B::Matrix{Float64}
    C::Matrix{Float64}
    T::Matrix{Float64}
    DNSFMatCoefs(size) = new(size)
    DNSFMatCoefs(size,A,B,C,T) = new(size,A,B,C,T)
end


function get_DNSFMatCoefs(n::Int,
    X::Matrix{Float64},
    m::Int,
    H::Vector{Float64},
    G::Vector{Float64}
    )::DNSFMatCoefs

    A,B,C,T = dwtmat2d(n,X,m,H,G)

    return DNSFMatCoefs(n,A,B,C,T)

end


mutable struct DNSFMatrix
    size::Int
    moments::Int
    scales::Int
    coefs::Vector{DNSFMatCoefs}
end

function snorm(A::Matrix{Float64},
    atol::Float64;
    max_iter::Int=500
    )::Union{Float64,Nothing}

    x = rand(size(A)[1])
    xnorm = norm(x)
    s = xnorm
    
    for _ in 1:max_iter
        x /= xnorm
        s_ = s
        
        x = A*x
        x = transpose(A)*x
        
        xnorm = norm(x)
        s = xnorm
                
        if xnorm <= atol || abs(s-s_) <= s_*atol
            return s
        end

    end
        
    # No convergence
    @warn("snorm maximum number of iteration reached")
end

function schulz(A::Matrix{Float64},
    atol::Float64;
    max_iter=50
    )::Matrix{Float64}

    eps = 1e-4
    α = 1/snorm(A,eps)
    G0 = α*copy(transpose(A))
    
    for _ in 1:max_iter
        err = max(maximum([1-dot(G0[i,:],A[:,i]) for i in 1:size(A)[1]]),
                maximum([1-dot(A[i,:],G0[:,i]) for i in 1:size(A)[1]]))
        
        println(err)
        
        if err < atol
            break
        end
        G0 = 2*G0-G0*A*G0 
    end
    return G0
end

function factorize(X::DNSFMatCoefs,
    atol::Float64
    )::DNSFMatCoefs

    Y = DNSFMatCoefs(X.size)

    # Y.A = inv(X.A)
    Y.A = schulz(X.A,atol)

    Y.B = copy(X.B)

    Y.C = Y.A*X.C

    Y.T = X.T-X.B*Y.C

    return Y
end


function get_DNSFMatrix(n::Int,
    X::Matrix{Float64},
    moments::Int,
    scales::Int,
    )::DNSFMatrix

    H,G = get_qmf(moments)
    Y = DNSFMatrix(n,moments,scales,Vector{DNSFMatCoefs}(undef,scales))
    T = X
    for j in 1:scales
        Y.coefs[j] = get_DNSFMatCoefs(n,T,moments,H,G)
        T = Y.coefs[j].T
        n ÷= 2
    end
    return Y
end



function get_DNSFMatrix_decomp(n::Int,
    X::Matrix{Float64},
    moments::Int,
    scales::Int,
    atol::Float64,
    )::DNSFMatrix

    H,G = get_qmf(moments)
    Y = DNSFMatrix(n,moments,scales,Vector{DNSFMatCoefs}(undef,scales))
    T = X
    for j in 1:scales
        coefs = get_DNSFMatCoefs(n,T,moments,H,G)
        Y.coefs[j] = factorize(coefs,atol)
        T = Y.coefs[j].T
        n ÷= 2
        atol /= 2
    end
    return Y
end

"""
Forward-backward solve M * x = b
"""
function solve(M::DNSFMatrix,b::NSFVector)::NSFVector

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

        println("size of A block is $(size(M.coefs[j].A))")

        y.coefs[j].A = reshape(M.coefs[j].A[1:n^2,1:n^2]*reshape(rhs[1],(n^2,1)),(n,n))+
        reshape(M.coefs[j].A[1:n^2,n^2+1:2*n^2]*reshape(rhs[2],(n^2,1)),(n,n))+
        reshape(M.coefs[j].A[1:n^2,2*n^2+1:3*n^2]*reshape(rhs[3],(n^2,1)),(n,n))

        y.coefs[j].B = reshape(M.coefs[j].A[n^2+1:2*n^2,1:n^2]*reshape(rhs[1],(n^2,1)),(n,n))+
            reshape(M.coefs[j].A[n^2+1:2*n^2,n^2+1:2*n^2]*reshape(rhs[2],(n^2,1)),(n,n))+
            reshape(M.coefs[j].A[n^2+1:2*n^2,2*n^2+1:3*n^2]*reshape(rhs[3],(n^2,1)),(n,n))

        y.coefs[j].C = reshape(M.coefs[j].A[2*n^2+1:3*n^2,1:n^2]*reshape(rhs[1],(n^2,1)),(n,n))+
            reshape(M.coefs[j].A[2*n^2+1:3*n^2,n^2+1:2*n^2]*reshape(rhs[2],(n^2,1)),(n,n))+
            reshape(M.coefs[j].A[2*n^2+1:3*n^2,2*n^2+1:3*n^2]*reshape(rhs[3],(n^2,1)),(n,n))

        if j < scales
            coefs = reshape(M.coefs[j].B[:,1:n^2]*reshape(y.coefs[j].A,(n^2,1)),(n,n))+
                    reshape(M.coefs[j].B[:,n^2+1:2*n^2]*reshape(y.coefs[j].B,(n^2,1)),(n,n))+
                    reshape(M.coefs[j].B[:,2*n^2+1:3*n^2]*reshape(y.coefs[j].C,(n^2,1)),(n,n))+
                    proj.T
            A,B,C,T = dwtmat(n,coefs,moments,H,G)
            proj = NSFVecCoefs(A,B,C,T)
            n ÷= 2
        end
    end

    # Solve the last scale
    rhs = b.coefs[scales].T-proj.T-
        reshape(M.coefs[scales].B[:,1:n^2]*reshape(y.coefs[scales].A,(n^2,1)),(n,n))-
        reshape(M.coefs[scales].B[:,n^2+1:2*n^2]*reshape(y.coefs[scales].B,(n^2,1)),(n,n))-
        reshape(M.coefs[scales].B[:,2*n^2+1:3*n^2]*reshape(y.coefs[scales].C,(n^2,1)),(n,n))
    y.coefs[scales].T = solve(M.coefs[scales].T,rhs)

    println("backward sub....")

    # Backward substitution U * x = y
    x = NSFVector(M.size,moments,scales,Vector{NSFVecCoefs}(undef,scales))
    x.coefs[scales] = NSFVecCoefs()
    x.coefs[scales].T = y.coefs[scales].T
    for j in scales:-1:1
        x.coefs[j].A = y.coefs[j].A-reshape(M.coefs[j].C[1:n^2,:]*reshape(x.coefs[j].T,(n^2,1)),(n,n))
        x.coefs[j].B = y.coefs[j].B-reshape(M.coefs[j].C[n^2+1:2*n^2,:]*reshape(x.coefs[j].T,(n^2,1)),(n,n))
        x.coefs[j].C = y.coefs[j].C-reshape(M.coefs[j].C[2*n^2+1:3*n^2,:]*reshape(x.coefs[j].T,(n^2,1)),(n,n))

        if j > 1
            x.coefs[j-1] = NSFVecCoefs()
            x.coefs[j-1].T = idwtmat(n,moments,H,G,x.coefs[j].A,x.coefs[j].B,x.coefs[j].C,x.coefs[j].T)
            n *= 2
        end
    end

    return x
end
