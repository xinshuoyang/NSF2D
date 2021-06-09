mutable struct NSFVecCoefs
    A::Matrix{Float64}
    B::Matrix{Float64}
    C::Matrix{Float64}
    T::Union{Matrix{Float64},Nothing}
    NSFVecCoefs() = new()
    NSFVecCoefs(A,B,C,T) = new(A,B,C,T)
end

mutable struct NSFVector
    size::Int
    moments::Int
    scales::Int
    coefs::Vector{NSFVecCoefs}
end

function NSFVector_decomp(size::Int,
        moments::Int,
        scales::Int,
        X::Matrix{Float64}
        )::NSFVector

    if size%(1<<scales) != 0
        throw("size not divisible by 2^scales")
    end

    H,G = get_qmf(moments)
    Y = NSFVector(size,moments,scales,Vector{NSFVecCoefs}(undef,scales))
    T = X
    for j in 1:scales
        A,B,C,T = dwtmat(size,T,moments,H,G)
        Y.coefs[j] = NSFVecCoefs(A,B,C,nothing)
        size ÷= 2
    end
    Y.coefs[scales].T = T

    return Y
end

function NSFVector_reconst(X::NSFVector)::Matrix{Float64}
    n = X.size÷(2^X.scales)
    H,G = get_qmf(X.moments)
    T = X.coefs[X.scales].T
    for j in X.scales:-1:1
        T = idwtmat(n,X.moments,H,G,X.coefs[j].A,X.coefs[j].B,X.coefs[j].C,T)
        n *= 2
    end
    return T
end
