mutable struct NSFMatCoefs1D
    size::Int
    A::Matrix{Float64}
    B::Matrix{Float64}
    C::Matrix{Float64}
    T::Matrix{Float64}
end

mutable struct NSFMatrix1D
    size::Int
    moments::Int
    scales::Int
    coefs::Vector{NSFMatCoefs1D}
end

function get_NSFMatrix_decomp(X::Matrix{Float64},
    moments::Int,
    scales::Int,
    atol::Float64,
    )::NSFMatrix1D

    H,G = get_qmf(moments)
    Y = NSFMatrix1D(size(X)[1],moments,scales,Vector{NSFMatCoefs1D}(undef,scales))
    for j in 1:scales
        A,B,C,T = dwtmat(size(X)[1],DD,m,H,G)

        Ainv = inv(A)
        Y.coefs[j] = NSFMatCoefs1D(n,A,Ainv*B,C,T-C*Ainv*B)
        X = Y.coefs[j].T
    end
    return Y
end