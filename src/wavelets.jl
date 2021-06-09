function get_qmf(m::Int)::Tuple{Vector{Float64},Vector{Float64}}
    if m > 10
        throw("unimplemented")
    end

    H,G = Dict{Int,Vector{Float64}}(),Dict{Int,Vector{Float64}}()

    H[1] = Vector{Float64}([0.7071067811865476, 0.7071067811865476])
    H[2] = Vector{Float64}([-0.12940952255126037, 0.2241438680420134, 0.8365163037378079, 0.48296291314453416])
    H[3] = Vector{Float64}([0.03522629188570953, -0.08544127388202666, -0.13501102001025458, 0.45987750211849154, 0.8068915093110925, 0.33267055295008263])
    H[4] = Vector{Float64}([-0.010597401785069032, 0.0328830116668852, 0.030841381835560764, -0.18703481171909309, -0.027983769416859854, 0.6308807679298589, 0.7148465705529157, 0.2303778133088965])
    H[5] = Vector{Float64}([0.0033357252854737712, -0.012580751999081999, -0.006241490212798274, 0.07757149384004572, -0.032244869584638375, -0.24229488706638203, 0.13842814590132074, 0.7243085284377729, 0.6038292697971896, 0.16010239797419293])
    H[6] = Vector{Float64}([-0.0010773010853084796, 0.004777257510945511, 0.0005538422011614961, -0.03158203931748603, 0.027522865530305727, 0.09750160558732304, -0.12976686756726194, -0.22626469396543983, 0.31525035170919763, 0.7511339080210954, 0.49462389039845306, 0.11154074335010947])
    H[7] = Vector{Float64}([0.00035371379997452024, -0.0018016407040474908, 0.0004295779729213665, 0.01255099855609984, -0.01657454163066688, -0.03802993693501441, 0.08061260915108308, 0.07130921926683026, -0.22403618499387498, -0.14390600392856498, 0.4697822874051931, 0.7291320908462351, 0.3965393194819173, 0.07785205408500918])
    H[8] = Vector{Float64}([-0.00011747678412476953, 0.0006754494064505693, -0.00039174037337694705, -0.004870352993451574, 0.008746094047405777, 0.013981027917398282, -0.044088253930794755, -0.017369301001807547, 0.12874742662047847, 0.0004724845739132828, -0.2840155429615469, -0.015829105256349306, 0.5853546836542067, 0.6756307362972898, 0.31287159091429995, 0.05441584224310401])
    H[9] = Vector{Float64}([3.93473203162716e-05, -0.0002519631889427101, 0.00023038576352319597, 0.0018476468830562265, -0.00428150368246343, -0.004723204757751397, 0.022361662123679096, 0.00025094711483145197, -0.06763282906132997, 0.03072568147933338, 0.14854074933810638, -0.09684078322297646, -0.2932737832791749, 0.13319738582500756, 0.6572880780513005, 0.6048231236901112, 0.24383467461259034, 0.038077947363878345])
    H[10] = Vector{Float64}([-1.3264202894521244e-05, 9.358867032006959e-05, -0.00011646685512928545, -0.0006858566949597116, 0.001992405295185056, 0.001395351747052901, -0.010733175483330575, 0.0036065535669561697, 0.033212674059341, -0.029457536821875813, -0.07139414716639708, 0.09305736460357235, 0.12736934033579325, -0.19594627437737705, -0.24984642432731538, 0.2811723436605775, 0.6884590394536035, 0.5272011889317256, 0.1881768000776915, 0.026670057900555554])

    G[1] = Vector{Float64}([-0.7071067811865476, 0.7071067811865476])
    G[2] = Vector{Float64}([-0.48296291314453416, 0.8365163037378079, -0.2241438680420134, -0.12940952255126037])
    G[3] = Vector{Float64}([-0.33267055295008263, 0.8068915093110925, -0.45987750211849154, -0.13501102001025458, 0.08544127388202666, 0.03522629188570953])
    G[4] = Vector{Float64}([-0.2303778133088965, 0.7148465705529157, -0.6308807679298589, -0.027983769416859854, 0.18703481171909309, 0.030841381835560764, -0.0328830116668852, -0.010597401785069032])
    G[5] = Vector{Float64}([-0.16010239797419293, 0.6038292697971896, -0.7243085284377729, 0.13842814590132074, 0.24229488706638203, -0.032244869584638375, -0.07757149384004572, -0.006241490212798274, 0.012580751999081999, 0.0033357252854737712])
    G[6] = Vector{Float64}([-0.11154074335010947, 0.49462389039845306, -0.7511339080210954, 0.31525035170919763, 0.22626469396543983, -0.12976686756726194, -0.09750160558732304, 0.027522865530305727, 0.03158203931748603, 0.0005538422011614961, -0.004777257510945511, -0.0010773010853084796])
    G[7] = Vector{Float64}([-0.07785205408500918, 0.3965393194819173, -0.7291320908462351, 0.4697822874051931, 0.14390600392856498, -0.22403618499387498, -0.07130921926683026, 0.08061260915108308, 0.03802993693501441, -0.01657454163066688, -0.01255099855609984, 0.0004295779729213665, 0.0018016407040474908, 0.00035371379997452024])
    G[8] = Vector{Float64}([-0.05441584224310401, 0.31287159091429995, -0.6756307362972898, 0.5853546836542067, 0.015829105256349306, -0.2840155429615469, -0.0004724845739132828, 0.12874742662047847, 0.017369301001807547, -0.044088253930794755, -0.013981027917398282, 0.008746094047405777, 0.004870352993451574, -0.00039174037337694705, -0.0006754494064505693, -0.00011747678412476953])
    G[9] = Vector{Float64}([-0.038077947363878345, 0.24383467461259034, -0.6048231236901112, 0.6572880780513005, -0.13319738582500756, -0.2932737832791749, 0.09684078322297646, 0.14854074933810638, -0.03072568147933338, -0.06763282906132997, -0.00025094711483145197, 0.022361662123679096, 0.004723204757751397, -0.00428150368246343, -0.0018476468830562265, 0.00023038576352319597, 0.0002519631889427101, 3.93473203162716e-05])
    G[10] = Vector{Float64}([-0.026670057900555554, 0.1881768000776915, -0.5272011889317256, 0.6884590394536035, -0.2811723436605775, -0.24984642432731538, 0.19594627437737705, 0.12736934033579325, -0.09305736460357235, -0.07139414716639708, 0.029457536821875813, 0.033212674059341, -0.0036065535669561697, -0.010733175483330575, -0.001395351747052901, 0.001992405295185056, 0.0006858566949597116, -0.00011646685512928545, -9.358867032006959e-05, -1.3264202894521244e-05])

    return H[m],G[m]
end

"""
Discrete wavelet transform of vector

:param n: size of x
:type n: Int
:param x: vector to be transformed
:type x: AbstractVector{Float64}
:param m: number of vanishing moments
:type m: Int
:param H: low pass filter
:type H: Vector{Float64}
:param G: high pass filter
:type G: Vector{Float64}
"""
function dwtvec(n::Int,
    x::AbstractVector{Float64},
    m::Int,
    H::Vector{Float64},
    G::Vector{Float64},
    )::Tuple{AbstractVector{Float64},AbstractVector{Float64}}
    d = [sum(G[k]*x[1+mod(2*(i-1)+k-1,n)] for k in 1:2*m) for i in 1:n÷2]
    s = [sum(H[k]*x[1+mod(2*(i-1)+k-1,n)] for k in 1:2*m) for i in 1:n÷2]
    if issparse(x)
        d,s = sparse(d),sparse(s)
    end
    return d,s
end

"""
Inverse wavelet transform of vector

:param n: size of d and s
:type n: Int
:param d: difference vector
:type d: AbstractVector{Float64}
:param s: average vector
:type s: AbstractVector{Float64}
:param m: number of vanishing moments
:type m: Int
:param H: low pass filter
:type H: Vector{Float64}
:param G: high pass filter
:type G: Vector{Float64}
"""
function idwtvec(n::Int,
    d::AbstractVector{Float64},
    s::AbstractVector{Float64},
    m::Int,
    H::Vector{Float64},
    G::Vector{Float64},
    )::AbstractVector{Float64}
    x = zeros(Float64,2*n)
    for i in 1:n
        x[2*(i-1)+1] = sum(H[2*(k-1)+1]*s[1+mod(i-k,n)]+G[2*(k-1)+1]*d[1+mod(i-k,n)] for k in 1:m)
        x[2*(i-1)+2] = sum(H[2*(k-1)+2]*s[1+mod(i-k,n)]+G[2*(k-1)+2]*d[1+mod(i-k,n)] for k in 1:m)
    end
    if issparse(d) & issparse(s)
        x = sparse(x)
    end
    return x
end

"""
Wavelet transform of a matrix

wt(X) -> | A C |
         | B T |

:param n: number of rows/columns in X
:type n: Int
:param X: matrix to be transformed
:param m: number of vanishing moments
:type m: Int
:param H: low pass filter
:type H: Vector{Float64}
:param G: high pass filter
:type G: Vector{Float64}

:return: A,B,C,T blocks
"""
function dwtmat(n::Int,
    X::AbstractMatrix{Float64},
    m::Int,
    H::Vector{Float64},
    G::Vector{Float64},
    )

    if size(X) != (n,n)
        throw("invalid matrix size")
    end

    if n%2 != 0
        throw("size of matrix not divisible by 2")
    end

    if length(H) != 2*m || length(G) != 2*m
        throw("invalid quadrature mirror filters")
    end

    if issparse(X)
        Y = spzeros(size(X,1),size(X,2))
    else
        Y = Matrix{Float64}(undef,size(X))
    end

    for i in 1:n
        Y[i,1:n÷2],Y[i,n÷2+1:n] = dwtvec(n,X[i,:],m,H,G)
    end

    for i in 1:n
        Y[1:n÷2,i],Y[n÷2+1:n,i] = dwtvec(n,Y[:,i],m,H,G)
    end

    return Y[1:n÷2,1:n÷2],Y[n÷2+1:n,1:n÷2],Y[1:n÷2,n÷2+1:n],Y[n÷2+1:n,n÷2+1:n]
end

"""
Inverse wavelet transform of a matrix

:param n: number of rows/columns in X
:type n: Int
:param A: A block
:type A:: AbstractMatrix{Float64}
:param B: B block
:type B:: AbstractMatrix{Float64}
:param C: C block
:type C:: AbstractMatrix{Float64}
:param T: T block
:type T:: AbstractMatrix{Float64}
:param m: number of vanishing moments
:type m: Int
:param H: low pass filter
:type H: Vector{Float64}
:param G: high pass filter
:type G: Vector{Float64}

:return: A,B,C,T blocks
"""
function idwtmat(n::Int,
    m::Int,
    H::Vector{Float64},
    G::Vector{Float64},
    A::AbstractMatrix{Float64},
    B::AbstractMatrix{Float64},
    C::AbstractMatrix{Float64},
    T::AbstractMatrix{Float64},
    )::AbstractMatrix{Float64}

    if issparse(A) & issparse(B) & issparse(C) & issparse(T)
        Y = spzeros(Float64,n*2,n*2)
    else
        Y = Matrix{Float64}(undef,n*2,n*2)
    end

    for i in 1:n
        Y[:,i] = idwtvec(n,A[:,i],B[:,i],m,H,G)
        Y[:,n+i] = idwtvec(n,C[:,i],T[:,i],m,H,G)
    end

    for i in 1:2*n
        Y[i,:] = idwtvec(n,Y[i,1:n],Y[i,n+1:2*n],m,H,G)
    end

    return Y
end

"""
Wavelet transform of a matrix of size (n^2, n^2). The matrix is obatined
by discretizing a 2 dimensional differential operator on a n-by-n grid.

:param n: size of the grid
:type n: Int
:param X: matrix of size (n^2, n^2)
:type X:: AbstractMatrix{Float64}
:param m: number of vanishing moments
:type m: Int
:param H: low pass filter
:type H: Vector{Float64}
:param G: high pass filter
:type G: Vector{Float64}

:return: wavelet transform
"""
function dwtmat2d(n::Int,
    X::AbstractMatrix{Float64},
    m::Int,
    H::Vector{Float64},
    G::Vector{Float64},
    )
    # Check inputs
    if size(X) != (n^2,n^2)
        throw("invalid matrix size of $(size(X))")
    end

    if n%2 != 0
        throw("size of matrix not divisible by 2")
    end

    if length(H) != 2*m || length(G) != 2*m
        throw("invalid quadrature mirror filters")
    end
    
    Y = copy(X)
    k = n÷2
    
    # Transform rows of n consecutive entries
    for i in 1:n^2
        for j in 1:n
            d,s = dwtvec(n,Y[i,(j-1)*n+1:j*n],m,H,G)
            Y[i,(j-1)*n+1:(j-1)*n+k] = d
            Y[i,(j-1)*n+k+1:j*n] = s
        end
    end

    # Transform columns of n consecutive entries
    for i in 1:n^2
        for j in 1:n
            d,s = dwtvec(n,Y[(j-1)*n+1:j*n,i],m,H,G)
            Y[(j-1)*n+1:(j-1)*n+k,i] = d
            Y[(j-1)*n+k+1:j*n,i] = s
        end
    end

    # Transform rows of n entries with a stride of n
    for i in 1:n^2
        for j in 1:n
            d,s = dwtvec(n,Y[i,j:n:(n-1)*n+j],m,H,G)
            Y[i,j:n:(k-1)*n+j] = d
            Y[i,k*n+j:n:(n-1)*n+j] = s
        end
    end

    # Transform column of n entries with a stride of n
    for i in 1:n^2
        for j in 1:n
            d,s = dwtvec(n,Y[j:n:(n-1)*n+j,i],m,H,G)
            Y[j:n:(k-1)*n+j,i] = d
            Y[k*n+j:n:(n-1)*n+j,i] = s
        end
    end
    
    # Re-indexing
    grid = reshape(collect(1:n^2),(n,n))
    index = vcat(collect(Iterators.flatten(grid[1:n÷2,1:n÷2])),
                collect(Iterators.flatten(grid[n÷2+1:n,1:n÷2])),
                collect(Iterators.flatten(grid[1:n÷2,n÷2+1:n])),
                collect(Iterators.flatten(grid[n÷2+1:n,n÷2+1:n])),
            )

    return Y[index[1:3*k^2],index[1:3*k^2]],Y[index[3*k^2+1:n^2],index[1:3*k^2]],
        Y[index[1:3*k^2],index[3*k^2+1:n^2]],Y[index[3*k^2+1:n^2],index[3*k^2+1:n^2]]
end

function solve(A::Matrix{Float64},
    b::Matrix{Float64},
    )::Matrix{Float64}

    if size(A) != (size(b)[1]*size(b)[2],size(b)[1]*size(b)[2])
        throw(DimensionMismatch("size of A not equal to size of b"))
    end

    n = size(A)[1]
    F = LinearAlgebra.svd(A)
    sig = Vector{Float64}(undef,n)
    for i in 1:n
        if F.S[i] > 10e-7
            sig[i] = 1.0/F.S[i]
        else
            sig[i] = 0.0
        end
    end
    println("coarsest scale condition number $(F.S[1]/F.S[n-1]) and $(F.S[n])")

    x = reshape(transpose(F.Vt)*(sig.*(transpose(F.U)*reshape(b,n,1))),size(b))

    return x
end