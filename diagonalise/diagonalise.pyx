cimport cython
from libc.math cimport sin, cos, acos, sqrt, cbrt

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
def eig_analytic(double[:] a_arr,
                 double[:] b_arr,
                 double[:] c_arr,
                 double[:] d_arr,
                 double[:] e_arr,
                 double[:, :] evals_out,
                 double[:, :, :] evecs_out):
    """Analytically diagonalise the matrix:

    [[a, d, 0],
     [d, b, e],
     [0, e, c]],

    for real 1D arrays a, b, c, d, e. Returns results in the same format as
    np.linalg.eigh."""

    cdef int n = a_arr.shape[0]

    cdef int i = 0
    cdef double a, b, c, d, e
    cdef double B, C, D, f, g, h, j, k, L, M, N, P
    cdef double l1, l2, l3
    cdef double v11, v12, v13, v21, v22, v23, v31, v32, v33
    cdef double norm_v1, norm_v2, norm_v3

    for i in range(n):

        a = a_arr[i]
        b = b_arr[i]
        c = c_arr[i]
        d = d_arr[i]
        e = e_arr[i]

        # Coefficients of the characteristic polynomial, which is cubic. Its roots
        # are our eigenvalues. A = -1 omitted:
        B = (a + b + c)
        C = (e*e + d*d - a*b - a*c - b*c)
        D = a*b*c - c*d*d  - a*e*e

        # Solving the cubic polynomial with the trigonometric method, which is
        # numerically robust when all three roots are real, as they are in our
        # case. This algorithm copied from: http://www.1728.org/cubic2.htm and
        # simplified for the specific case of A = -1
        f = - C - B*B/3.0
        g = (-2*B*B*B - 9*B*C)/27.0 - D
        h = sqrt(-f*f*f/27.0)
        j = cbrt(h)
        k = acos(-g/(2*h))
        M = cos(k/3.0)
        N = sqrt(3) * sin(k/3.0)
        P = B/3

        # The three eigenvalues:
        l1 = -j*(M + N) + P
        l2 = -j*(M - N) + P
        l3 = 2*j*cos(k/3.0) + P

        # Compute the corresponding eigenvectors, using analytic expressions given
        # by Wolfram|Alpha. v13 = v23 = v33 = 1 omitted:
        v11 = -e/d + 1/(d*e) * (b - l1) * (c - l1)
        v12 = -1/e * (c - l1)
        v21 = -e/d + 1/(d*e) * (b - l2) * (c - l2)
        v22 = -1/e * (c - l2)
        v31 = -e/d + 1/(d*e) * (b - l3) * (c - l3)
        v32 = -1/e * (c - l3)

        # Normalise the eigenvectors:
        norm_v1 = sqrt(v11*v11 + v12*v12 + 1)
        norm_v2 = sqrt(v21*v21 + v22*v22 + 1)
        norm_v3 = sqrt(v31*v31 + v32*v32 + 1)

        v11 /= norm_v1
        v12 /= norm_v1
        v13 = 1/norm_v1
        v21 /= norm_v2
        v22 /= norm_v2
        v23 = 1/norm_v2
        v31 /= norm_v3
        v32 /= norm_v3
        v33 = 1/norm_v3

        evals_out[i, 0] = l1
        evals_out[i, 1] = l2
        evals_out[i, 2] = l3

        evecs_out[i, 0, 0] = v11
        evecs_out[i, 1, 0] = v12
        evecs_out[i, 2, 0] = v13
        evecs_out[i, 0, 1] = v21
        evecs_out[i, 1, 1] = v22
        evecs_out[i, 2, 1] = v23
        evecs_out[i, 0, 2] = v31
        evecs_out[i, 1, 2] = v32
        evecs_out[i, 2, 2] = v33


