from __future__ import division, print_function
import sys
import os
import shutil

# Compile the Cython extension if it's not already compiled:
this_dir = os.path.abspath(os.path.dirname(__file__))
extension = os.path.join(this_dir, 'diagonalise')
if not (os.path.exists(extension + '.pyd') or os.path.exists(extension + '.so')):
    cwd = os.getcwd()
    try:
        os.chdir(this_dir)
        if os.system(sys.executable + ' setup.py build_ext --inplace'):
            raise Exception('error building Cython extensions')
        os.unlink('diagonalise.c')
        shutil.rmtree('build')
    finally:
        os.chdir(cwd)


from diagonalise import eig_analytic


if __name__ == '__main__':
    # Test diagonalising some matrices. Compare solution and run time to
    # np.linalg.eigh.
    import numpy as np
    from numpy.linalg import eigh
    import time

    N = 1000000

    # All elements are random real numbers between -1 and 1:
    a = 2*np.random.random(N) - 1
    b = 2*np.random.random(N) - 1
    c = 2*np.random.random(N) - 1
    d = 2*np.random.random(N) - 1
    e = 2*np.random.random(N) - 1

    # Construct the actual matrix to pass to np.linalg.eigh:
    H = np.zeros((N, 3, 3))
    H[:, 0, 0] = a
    H[:, 0, 1] = H[:, 1, 0] = d
    H[:, 1, 1] = b
    H[:, 1, 2] = H[:, 2, 1] = e
    H[:, 2, 2] = c

    # Output arrays for analytic method:
    evals_analytic = np.empty((N, 3))
    evecs_analytic = np.empty((N, 3, 3))

    for i in range(10):
        # Analytic solution, Cython implementation:
        start_time = time.time()
        eig_analytic(a, b, c, d, e, evals_analytic, evecs_analytic)
        print('analytic cython:', round(1e6*(time.time() - start_time)/N, 3), 'us per diagonalisation')

    # Numerical solution with np.linalg.eigh:
    start_time = time.time()
    evals, evecs = eigh(H)
    print('numeric:', round(1e6*(time.time() - start_time)/N, 3), 'us per diagonalisation')

    # Print the differences:
    print('max evals err:', np.abs(evals - evals_analytic).max())
    print('max evecs err:', (np.abs(evecs) - np.abs(evecs_analytic)).max())

    # Checking that eigenvalues are correct:
    assert np.allclose(evals, evals_analytic)
    print('analytic and numeric eigenvalues are equal')
    # Check that eigenvectors are the same, up to a sign:
    assert np.allclose(np.abs(evecs), np.abs(evecs_analytic), atol=1e-7, rtol=1)
    print('analytic and numeric eigenvectors are equal up to a sign')


