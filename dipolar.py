#!/usr/bin/env python

from __future__ import division, print_function

import time

import numpy as np

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from numpy.linalg import eigh
from scipy.interpolate import interp1d
from scipy.optimize import fsolve

import argparse

pi = np.pi

# Load CUDA functions:
with open('cuda_module.cu') as f:
    cuda_module = SourceModule(f.read())
epsilon_of_p_GPU = cuda_module.get_function("epsilon_of_p_GPU")
h_of_p_GPU = cuda_module.get_function("h_of_p_GPU")


def get_initial_guess(g, theta):
    q_guess = 2 * np.sqrt(2)
    mu_guess = 1

    # h_k has shape (N_kx, N_ky, 2), where the last dimension is for
    # wavenumbers k-q and k, in that order.
    h_guess =  np.ones((N_kx, N_ky, 2))/10

    # epsilon has shape (N_kx, N_ky, 3), where the last dimension is for
    # the three bands with wavenumbers k-q, k, k+q:
    epsilon_guess = np.zeros((N_kx, N_ky, 3))
    kx = q_guess * reduced_kx
    ky = q_guess * reduced_ky
    epsilon_guess[:, :, 0] = 1/2 * ((kx - q_guess)**2 + ky**2)
    epsilon_guess[:, :, 1] = 1/2 * (kx**2 + ky**2)
    epsilon_guess[:, :, 2] = 1/2 * ((kx + q_guess)**2 + ky**2)
    return q_guess, mu_guess, h_guess, epsilon_guess


def construct_H_k(epsilon, h):
    # H_k is N_kx*N_Ky 3 x 3 matrices:
    H_k = np.zeros((N_kx, N_ky, 3, 3))
    epsilon_k_minus_q = epsilon[:, :, 0]
    epsilon_k = epsilon[:, :, 1]
    epsilon_k_plus_q = epsilon[:, :, 2]
    h_k_minus_q = h[:, :, 0]
    h_k = h[:, :, 1]
    H_k[:, :, 0, 0] = epsilon_k_minus_q
    H_k[:, :, 0, 1] = H_k[:, :, 1, 0] = h_k_minus_q
    H_k[:, :, 1, 1] = epsilon_k
    H_k[:, :, 1, 2] = H_k[:, :, 2, 1] = h_k
    H_k[:, :, 2, 2] = epsilon_k_plus_q
    return H_k


def compute_q_and_mu(E_k_n, old_q):
    sorted_energy_eigenvalues = np.sort(E_k_n.flatten())
    mu = sorted_energy_eigenvalues[int(round(N_particles))]

    def plot_fermi_surface():
        import pylab as pl
        threshold = 0.1
        image = np.zeros((N_kx, N_ky, 3))
        image[np.abs(mu - E_k_n) < threshold] = 1
        image = image.sum(axis=-1)
        image[image > 0] = 1
        pl.imshow(image.transpose(), extent=[-0.5*old_q, 0.5*old_q, -old_q, old_q], origin='lower')
        pl.show()

    # plot_fermi_surface()

    y_origin_index = np.where(reduced_kx==0)[0][0]
    energies_along_x_axis = np.sort(E_k_n[N_kx/2:, y_origin_index])[:, 1]
    next_band_energies = np.sort(E_k_n[N_kx/2:, y_origin_index])[:, 0]
    kx_on_positive_axis = old_q*reduced_kx[N_kx/2:, 0]
    next_band_kx = old_q*(1-reduced_kx[N_kx/2:, 0])

    kx_both_bands = np.concatenate((kx_on_positive_axis, next_band_kx))
    energies_both_bands = np.concatenate((energies_along_x_axis, next_band_energies))
    interpolator = interp1d(kx_both_bands, energies_both_bands, kind='cubic')

    kx_at_fermi_surface = fsolve(lambda x: interpolator(x) - mu, old_q/2)[0]

    k_F = kx_at_fermi_surface
    q = 2*k_F
    return q, mu


def epsilon_of_p(px, py, E_k_n, U_k, mu, q, g, theta, debug=0):
    kxprime = q * reduced_kx
    kyprime = q * reduced_ky
    epsilon_p = np.zeros((N_kx, N_ky))

    block=(16,16,1)
    grid=(int(N_kx/16 + 1),int(N_ky/16 + 1))

    epsilon_of_p_GPU(drv.Out(epsilon_p),
                     drv.In(px), drv.In(py),
                     drv.In(kxprime), drv.In(kyprime),
                     drv.In(E_k_n), drv.In(U_k),
                     np.double(mu), np.double(q), np.double(g), np.double(theta),
                     np.int32(N_kx), np.int32(N_ky), np.int32(debug),
                     block=block, grid=grid)
    return epsilon_p


def compute_epsilon(E_k_n, U_k, mu, q, g, theta):
    # epsilon has shape (N_kx, N_ky, 3), where the last dimension is for
    # the three bands with wavenumbers k-q, k, k+q:
    epsilon = np.zeros((N_kx, N_ky, 3))
    kx = q * reduced_kx
    ky = q * reduced_ky
    epsilon[:, :, 0] = epsilon_of_p(kx - q, ky, E_k_n, U_k, mu, q, g, theta)
    epsilon[:, :, 1] = epsilon_of_p(kx, ky, E_k_n, U_k, mu, q, g, theta, debug=1)
    epsilon[:, :, 2] = epsilon_of_p(kx + q, ky, E_k_n, U_k, mu, q, g, theta)
    return epsilon


def h_of_p( px, py, E_k_n, U_k, mu, q, g, theta, debug=0):
    kxprime = q * reduced_kx
    kyprime = q * reduced_ky
    h_p = np.zeros((N_kx, N_ky))

    block=(16,16,1)
    grid=(int(N_kx/16 + 1),int(N_ky/16 + 1))

    h_of_p_GPU(drv.Out(h_p),
               drv.In(px), drv.In(py),
               drv.In(kxprime), drv.In(kyprime),
               drv.In(E_k_n), drv.In(U_k),
               np.double(mu), np.double(q), np.double(g), np.double(theta),
               np.int32(N_kx), np.int32(N_ky), np.int32(debug),
               block=block, grid=grid)
    return h_p


def compute_h( E_k_n, U_k, mu, q, g, theta):
    h = np.zeros((N_kx, N_ky, 2))
    kx = q * reduced_kx
    ky = q * reduced_ky
    h[:, :, 0] = h_of_p(kx - q, ky, E_k_n, U_k, mu, q, g, theta)
    h[:, :, 1] = h_of_p(kx, ky, E_k_n, U_k, mu, q, g, theta, debug=1)
    return h


def iterate(g, theta, q, mu, h, epsilon):

    print('(N_kx, N_ky):'.rjust(15),'(%d, %d)'%(N_kx, N_ky))
    print( 'N_particles:'.rjust(15), N_particles)
    print(           'g:'.rjust(15), g)
    print(       'theta:'.rjust(15), theta)
    print(            '='.rjust(15), theta/pi, 'pi')
    print(  "initial mu:".rjust(15), mu)
    print("   initial q:".rjust(15), q)

    i = 0

    start_time = time.time()
    time_of_last_print = time.time() - 10
    while True:

        # Construct the Hamiltonian, which has shape (N_kx, N_ky, 3, 3):
        H_k =  construct_H_k(epsilon, h)

        # Diagonalise it to give an array of shape (N_kx, N_ky, 3) of
        # energy eigenvalues, and an array of shape (N_kx, N_ky, 3, 3) of
        # eigenvectors, or equivalently of rotation matrices that
        # diagonalise each Hamiltonian.
        E_k_n, U_k = eigh(H_k)

        # Compute new guesses of q, h and epsilon:
        new_q, new_mu = compute_q_and_mu(E_k_n, q)
        new_epsilon = compute_epsilon(E_k_n, U_k, new_mu, new_q, g, theta)
        new_h = compute_h(E_k_n, U_k, new_mu, new_q, g, theta)

        convergence = abs((mu - new_mu)/mu)
        q += RELAXATION_PARAMETER*(new_q - q)
        mu += RELAXATION_PARAMETER*(new_mu - mu)
        h += RELAXATION_PARAMETER*(new_h - h)
        epsilon += RELAXATION_PARAMETER*(new_epsilon - epsilon)
        i += 1

        now = time.time()
        if (now - time_of_last_print > PRINT_INTERVAL) or (convergence < CONVERGENCE_THRESHOLD):
            print('\n  loop iteration:', i)
            print('    time per step:', round(1000*(now - start_time)/i, 2), 'ms')
            print('    convergence:', convergence)
            print("    mu", mu)
            print("    q", q)
            time_of_last_print = now

            if convergence < CONVERGENCE_THRESHOLD:
                print('time taken:', time.time() - start_time)
                return q, mu, h, epsilon, E_k_n, U_k


if __name__ == '__main__':

    CONVERGENCE_THRESHOLD = 1e-13
    RELAXATION_PARAMETER = 1.8
    PRINT_INTERVAL = 5 # seconds

    # Number of k points in x and y directions:
    N_kx=50
    N_ky=50

    # Range of kx and ky in units of q:
    reduced_kx_max=0.5
    reduced_ky_max=1

    parser = argparse.ArgumentParser(description='Determine the phase of a 2D dipolar Fermi gas at zero temperature.')
    parser.add_argument('--h5file', type=str, help='the output HDF5 file')
    parser.add_argument('--zlockserver', type=str,
                       help='hostname:port of a zlock server, so that multiple ' +
                            'processes running this script can ensure they don\'t access the same ' +
                            'HDF5 file at the same time. If not set, file locking will not be used and ' +
                            'multiple processes using the same file run the risk of corrupting it.')
    parser.add_argument('g',  type=float, default=-20,
                       help='the strength of the dipolar interaction')
    parser.add_argument('theta', type=float,
                       help='the angle of the dipoles')

    args = parser.parse_args()

    g = args.g
    theta = args.theta
    h5file = args.h5file
    zlockserver = args.zlockserver

    if zlockserver is not None:
        host, port = zlockserver.split(':')
        import h5_lock
        h5_lock.init(host, int(port))

    if h5file is not None:
        print('h5 file:'.rjust(15), h5file)
    if zlockserver is not None:
        print('zlock server:'.rjust(15), zlockserver)

    # We create arrays of 'reduced' wavenumbers, that is, wavenumbers as a fraction of q_c.
    # We'll multiply them by q_c each time we need actual wavenumbers:

    # x component of k in units of q, shape (N_kx, 1):
    reduced_kx = np.linspace(-reduced_kx_max, reduced_kx_max, N_kx, endpoint=False).reshape((N_kx, 1))

    # y component of k in units of q, shape (1, N_ky):
    reduced_ky = np.linspace(-reduced_ky_max, reduced_ky_max, N_ky, endpoint=False).reshape((1, N_ky))

    # k vectors in units of q, shape (N_kx, N_ky, 2):
    reduced_k = np.zeros((N_kx, N_ky, 2))
    reduced_k[:, :, 0] = reduced_kx
    reduced_k[:, :, 1] = reduced_ky

    initial_q, initial_mu, initial_h, initial_epsilon = get_initial_guess(g, theta)

    dkx_initial = (reduced_kx[1, 0] - reduced_kx[0, 0]) * initial_q
    dky_initial = (reduced_ky[0, 1] - reduced_ky[0, 0]) * initial_q

    N_particles = 2*pi / (dkx_initial * dky_initial)

    # import lineprofiler
    # lineprofiler.setup(outfile='lineprofiler')
    q, mu, h, epsilon, E_k_n, U_k = iterate(g, theta, initial_q, initial_mu, initial_h, initial_epsilon)

