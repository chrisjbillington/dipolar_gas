#!/usr/bin/env python

from __future__ import division, print_function

import argparse

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

import os
import time

import numpy as np

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from numpy.linalg import eigh

pi = np.pi

# Load CUDA functions:
this_directory = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(this_directory, 'cuda_module.cu')) as f:
    cuda_module = SourceModule(f.read())
epsilon_of_p_GPU = cuda_module.get_function("epsilon_of_p_GPU")
h_of_p_GPU = cuda_module.get_function("h_of_p_GPU")

def get_initial_guess(g, theta):
    q_guess = 2
    mu_guess = 1

    # h_k has shape (N_kx, N_ky, 2), where the last dimension is for
    # wavenumbers k-q and k, in that order.
    h_guess =  np.ones((N_kx, N_ky, 2))/10

    # epsilon has shape (N_kx, N_ky, 3), where the last dimension is for
    # the three bands with wavenumbers k-q, k, k+q:
    epsilon_guess = np.zeros((N_kx, N_ky, 3))
    kx = q_guess * reduced_kx
    ky = q_guess * reduced_ky
    epsilon_guess[:, :, 0] = ((kx - q_guess)**2 + ky**2)
    epsilon_guess[:, :, 1] = (kx**2 + ky**2)
    epsilon_guess[:, :, 2] = ((kx + q_guess)**2 + ky**2)
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


def compute_q_and_mu(E_k_n, old_q, N_states_under_fermi_surface):
    sorted_energy_eigenvalues = np.sort(E_k_n.flatten())

    from scipy.stats import linregress
    j = int(round(N_states_under_fermi_surface))
    energies = sorted_energy_eigenvalues[j-10:j+10]
    assert energies[10] == sorted_energy_eigenvalues[j]
    m, c, _, _, _ = linregress(range(20), energies)
    mu = m*(10 + N_states_under_fermi_surface - j) + c

    x_origin_index = np.where(reduced_ky==0)[1][0]
    ky_band_0 = old_q*reduced_ky[0, N_ky/2:]
    energies_along_y_axis = np.sort(E_k_n[x_origin_index, N_ky/2:], axis=0)
    energies_along_y_axis = np.sort(energies_along_y_axis, axis=1)[:, 0]

    E2_index = np.searchsorted(energies_along_y_axis, mu)
    E1_index = E2_index - 1
    E1 = energies_along_y_axis[E1_index]
    E2 = energies_along_y_axis[E2_index]
    ky1 = ky_band_0[E1_index]
    ky2 = ky_band_0[E2_index]
    m = (E2 - E1)/(ky2 - ky1)
    ky_at_fermi_surface = (mu-E1)/m + ky1

    # interpolator = interp1d(ky_band_0, energies_along_y_axis, kind='cubic')
    # ky_at_fermi_surface = fsolve(lambda x: interpolator(x) - mu, old_q/4)[0]
    k_F = ky_at_fermi_surface

    print(k_F)

    q = 2/k_F
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
    h[:, :, 1] = h_of_p(kx, ky, E_k_n, U_k, mu, q, g, theta)
    return h


def iterate(g, theta, q, mu, h, epsilon, N_states_under_fermi_surface):

    def plot_fermi_surface():
        import pylab as pl
        first_band_energies = (np.sort(E_k_n, axis=-1)[:, :, 0] - mu)
        second_band_energies = (np.sort(E_k_n, axis=-1)[:, :, 1] - mu)
        third_band_energies = (np.sort(E_k_n, axis=-1)[:, :, 2] - mu)
        import warnings
        # Supress unicode warning from matplotlib:
        with warnings.catch_warnings():
            pl.figure(figsize=(16,8))
            pl.subplot(231)
            pl.title('$E_{k}$')
            pl.xlabel('$k_x$')
            pl.ylabel('$k_y$')
            warnings.simplefilter("ignore")
            KX = reduced_kx*q*np.ones((N_kx, N_ky))
            KY = reduced_ky*q*np.ones((N_kx, N_ky))
            pl.contour(KX, KY, first_band_energies, 30)
            CS = pl.contour(KX, KY, first_band_energies, 0, colors='k', linewidths=3)
            pl.clabel(CS, inline=1, fontsize=20, fmt='$\mu$')
            pl.axhline(q/2, color='k', linestyle='--')
            pl.grid(True)

            pl.subplot(232)
            CS = pl.contour(KX, KY, second_band_energies, 10)
            pl.clabel(CS, inline=1, fontsize=10)
            pl.grid(True)

            pl.subplot(233)
            CS = pl.contour(KX, KY, third_band_energies, 10)
            pl.clabel(CS, inline=1, fontsize=10)
            pl.grid(True)

            x_origin_index = np.where(reduced_ky==0)[1][0]
            y_origin_index = np.where(reduced_kx==0)[0][0]
            kx = q * reduced_kx[:, 0]
            ky = q * reduced_ky[0, :]
            pl.subplot(234)
            for i in range(3):
                pl.plot(kx, E_k_n[:, y_origin_index, i])
                pl.ylabel('$E_{k_x}$')
                pl.xlabel('$k_x$')
                pl.axhline(mu, color='k', linestyle='--')
                pl.grid(True)
            pl.axis(xmin=kx.min(), xmax=kx.max())

            pl.subplot(235)
            for i in range(3):
                pl.plot(ky, E_k_n[x_origin_index, :, i])
                pl.ylabel('$E_{k_y}$')
                pl.xlabel('$k_y$')
                pl.axhline(mu, color='k', linestyle='--')
                pl.grid(True)
            pl.axis(xmin=ky.min(), xmax=ky.max())
            pl.show()

    global i

    print('(N_kx, N_ky):'.rjust(15),'(%d, %d)'%(N_kx, N_ky))
    print(    'N_states:'.rjust(15), N_states_under_fermi_surface)
    print(           'g:'.rjust(15), g)
    print(       'theta:'.rjust(15), theta)
    print(            '='.rjust(15), theta/pi, 'pi')
    print(  "initial mu:".rjust(15), mu)
    print("   initial q:".rjust(15), q)

    i = 0

    start_time = time.time()
    time_of_last_print = time.time() - 10
    while True:
        try:
            # Construct the Hamiltonian, which has shape (N_kx, N_ky, 3, 3):
            H_k =  construct_H_k(epsilon, h)

            # Diagonalise it to give an array of shape (N_kx, N_ky, 3) of
            # energy eigenvalues, and an array of shape (N_kx, N_ky, 3, 3) of
            # eigenvectors, or equivalently of rotation matrices that
            # diagonalise each Hamiltonian.
            E_k_n, U_k = eigh(H_k)

            dkx = (reduced_kx[1, 0] - reduced_kx[0, 0]) * q
            dky = (reduced_ky[0, 1] - reduced_ky[0, 0]) * q
            N_states_under_fermi_surface = pi / (dkx * dky)

            density = N_states_under_fermi_surface*dkx*dky/(4*pi**2)

            # Compute new guesses of q, h and epsilon:
            new_q, new_mu = compute_q_and_mu(E_k_n, q, N_states_under_fermi_surface)
            new_epsilon = compute_epsilon(E_k_n, U_k, new_mu, new_q, g, theta)
            new_h = compute_h(E_k_n, U_k, new_mu, new_q, g, theta)

            convergence = abs((mu - new_mu)/mu)
            q += RELAXATION_PARAMETER*(new_q - q)
            mu += RELAXATION_PARAMETER*(new_mu - mu)
            h += RELAXATION_PARAMETER*(new_h - h)
            epsilon += RELAXATION_PARAMETER*(new_epsilon - epsilon)
            i += 1

            # if not i % 10:
            #     plot_fermi_surface()
            now = time.time()
            if (now - time_of_last_print > PRINT_INTERVAL or True) or (convergence < CONVERGENCE_THRESHOLD):
                print('\n  loop iteration:', i)
                print('    time per step:', round(1000*(now - start_time)/i, 2), 'ms')
                print('    convergence:', convergence)
                print("    mu", mu)
                print("    q", q)
                print("    N_states:", N_states_under_fermi_surface)
                print('    density', density)
                time_of_last_print = now

                plot_fermi_surface()

                if convergence < CONVERGENCE_THRESHOLD:
                    print('time taken:', time.time() - start_time)
                    plot_fermi_surface()
                    return q, mu, h, epsilon, E_k_n, U_k
        except KeyboardInterrupt:
            import IPython
            IPython.embed()
            break

mus = []

if __name__ == '__main__':

    CONVERGENCE_THRESHOLD = 1e-13
    RELAXATION_PARAMETER = 1.0
    PRINT_INTERVAL = 5 # seconds

    # Number of k points in x and y directions:
    N_kx=201
    N_ky=201

    # Range of kx and ky in units of q:
    reduced_kx_max=0.5
    reduced_ky_max=0.6

    g = args.g # 10*(-3*pi**2/2 * 1.01) # args.g
    theta = args.theta # 0.3*pi # args.theta
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
    reduced_kx = np.linspace(-reduced_kx_max, reduced_kx_max, N_kx, endpoint=True).reshape((N_kx, 1))

    # y component of k in units of q, shape (1, N_ky):
    reduced_ky = np.linspace(-reduced_ky_max, reduced_ky_max, N_ky, endpoint=True).reshape((1, N_ky))

    # k vectors in units of q, shape (N_kx, N_ky, 2):
    reduced_k = np.zeros((N_kx, N_ky, 2))
    reduced_k[:, :, 0] = reduced_kx
    reduced_k[:, :, 1] = reduced_ky

    initial_q, initial_mu, initial_h, initial_epsilon = get_initial_guess(g, theta)

    dkx_initial = (reduced_kx[1, 0] - reduced_kx[0, 0]) * initial_q
    dky_initial = (reduced_ky[0, 1] - reduced_ky[0, 0]) * initial_q

    initial_N_states_under_fermi_surface = 2*pi / (dkx_initial * dky_initial)

    # import lineprofiler
    # lineprofiler.setup(outfile='lineprofiler')
    q, mu, h, epsilon, E_k_n, U_k = iterate(g, theta, initial_q, initial_mu, initial_h, initial_epsilon, initial_N_states_under_fermi_surface)


