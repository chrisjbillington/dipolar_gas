from __future__ import division, print_function

import time

import numpy as np

import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from scipy.interpolate import interp1d
from scipy.optimize import fsolve

# Load CUDA functions:
with open('cuda_module.cu') as f:
    mod = SourceModule(f.read())
epsilon_of_p_GPU = mod.get_function("epsilon_of_p_GPU")
h_of_p_GPU = mod.get_function("h_of_p_GPU")

pi = np.pi


def f(mu, E):
    """Fermi Dirac distribution at zero temperature"""
    occupation = np.zeros(E.shape)
    occupation[E <= mu] = 1
    # occupation[E > mu] = 0
    return occupation

def V(px, py, g, theta_dipole):
    result = g * np.sqrt(px**2 + py**2) * (np.cos(theta_dipole)**2 - np.sin(theta_dipole)**2)
    return result

class DipoleGasProblem(object):
    def __init__(self, N_kx=100, N_ky=100, reduced_kx_max=0.5, reduced_ky_max=1):
        print('(N_kx, N_ky) is: (%d, %d)'%(N_kx, N_ky))

        # number of k points in x and y directions:
        self.N_kx = N_kx
        self.N_ky = N_ky

        # range of k in x and y directions:
        self.reduced_kx_max = reduced_kx_max
        self.reduced_ky_max = reduced_ky_max

        # We create arrays of 'reduced' wavenumbers, that is, wavenumbers as a fraction of q_c.
        # We'll multiply them by q_c each time we need actual wavenumbers:

        # x component of k, shape (N_kx, 1):
        self.reduced_kx = np.linspace(-reduced_kx_max, reduced_kx_max, N_kx, endpoint=False).reshape((N_kx, 1))

        # y component of k, shape (1, N_ky):
        self.reduced_ky = np.linspace(-reduced_ky_max, reduced_ky_max, N_ky, endpoint=False).reshape((1, N_ky))

        # k vectors, shape (N_kx, N_ky, 2):
        self.reduced_k = np.zeros((N_kx, N_ky, 2))
        self.reduced_k[:, :, 0] = self.reduced_kx
        self.reduced_k[:, :, 1] = self.reduced_ky

        initial_q, _ = self.get_q_and_mu_guess()
        dkx = (self.reduced_kx[1, 0] - self.reduced_kx[0, 0]) * initial_q
        dky = (self.reduced_ky[0, 1] - self.reduced_ky[0, 0]) * initial_q

        self.N_particles = 2*pi / (dkx * dky)

        print('Number of particles is:', self.N_particles)

    def get_h_guess(self):
        # h_k has shape (N_kx, N_ky, 2), where the last dimension is for
        # wavenumbers k-q and k, in that order.
        # return np.zeros((self.N_kx, self.N_ky, 2))
        return np.random.randn(self.N_kx, self.N_ky, 2)

    def get_epsilon_guess(self, q):
        # epsilon has shape (N_kx, N_ky, 3), where the last dimension is for
        # the three bands with wavenumbers k-q, k, k+q:
        epsilon_guess = np.zeros((self.N_kx, self.N_ky, 3))
        kx = q * self.reduced_kx
        ky = q * self.reduced_ky
        epsilon_guess[:, :, 0] = 1/2 * ((kx - q)**2 + ky**2)
        epsilon_guess[:, :, 1] = 1/2 * (kx**2 + ky**2)
        epsilon_guess[:, :, 2] = 1/2 * ((kx + q)**2 + ky**2)
        return epsilon_guess

    def get_q_and_mu_guess(self):
        # q is a vector of length 2 in units of k_F in the x direction. The
        # direction is in principle arbitrary, and so we simply set it to be
        # the x direction and just make q a scalar for its magnitude:
        q_guess = 2 * np.sqrt(2)
        mu_guess = 1
        return q_guess, mu_guess

    def construct_H_k(self, epsilon, h):
        # H_k is N_kx*N_Ky 3 x 3 matrices:
        H_k = np.zeros((self.N_kx, self.N_ky, 3, 3))
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

    def compute_q_and_mu(self, E_k_n, old_q):
        sorted_energy_eigenvalues = np.sort(E_k_n.flatten())
        mu = sorted_energy_eigenvalues[int(round(self.N_particles))]

        def plot_fermi_surface():
            import pylab as pl
            threshold = 0.1
            image = np.zeros((self.N_kx, self.N_ky, 3))
            image[np.abs(mu - E_k_n) < threshold] = 1
            image = image.sum(axis=-1)
            image[image > 0] = 1
            pl.imshow(image.transpose(), extent=[-0.5*old_q, 0.5*old_q, -old_q, old_q], origin='lower')
            pl.show()

        # plot_fermi_surface()

        y_origin_index = np.where(self.reduced_kx==0)[0][0]
        energies_along_x_axis = np.sort(E_k_n[self.N_kx/2:, y_origin_index])[:, 1]
        next_band_energies = np.sort(E_k_n[self.N_kx/2:, y_origin_index])[:, 0]
        kx_on_positive_axis = old_q*self.reduced_kx[self.N_kx/2:, 0]
        next_band_kx = old_q*(1-self.reduced_kx[self.N_kx/2:, 0])

        kx_both_bands = np.concatenate((kx_on_positive_axis, next_band_kx))
        energies_both_bands = np.concatenate((energies_along_x_axis, next_band_energies))
        interpolator = interp1d(kx_both_bands, energies_both_bands, kind='cubic')

        kx_at_fermi_surface = fsolve(lambda x: interpolator(x) - mu, old_q/2)[0]

        k_F = kx_at_fermi_surface
        q = 2*k_F
        return q, mu

    def epsilon_of_p_slow(self, px, py, E_k_n, U_k, mu, q, g, theta_dipole, debug=False):
        kxprime = q * self.reduced_kx.reshape(self.N_kx, 1, 1, 1)
        kyprime = q * self.reduced_ky.reshape(1, self.N_ky, 1, 1)
        terms_over_kprime = np.zeros((self.N_kx, self.N_ky, self.N_kx, self.N_ky))
        for n in (-1, 0, 1): # Which band:
            potential_terms = V(0, 0, g, theta_dipole) - V(px - kxprime - n*q, py - kyprime, g, theta_dipole)
            for l in (0, 1, 2): # Which eigenvector/eigenvalue:
                eigenvector_element = U_k[:, :, n+1, l, np.newaxis, np.newaxis]
                eigenvalue = E_k_n[:, :, l, np.newaxis, np.newaxis]
                terms_over_kprime += potential_terms * np.abs(eigenvector_element)**2 * f(mu, eigenvalue)
                if debug and n==0 and l==0:
                    print('  epsilon_k[7,5] kprime[22,33] n=0, l=0 (Python):')
                    print('    px is:', px[7, 0])
                    print('    py is:', py[0, 5])
                    print('    kxprime is:', kxprime[22,0,0,0])
                    print('    kyprime is:', kyprime[0,33,0,0])
                    print('    potential terms is', potential_terms[22, 33, 7, 5])
                    print('    evec element is', eigenvector_element[22, 33, 0, 0])
                    this_term = potential_terms * np.abs(eigenvector_element)**2 * f(mu, eigenvalue)
                    print('    value of term is:', this_term[22, 33, 7, 5])
        epsilon_p = (px**2 + py**2)/ 2 + terms_over_kprime.sum(axis=(0,1))
        if debug:
            print('  epsilon_k[7,5] (Python):', epsilon_p[7, 5])
            print()
        return epsilon_p

    def epsilon_of_p(self, px, py, E_k_n, U_k, mu, q, g, theta_dipole, debug=0):
        kxprime = q * self.reduced_kx
        kyprime = q * self.reduced_ky
        epsilon_p = np.zeros((self.N_kx, self.N_ky))

        block=(16,16,1)
        grid=(int(self.N_kx/16 + 1),int(self.N_ky/16 + 1))

        epsilon_of_p_GPU(drv.Out(epsilon_p),
                         drv.In(px), drv.In(py),
                         drv.In(kxprime), drv.In(kyprime),
                         drv.In(E_k_n), drv.In(U_k),
                         np.double(mu), np.double(q), np.double(g), np.double(theta_dipole),
                         np.int32(self.N_kx), np.int32(self.N_ky), np.int32(debug),
                         block=block, grid=grid)
        return epsilon_p

    def compute_epsilon(self, E_k_n, U_k, mu, q, g, theta_dipole):
        # epsilon has shape (N_kx, N_ky, 3), where the last dimension is for
        # the three bands with wavenumbers k-q, k, k+q:
        epsilon = np.zeros((self.N_kx, self.N_ky, 3))
        kx = q * self.reduced_kx
        ky = q * self.reduced_ky
        epsilon[:, :, 0] = self.epsilon_of_p(kx - q, ky, E_k_n, U_k, mu, q, g, theta_dipole)
        epsilon[:, :, 1] = self.epsilon_of_p(kx, ky, E_k_n, U_k, mu, q, g, theta_dipole, debug=1)
        epsilon[:, :, 2] = self.epsilon_of_p(kx + q, ky, E_k_n, U_k, mu, q, g, theta_dipole)

        # epsilon2 = np.zeros((self.N_kx, self.N_ky, 3))
        # epsilon2[:, :, 0] = self.epsilon_of_p_slow(kx - q, ky, E_k_n, U_k, mu, q, g, theta_dipole)
        # epsilon2[:, :, 1] = self.epsilon_of_p_slow(kx, ky, E_k_n, U_k, mu, q, g, theta_dipole, debug=True)
        # epsilon2[:, :, 2] = self.epsilon_of_p_slow(kx + q, ky, E_k_n, U_k, mu, q, g, theta_dipole)
        return epsilon

    def h_of_p_slow(self, px, py, E_k_n, U_k, mu, q, g, theta_dipole, debug=False):
        kxprime = q * self.reduced_kx.reshape(self.N_kx, 1, 1, 1)
        kyprime = q * self.reduced_ky.reshape(1, self.N_ky, 1, 1)
        terms_over_kprime = np.zeros((self.N_kx, self.N_ky, self.N_kx, self.N_ky))
        for l in (0, 1, 2): # Which eigenvector/eigenvalue:
            eigenvector_elements =  U_k[:, :, :, l, np.newaxis, np.newaxis]
            eigenvalue = E_k_n[:, :, l, np.newaxis, np.newaxis]
            terms_over_kprime += ((V(q, 0, g, theta_dipole) -
                                  V(px - kxprime + q, py - kyprime, g, theta_dipole)) *
                                  eigenvector_elements[:, :, 0] * eigenvector_elements[:, :, 1] * f(mu, eigenvalue))
            terms_over_kprime += ((V(q, 0, g, theta_dipole) -
                                  V(px - kxprime, py - kyprime, g, theta_dipole))
                                  * eigenvector_elements[:, :, 1] * eigenvector_elements[:, :, 2] * f(mu, eigenvalue))
            if debug and l==0:
                potential_term_1 = (V(q, 0, g, theta_dipole) - V(px - kxprime + q, py - kyprime, g, theta_dipole))
                potential_term_2 = (V(q, 0, g, theta_dipole) - V(px - kxprime, py - kyprime, g, theta_dipole))
                print('  h_k[7,5] kprime[22,33], l=0 (Python):')
                print('    px is:', px[7, 0])
                print('    py is:', py[0, 5])
                print('    kxprime is:', kxprime[22,0,0,0])
                print('    kyprime is:', kyprime[0,33,0,0])
                print('    potential term 1 is:', potential_term_1[22, 33, 7, 5])
                print('    potential term 2 is:', potential_term_2[22, 33, 7, 5])
                first_term = potential_term_1 * eigenvector_elements[:, :, 0] * eigenvector_elements[:, :, 1] * f(mu, eigenvalue)
                second_term = potential_term_2 * eigenvector_elements[:, :, 1] * eigenvector_elements[:, :, 2] * f(mu, eigenvalue)
                this_term = first_term + second_term
                print('  value of term is:', this_term[22, 33, 7, 5])
        h_p = terms_over_kprime.sum(axis=(0,1))
        if debug:
            print('  h_k[7,5] (Python):', h_p[7, 5])
            print()
        return h_p

    def h_of_p(self, px, py, E_k_n, U_k, mu, q, g, theta_dipole, debug=0):
        kxprime = q * self.reduced_kx
        kyprime = q * self.reduced_ky
        h_p = np.zeros((self.N_kx, self.N_ky))

        block=(16,16,1)
        grid=(int(self.N_kx/16 + 1),int(self.N_ky/16 + 1))

        h_of_p_GPU(drv.Out(h_p),
                   drv.In(px), drv.In(py),
                   drv.In(kxprime), drv.In(kyprime),
                   drv.In(E_k_n), drv.In(U_k),
                   np.double(mu), np.double(q), np.double(g), np.double(theta_dipole),
                   np.int32(self.N_kx), np.int32(self.N_ky), np.int32(debug),
                   block=block, grid=grid)
        return h_p

    def compute_h(self, E_k_n, U_k, mu, q, g, theta_dipole):
        h = np.zeros((self.N_kx, self.N_ky, 2))
        kx = q * self.reduced_kx
        ky = q * self.reduced_ky
        h[:, :, 0] = self.h_of_p(kx - q, ky, E_k_n, U_k, mu, q, g, theta_dipole)
        h[:, :, 1] = self.h_of_p(kx, ky, E_k_n, U_k, mu, q, g, theta_dipole, debug=1)

        # h2 = np.zeros((self.N_kx, self.N_ky, 3))
        # h2[:, :, 0] = self.h_of_p_slow(kx - q, ky, E_k_n, U_k, mu, q, g, theta_dipole)
        # h2[:, :, 1] = self.h_of_p_slow(kx, ky, E_k_n, U_k, mu, q, g, theta_dipole, debug=True)

        return h

    def find_eigenvalues(self, g, theta_dipole,
                         h_guess=None, epsilon_guess=None, q_guess=None, mu_guess=None,
                         relaxation_parameter=1.7, threshold=1e-13):
        from numpy.linalg import eigh
        if q_guess is None or mu_guess is None:
            q_guess, mu_guess = self.get_q_and_mu_guess()
        if ((q_guess is not None and mu_guess is None) or
            (q_guess is None and mu_guess is not None)):
            raise ValueError('both q_guess and mu_guess must be None, or both must be not None')
        if h_guess is None:
            h_guess = self.get_h_guess()
        if epsilon_guess is None:
            epsilon_guess = self.get_epsilon_guess(q_guess)
        h = h_guess
        epsilon = epsilon_guess
        q = q_guess
        mu = mu_guess

        # import cPickle as pickle
        # with open('cache.pickle') as f:
        #     (q, mu, h, epsilon) = pickle.load(f)

        print('g is:', g)
        print('theta is:', theta_dipole)
        print('        =', theta_dipole/pi, 'pi')
        print("initial mu is:", mu)
        print("initial q is:", q)

        i = 0

        start_time = time.time()
        time_of_last_print = time.time() - 10
        while True:

            # Construct the Hamiltonian, which has shape (N_kx, N_ky, 3, 3):
            H_k =  self.construct_H_k(epsilon, h)

            # Diagonalise it to give an array of shape (N_kx, N_ky, 3) of
            # energy eigenvalues, and an array of shape (N_kx, N_ky, 3, 3) of
            # eigenvectors, or equivalently of rotation matrices that
            # diagonalise each Hamiltonian.
            E_k_n, U_k = eigh(H_k)

            # Compute new guesses of q, h and epsilon:
            new_q, new_mu = self.compute_q_and_mu(E_k_n, q)
            new_epsilon = self.compute_epsilon(E_k_n, U_k, new_mu, new_q, g, theta_dipole)
            new_h = self.compute_h(E_k_n, U_k, new_mu, new_q, g, theta_dipole)

            # For the moment just use relaxation parameter of 1:
            # TODO: do over relaxation to make much fast for glorious nation
            # of Kazakhstan
            relaxation_parameter = 1.80
            convergence = abs((mu - new_mu)/mu)
            q += relaxation_parameter*(new_q - q)
            mu += relaxation_parameter*(new_mu - mu)
            h += relaxation_parameter*(new_h - h)
            epsilon += relaxation_parameter*(new_epsilon - epsilon)
            i += 1

            now = time.time()
            if (now - time_of_last_print > 1) or (convergence < threshold):
                print('\nloop iteration:', i+1)
                print('  time per step:', round(1000*(now - start_time)/i, 2), 'ms')
                print('  convergence:', convergence)
                print("  mu", mu)
                print("  q", q)
                time_of_last_print = now

                if convergence < threshold:
                    print('time taken:', time.time() - start_time)
                    # import cPickle as pickle
                    # with open('cache.pickle','w') as f:
                    #     pickle.dump((q, mu, h, epsilon), f)
                    break

if __name__ == '__main__':

    # import lineprofiler
    # lineprofiler.setup(outfile='lineprofiler')
    problem = DipoleGasProblem()
    problem.find_eigenvalues(-20, 0.1*pi)


