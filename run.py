from __future__ import division, print_function
import numpy as np

pi = np.pi


def f(mu, E):
    """Fermi Dirac distribution at zero temperature"""
    occupation = np.empty(E.shape)
    occupation[E <= mu] = 1
    occupation[E > mu] = 0
    return occupation

V_input_arrays_x = []
V_input_arrays_y = []

evaluations = 0
dups = 0

def V(px, py, g, theta_dipole):
    global evaluations, dups
    if isinstance(px, np.ndarray) and isinstance(py, np.ndarray):
        points = np.prod(py.shape) * np.prod(px.shape)
        evaluations += points
        x_is_dup = False
        y_is_dup = False
        for previously_used in V_input_arrays_x:
            if np.allclose(px, previously_used, atol=1e-2, rtol=1e-2):
                x_is_dup = True
        for previously_used in V_input_arrays_y:
            if np.allclose(py, previously_used, atol=1e-2, rtol=1e-2):
                y_is_dup = True

        if x_is_dup and y_is_dup:
            dups += points
        print('dups:', 100*dups/evaluations, 'percent')
        if not x_is_dup:
            V_input_arrays_y.append(px)
        if not y_is_dup:
            V_input_arrays_x.append(py)
    import time
    start_time = time.time()
    result = g * np.sqrt(px**2 + py**2) * (np.cos(theta_dipole)**2 - np.sin(theta_dipole)**2)
    print('V', time.time() - start_time)
    return result

class DipoleGasProblem(object):
    def __init__(self, N_kx=50, N_ky=50, reduced_kx_max=0.5, reduced_ky_max=1):

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

        print(self.N_particles)

    def get_h_guess(self):
        # h_k has shape (N_kx, N_ky, 2), where the last dimension is for
        # wavenumbers k-q and k, in that order.
        return np.zeros((self.N_kx, self.N_ky, 2))

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

    def compute_q_and_mu(self, E_k_n):
        sorted_energy_eigenvalues = np.sort(E_k_n.flatten())
        mu = sorted_energy_eigenvalues[int(round(self.N_particles))]
        k_F = 'dunno' #TODO: figure it out.
        q = 2*k_F
        return q, mu

    def epsilon_of_p(self, px, py, E_k_n, U_k, mu, q, g, theta_dipole):
        kxprime = q * self.reduced_kx.reshape(self.N_kx, 1, 1, 1)
        kyprime = q * self.reduced_ky.reshape(1, self.N_ky, 1, 1)
        terms_over_kprime = np.zeros((self.N_kx, self.N_ky, self.N_kx, self.N_ky))
        for n in (-1, 0, 1): # Which band:
            potential_terms = V(0, 0, g, theta_dipole) - V(px - kxprime - n*q, py - kyprime, g, theta_dipole)
            for l in (0, 1, 2): # Which eigenvector/eigenvalue:
                terms_over_kprime += potential_terms * np.abs(U_k[:, :, n+1, l])**2 * f(mu, E_k_n[:, :, l])
        epsilon_p = (px**2 + py**2)/ 2 + terms_over_kprime.sum(axis=(0,1))
        return epsilon_p

    def compute_epsilon(self, E_k_n, U_k, mu, q, g, theta_dipole):
        # epsilon has shape (N_kx, N_ky, 3), where the last dimension is for
        # the three bands with wavenumbers k-q, k, k+q:
        epsilon = np.zeros((self.N_kx, self.N_ky, 3))
        kx = q * self.reduced_kx
        ky = q * self.reduced_ky
        import time
        start_time = time.time()
        epsilon[:, :, 0] = self.epsilon_of_p(kx - q, ky, E_k_n, U_k, mu, q, g, theta_dipole)
        epsilon[:, :, 1] = self.epsilon_of_p(kx, ky, E_k_n, U_k, mu, q, g, theta_dipole)
        epsilon[:, :, 2] = self.epsilon_of_p(kx + q, ky, E_k_n, U_k, mu, q, g, theta_dipole)
        print('epsilon:', time.time() - start_time)
        return epsilon

    def h_of_p(self, px, py, E_k_n, U_k, mu, q, g, theta_dipole):
        kxprime = q * self.reduced_kx.reshape(self.N_kx, 1, 1, 1)
        kyprime = q * self.reduced_ky.reshape(1, self.N_ky, 1, 1)
        terms_over_kprime = np.zeros((self.N_kx, self.N_ky, self.N_kx, self.N_ky))
        for l in (0, 1, 2): # Which eigenvector/eigenvalue:
            terms_over_kprime += ((V(q, 0, g, theta_dipole) -
                                  V(px - kxprime + q, py - kyprime, g, theta_dipole)) *
                                  U_k[:, :, 1, l] * U_k[:, :, 2, l] * f(mu, E_k_n[:, :, l]))
            terms_over_kprime += ((V(q, 0, g, theta_dipole) -
                                  V(px - kxprime, py - kyprime, g, theta_dipole))
                                  * U_k[:, :, 2, l] * U_k[:, :, 3, l] * f(mu, E_k_n[:, :, l]))
        h_p = terms_over_kprime.sum(axis=(0,1))
        return h_p

    def compute_h(self, E_k_n, U_k, mu, q, g, theta_dipole):
        h = np.zeros((self.N_kx, self.N_ky, 2))
        kx = q * self.reduced_kx
        ky = q * self.reduced_ky
        h[:, :, 0] = self.h_of_p(kx - q, ky, E_k_n, U_k, mu, q, g, theta_dipole)
        h[:, :, 1] = self.h_of_p(kx, ky, E_k_n, U_k, mu, q, g, theta_dipole)
        return h

    def find_eigenvalues(self, g, theta_dipole,
                         h_guess=None, epsilon_guess=None, q_guess=None, mu_guess=None,
                         relaxation_parameter=1.7):
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

        while True:
            # Construct the Hamiltonian, which has shape (N_kx, N_ky, 3, 3):
            H_k =  self.construct_H_k(epsilon, h)

            # Diagonalise it to give an array of shape (N_kx, N_ky, 3) of
            # energy eigenvalues, and an array of shape (N_kx, N_ky, 3, 3) of
            # eigenvectors, or equivalently of rotation matrices that
            # diagonalise each Hamiltonian.
            E_k_n, U_k = eigh(H_k)

            # Compute new guesses of q, h and epsilon:
            new_q, new_mu = self.compute_q_and_mu(E_k_n)
            new_epsilon = self.compute_epsilon(E_k_n, U_k, mu, q, g, theta_dipole)
            new_h = self.compute_h(E_k_n, U_k, mu, q, g, theta_dipole)

            # For the moment just use relaxation parameter of 1:
            # TODO: do over relaxation to make much fast for glorious nation
            # of Kazakhstan
            q = new_q
            mu = new_mu
            h = new_h
            epsilon = new_epsilon



if __name__ == '__main__':
    problem = DipoleGasProblem()
    problem.find_eigenvalues(0, 0)



