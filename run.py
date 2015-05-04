from __future__ import division
import numpy as np

pi = np.pi

class DipoleGasProblem(object):
    def __init__(self, N_kx=1000, N_ky=2000, reduced_kx_max=0.5, reduced_ky_max=1):

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
        print(mu)
        assert False

    def compute_epsilon(self, E_k_n, U_k, mu, q):
        raise NotImplementedError()

    def compute_h_k(self, E_k_n, U_k, mu, q):
        raise NotImplementedError()


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
            new_epsilon = self.compute_epsilon(E_k_n, U_k, mu, q)
            new_h = self.compute_h(E_k_n, U_k, mu, q)

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



