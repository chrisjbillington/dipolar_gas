#include <stdio.h>
#include <math.h>


__global__ void epsilon_of_p_GPU(double *output_arr, double *px_arr, double *py_arr,
                                 double *kxprime_arr, double *kyprime_arr,
                                 double *E_k_n, double *U_k,
                                 double mu, double q, double g, double theta,
                                 int N_kx, int N_ky, int debug)
{
    const int i = threadIdx.x + blockDim.x*blockIdx.x;
    const int j = threadIdx.y + blockDim.y*blockIdx.y;
    const int N_n = 3;
    const int N_l = 3;

    int iprime, jprime, n, l;

    const double px = px_arr[i];
    const double py = py_arr[j];

    double kxprime, kyprime;
    double px_pot, py_pot, potential;
    double evec_element, eigenval;
    double accumulator = 0;

    double pot_trig_terms = pow(cos(theta), 2) - pow(sin(theta), 2)*0;

    const double dpx = px_arr[1] - px_arr[0];
    const double dpy = py_arr[1] - py_arr[0];
    const double prefactor = dpx*dpy/(4*M_PI*M_PI);

    if ((i < N_kx) & (j < N_ky)){
        for (iprime=0; iprime<N_kx; iprime++){
            for (jprime=0; jprime<N_ky; jprime++){
                kxprime = kxprime_arr[iprime];
                kyprime = kyprime_arr[jprime];
                for (n=-1; n<2; n++){
                    px_pot = px - kxprime - n*q;
                    py_pot = py - kyprime;
                    potential = - g*sqrt(px_pot*px_pot + py_pot*py_pot) * pot_trig_terms;
                    for (l=0; l<3; l++){
                        eigenval = E_k_n[N_ky*N_l*iprime + N_l*jprime + l];
                        if (eigenval <= mu){
                            evec_element = U_k[N_ky*N_n*N_l*iprime + N_n*N_l*jprime + N_l*(n+1) + l];
                            // if ((debug > 0) & (i==7) & (j==5) & (iprime==22) & (jprime==33) & (n==0) & (l==0)){
                            //      printf("  epsilon_k[7,5] kprime[22,33] n=0, l=0 (CUDA):\n");
                            //      printf("    px is: %f\n", px);
                            //      printf("    py is: %f\n", py);
                            //      printf("    kxprime is: %f\n", kxprime);
                            //      printf("    kyprime is: %f\n", kyprime);
                            //      printf("    potential terms is %f\n",potential);
                            //      printf("    evec element is %f\n", evec_element);
                            //      printf("    value of term is: %f\n", potential*evec_element*evec_element);
                            //  }
                            accumulator += potential * evec_element*evec_element;
                        }
                    }
                }
            }
        }

        // Multiple the sum by the appropriate prefactor:
        accumulator *= prefactor;

        // Add the kinetic term:
        accumulator += (px*px + py*py);

        output_arr[N_ky*i + j] = accumulator;
        // if ((debug > 0) & (i==7) & (j==5)){
        //     printf("  epsilon_k[7,5] (CUDA): %f\n\n", accumulator);
        // }
    }
}

__global__ void h_of_p_GPU(double *output_arr, double *px_arr, double *py_arr,
                                 double *kxprime_arr, double *kyprime_arr,
                                 double *E_k_n, double *U_k,
                                 double mu, double q, double g, double theta,
                                 int N_kx, int N_ky, int debug)
{
    const int i = threadIdx.x + blockDim.x*blockIdx.x;
    const int j = threadIdx.y + blockDim.y*blockIdx.y;
    const int N_n = 3;
    const int N_l = 3;

    int iprime, jprime, l;

    double px = px_arr[i];
    double py = py_arr[j];

    double kxprime, kyprime;
    double px_pot, py_pot, potential_1, potential_2;
    double evec_element_1, evec_element_2, evec_element_3, eigenval;
    double accumulator = 0;

    double pot_trig_terms = pow(cos(theta), 2) - pow(sin(theta), 2)*0;
    double V_of_q = g*sqrt(q*q) * pot_trig_terms; // V(px=q, py=0)

    const double dpx = px_arr[1] - px_arr[0];
    const double dpy = py_arr[1] - py_arr[0];
    const double prefactor = dpx*dpy/(4*M_PI*M_PI);

    if ((i < N_kx) & (j < N_ky)){
        for (iprime=0; iprime<N_kx; iprime++){
            for (jprime=0; jprime<N_ky; jprime++){
                kxprime = kxprime_arr[iprime];
                kyprime = kyprime_arr[jprime];
                for (l=0; l<3; l++){
                    eigenval = E_k_n[N_ky*N_l*iprime + N_l*jprime + l];
                    if (eigenval <= mu){
                        evec_element_1 = U_k[N_ky*N_n*N_l*iprime + N_n*N_l*jprime + N_l*0 + l];
                        evec_element_2 = U_k[N_ky*N_n*N_l*iprime + N_n*N_l*jprime + N_l*1 + l];
                        evec_element_3 = U_k[N_ky*N_n*N_l*iprime + N_n*N_l*jprime + N_l*2 + l];

                        px_pot = px - kxprime + q;
                        py_pot = py - kyprime;
                        potential_1 = V_of_q - g*sqrt(px_pot*px_pot + py_pot*py_pot) * pot_trig_terms;

                        px_pot = px - kxprime;
                        py_pot = py - kyprime;
                        potential_2 = V_of_q - g*sqrt(px_pot*px_pot + py_pot*py_pot) * pot_trig_terms;

                        accumulator += potential_1 * evec_element_1*evec_element_2 + potential_2 * evec_element_2*evec_element_3;

                        // if ((debug > 0) & (i==7) & (j==5) & (iprime==22) & (jprime==33) & (l==0)){
                             // printf("  h_k[7,5] kprime[22,33], l=0 (CUDA):\n");
                             // printf("    px is: %f\n", px);
                             // printf("    py is: %f\n", py);
                             // printf("    kxprime is: %f\n", kxprime);
                             // printf("    kyprime is: %f\n", kyprime);
                             // printf("    potential term 1 is %f\n", potential_1);
                             // printf("    potential term 2 is %f\n", potential_2);
                             // printf("    value of term is: %f\n",
                             //        potential_1 * evec_element_1*evec_element_2 + potential_2 * evec_element_2*evec_element_3);
                         // }
                    }
                }
            }
        }
        // Multiple the sum by the appropriate prefactor:
        accumulator *= prefactor;

        output_arr[N_ky*i + j] = accumulator;
        // if ((debug > 0) & (i==7) & (j==5)){
        //     printf("  h_k[7,5] (CUDA): %f\n\n", accumulator);
        // }
    }
}
