#ifndef __KMEANS_H__
#define __KMEANS_H__


double** kmeans_main(int K, int iter, int *vec_idxs, int vec_idxs_size, const char *infile_name, double eps);
void dMatrix_to_file(double **A, int N, int m);
void compute_w(double** X, int N, int d, double** W);
double **d_matrix(double **A, int N, int d);
double **laplacian_matrix(double **A, int N, int d);
double **jacobi_mat(double **A, int N, int k_opt, int all_or_k);
double **final_centroids(double **vectors, int n, int k, int *init_clus);
#endif