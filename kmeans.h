#ifndef __KMEANS_H__
#define __KMEANS_H__
# include <Python.h>


double** kmeans_main(int K, int iter, int *vec_idxs, int vec_idxs_size, const char *infile_name, double eps);

#endif