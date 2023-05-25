# define PY_SSIZE_T_CLEAN
# include <Python.h>
# include <stdio.h>
# include "spkmeans.h"
# include "spkmeans.c"

int counter = 0;
// convert python matrix to C matrix
double **matrix_converter_to_C(PyObject *data, int rows, int cols){
    // allocate memory for new matrix
    double **mat = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++)
    {
        mat[i] = (double*)malloc(cols *sizeof(double));
    }
    for (int i = 0; i < rows; i++) 
    {
        PyObject* curr_row = PyList_GetItem(data, i);
        for (int j = 0; j < cols; j++) 
        {
            PyObject* elem = PyList_GetItem(curr_row, j);
            float val = (float)PyFloat_AsDouble(elem);
            mat[i][j] = (double)val;
        }
    }
    return mat;
}
// convert C matrix to python matrix
PyObject *matrix_converter_to_Py(double **A, int rows, int cols){
        PyObject *pyMat = PyList_New(rows);
        for (int i = 0; i < rows; i++) {
        PyObject* new_row = PyList_New(cols);
        for (int j = 0; j < cols; j++) {
            float val = (float)A[i][j];
            PyObject* new_elem = PyFloat_FromDouble(val);
            PyList_SET_ITEM(new_row, j, new_elem);
        }
        PyList_SET_ITEM(pyMat, i, new_row);
    }
    return pyMat;
}
// kmeans ++

// Perform full spectral kmeans
static PyObject *spk(PyObject *self, PyObject *args){
    PyObject *data_points, *cVec;
    double **A, **final_centroid;
    int *cen_inx, N, k;
    // parsing all the data from python to C variables
    if (!PyArg_ParseTuple(args, "OOii", &data_points, &cVec, &N, &k)) {
        return NULL;
    }
    A = matrix_converter_to_C(data_points, N, k);

    // convert initial centroids indexes to C array
    cen_inx = (int*) malloc(k * sizeof(int));
    for (int i = 0; i < k; i++) {
        cen_inx[i] = PyLong_AsLong(PyList_GetItem(cVec, i));
    }
    // compute kmeans ++  final cenrtroids
    final_centroid = final_centroids(A, N, k, cen_inx);
    return matrix_converter_to_Py(final_centroid, k, k);
}
// Calculate and output the Weighted Adjacency Matrix
static PyObject *wam(PyObject *self, PyObject *args){
    PyObject *data_points;
    double **A, **W;
    // parsing all the data from python to C variables
    if (!PyArg_ParseTuple(args, "O", &data_points)) {
        return NULL;
    }
    int rows = PyList_Size(data_points);
    int cols = PyList_Size(PyList_GetItem(data_points, 0));
    A = matrix_converter_to_C(data_points, rows, cols);
    W = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++)
    {
        W[i] = (double*)malloc(rows *sizeof(double));
    }
    compute_w(A, rows, cols, W);
    return matrix_converter_to_Py(W, rows, rows);
}
// Calculate and output the Diagonal Degree Matrix 
static PyObject *ddg(PyObject *self, PyObject *args){
    PyObject *data_points;
    double **A, **D;
    // parsing all the data from python to C variables
    if (!PyArg_ParseTuple(args, "O", &data_points)) {
        return NULL;
    }
    int rows = PyList_Size(data_points);
    int cols = PyList_Size(PyList_GetItem(data_points, 0));
    A = matrix_converter_to_C(data_points, rows, cols);
    D = d_matrix(A, rows, cols);
    return matrix_converter_to_Py(D, rows, rows);
}
// Calculate and output the Graph Laplacian
static PyObject *gl(PyObject *self, PyObject *args){
    PyObject *data_points;
    double **A, **L;
    // parsing all the data from python to C variables
    if (!PyArg_ParseTuple(args, "O", &data_points)) {
        return NULL;
    }
    int rows = PyList_Size(data_points);
    int cols = PyList_Size(PyList_GetItem(data_points, 0));
    A = matrix_converter_to_C(data_points, rows, cols);
    L = laplacian_matrix(A, rows, cols);
    return matrix_converter_to_Py(L, rows, rows);
}
// Calculate and output the eigenvalues and eigenvectors
static PyObject *jacobi(PyObject *self, PyObject *args){
    PyObject *data_points;
    double **A, **jac;
    int k, is_all;
    // parsing all the data from python to C variables
    if (!PyArg_ParseTuple(args, "Oii", &data_points, &k, &is_all)) {
        return NULL;
    }
    int rows = PyList_Size(data_points);
    A = matrix_converter_to_C(data_points, rows, rows);
    jac = jacobi_mat(A, rows ,k, is_all);
    return matrix_converter_to_Py(jac, rows + 1, rows);
}
// module's function table
static PyMethodDef KmeansMethods[] = {
    {
        "spk", // name exposed to Python
        (PyCFunction)spk, // C wrapper function
        METH_VARARGS, // received variable args (but really just 1)
        "Perform full spectral kmeans" // documentation
    },
    {
        "wam", // name exposed to Python
        (PyCFunction)wam, // C wrapper function
        METH_VARARGS, // received variable args (but really just 1)
        "Calculate and output the Weighted Adjacency Matrix" // documentation
    },
    {
        "ddg", // name exposed to Python
        (PyCFunction)ddg, // C wrapper function
        METH_VARARGS, // received variable args (but really just 1)
        "Calculate and output the Diagonal Degree Matrix" // documentation
    },
    {
        "gl", // name exposed to Python
        (PyCFunction)gl, // C wrapper function
        METH_VARARGS, // received variable args (but really just 1)
        "Calculate and output the Graph Laplacian" // documentation
    },
    {
        "jacobi", // name exposed to Python
        (PyCFunction)jacobi, // C wrapper function
        METH_VARARGS, // received variable args (but really just 1)
        "Calculate and output the eigenvalues and eigenvectors" // documentation
    }
    ,{
        NULL, NULL, 0, NULL
    }
};

// modules definition
static struct PyModuleDef kmeansmodule = {
    PyModuleDef_HEAD_INIT,
    "mykmeanssp",     // name of module exposed to Python
    "Python wrapper for custom C extension library calculating kmeans.", // module documentation
    -1,
    KmeansMethods
};

PyMODINIT_FUNC 
PyInit_mykmeanssp(void) {
    PyObject *m;
    m = PyModule_Create(&kmeansmodule);
    if (!m){
        return NULL;
    }
    return m;
}
