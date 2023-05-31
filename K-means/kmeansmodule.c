# define PY_SSIZE_T_CLEAN
# include <Python.h>
# include <stdio.h>
# include "kmeans_to_py.c"

int counter = 0;

static PyObject *fit(PyObject *self, PyObject *args) {
    int k, i = 0, m=0, j=0, n = 0;
    int iter;
    char *vec_idx_lst;
    int *cVec;
    double eps;
    PyObject *python_list, *python_d;
    double **centroids, *list_cen;
    int cord_len, N;
    /*
        # parsing all the data from python to C variables
    */
    if (!PyArg_ParseTuple(args, "iidsi", &k, &iter, &eps, &vec_idx_lst,&cord_len )) {
        return NULL;
    }
    cVec = (int*)calloc(k, sizeof(int));
    /*
        # cast string indexes to int array
    */
    char *end = vec_idx_lst;
    while(*end) 
    {
    n = strtol(vec_idx_lst, &end, 10);
    cVec[i] = n;
    i++;
    while (*end == ','){end++;}
    vec_idx_lst = end;
    }
    i = 0;
    N = k * cord_len;
    centroids = kmeans_main(k,iter,cVec, k, "file_1.txt",eps);
    list_cen = (double*)calloc(N, sizeof(double));
    /*
        # make centroids matrix as double list
    */
    for (i = 0; i < k; i++)
    {
        for (j= 0; j < cord_len; j++)
        {
            list_cen[m] = centroids[i][j];
            m++;
        }
        
    }
    /*
        # send the final centroids list as output
    */
    python_list = PyList_New(N);
    for (i = 0; i < N; i++)
    {
        python_d = Py_BuildValue("d", list_cen[i]);
        PyList_SetItem(python_list, i, python_d);
    }
    return python_list;
}

// module's function table
static PyMethodDef KmeansMethods[] = {
    {
        "fit", // name exposed to Python
        (PyCFunction)fit, // C wrapper function
        METH_VARARGS, // received variable args (but really just 1)
        "Calculates kmeans centroids" // documentation
    }
    ,{
        NULL, NULL, 0, NULL
    }
};

// modules definition
static struct PyModuleDef kmeansmodule = {
    PyModuleDef_HEAD_INIT,
    "kmeans",     // name of module exposed to Python
    "Python wrapper for custom C extension library calculating kmeans.", // module documentation
    -1,
    KmeansMethods
};

PyMODINIT_FUNC 
PyInit_kmeans(void) {
    PyObject *m;
    m = PyModule_Create(&kmeansmodule);
    if (!m){
        return NULL;
    }
    return m;
}
