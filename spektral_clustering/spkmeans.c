# include <stdio.h>
# include <stdlib.h>
# include "spkmeans.h"
# include <math.h>
# include <stdio.h>
# include <string.h>

struct cord
{
    double value;
    struct cord *next;
};
struct vector
{
    struct vector *next;
    struct cord *cords;
};
/*
# free all vector
*/
void free_cord (struct cord *cords)
{
    if(cords != NULL)
    {
        free_cord(cords->next);
        free(cords);
    }
}
/*
    # gets double matrix
    # write the matrix to a text file, like data is given
*/
void dMatrix_to_file(double **A, int N, int m){
    int i = 0, j = 0;
    FILE *f = fopen("file_1.txt", "w");
    if (f == NULL){
        return;
    }
    for ( i = 0; i < N; i++)
    {
        for ( j = 0; j < m; j++)
        {
            fprintf(f, "%.4f", A[i][j]);
            if (j < m -1)   
            {
                fprintf(f, ",");
            }
            
        }
        fprintf(f, "\n");
    }
    fclose(f);
}
/*
    # gets a file
    # make double matrix first row: [0] = rows, [1] = cols
*/
double **mat_from_file(char *filename){
    int i = 0, size = 0, num_cord = 1, j = 0;
    FILE *infile = NULL;
    struct vector *head_vec = NULL, *curr_vec = NULL;
    struct cord *head_cord = NULL, *curr_cord = NULL;
    double n = 0.;
    double **mat;
    char c = 0;
    infile = fopen(filename, "r");
    if (infile == NULL)
    {
        return NULL;
    }
    head_cord = calloc(1, sizeof(struct cord));
    curr_cord = head_cord;
    curr_cord->next = NULL;

    head_vec = calloc(1 ,sizeof(struct vector));
    curr_vec = head_vec;
    curr_vec->next = NULL;

    while ((fscanf(infile, "%lf%c", &n, &c) == 2))
    {   
        if (c == '\n')
        {
            size ++;
            curr_cord->value = n;
            curr_vec->cords = head_cord;
            curr_vec->next = calloc(1 ,sizeof(struct vector));
            curr_vec = curr_vec->next;
            curr_vec->next = NULL;
            head_cord = calloc(1, sizeof(struct cord));
            curr_cord = head_cord;
            curr_cord->next = NULL;
            continue;
        }
        if (size == 0){
            num_cord ++;
        }
        curr_cord->value = n;
        curr_cord->next = calloc(1, sizeof(struct cord));
        curr_cord = curr_cord->next;
        curr_cord->next = NULL;    
    }
    fclose(infile);
    mat = (double**)malloc((size+1) * sizeof(double*));
    for ( i = 0; i < size + 1; i++)
    {
        mat[i] = (double*)malloc(num_cord * sizeof(double));
    }
    
    curr_vec = head_vec;
    for ( i = 1; i < size + 1; i++)
    {   
        curr_cord = curr_vec->cords;
        for ( j = 0; j < num_cord; j++)
        {
            mat[i][j] = curr_cord->value;
            curr_cord = curr_cord->next;
        }
        curr_vec = curr_vec->next;
    }
    mat[0][0] = size;
    mat[0][1] = num_cord;
    return mat;

}
/*
    # free all vector
*/
void free_vec (struct vector *vec)
{
    if(vec != NULL)
    {
        free_vec(vec->next);
        free_cord(vec->cords);
        free(vec);
    }
}
/*
# deep copy vector 
*/
struct vector* copy_vec(struct vector *vec)
{
    struct vector *vec_2 = calloc(1 ,sizeof(struct vector));
    struct cord *head_cord, *curr_cord, *temp_cord;

    vec_2->next=NULL;
    head_cord = calloc(1 ,sizeof(struct cord));
    curr_cord = head_cord;
    temp_cord = vec->cords;

    while (temp_cord != NULL)
    {
        curr_cord->value = temp_cord->value;
        temp_cord = temp_cord->next;
        if (temp_cord != NULL)
        {
        curr_cord->next = calloc(1, sizeof(struct cord));
        curr_cord = curr_cord->next;
        curr_cord->next = NULL;   
        }    
    }
    vec_2->cords = head_cord;
    return vec_2;
    
}
/*
    # get 2 vectors
    # return euclidian distance of them
*/
double euclidian_distance(struct vector *vec1,struct vector *vec2)
{
    double sum = 0;
    struct cord *cord1 = vec1->cords;
    struct cord *cord2 = vec2->cords;
    while (cord1 != NULL && cord2 != NULL)
    {
        sum = sum + ((cord1->value)-(cord2->value))*((cord1->value)-(cord2->value));
        cord1 = cord1->next;
        cord2 = cord2->next;
    }
    return sum;
}
/*
    # get a vector
    # return number of cords
*/
int get_vec_len(struct vector *vec)
{
    int cnt = 0;
    struct cord *cord = vec->cords;
    while (cord != NULL)
    {
        cnt++;
        cord = cord->next;
    }
    return cnt;
}

/*
    # gets 2 size n*n matrices
    # return new n*n matrix: C = A*b
*/
double** matrix_mul(double** A, double** B, int n) {
    int i = 0, j = 0, k = 0;
    double** C = malloc(n * sizeof(double*));
    /*
    memory allocate check
    */
    if (C == NULL) {
        return NULL;
    }
    for ( i = 0; i < n; i++) {
        C[i] = malloc(n * sizeof(double));
            /*
             memory allocate check
            */
        if (C[i] == NULL) {
            for ( j = 0; j < i; j++) {
                free(C[j]);
            }
            free(C);
            return NULL;
        }
        for ( j = 0; j < n; j++) {
            /*matrices multiplication */
            double sum = 0.0;
            for ( k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            C[i][j] = sum;
        }
    }
    return C;
}
/*
    # gets 2 matrices size N*N
    # return : P^t * A * P
*/
double **transform_matrix(double** A, double** P, int n) {
    int i = 0, j = 0;
    /*generate P^t*/
    double **P_transpose, **PA, **B;
    P_transpose = malloc(n * sizeof(double*));
    if (P_transpose == NULL) {
        return NULL;
    }
    for ( i = 0; i < n; i++) 
    {
        P_transpose[i] = malloc(n * sizeof(double));
        if (P_transpose[i] == NULL) {
            for ( j = 0; j < i; j++) 
            {
                free(P_transpose[j]);
            }
            free(P_transpose);
            return NULL;
        }
        for ( j = 0; j < n; j++) 
        {
            P_transpose[i][j] = P[j][i];
        }
    }
    
    /*Compute P^t * A*/
    PA = matrix_mul(P_transpose, A, n);
    if (PA == NULL) {
        for ( i = 0; i < n; i++) {
            free(P_transpose[i]);
        }
        free(P_transpose);
        return NULL;
    }
    /*Compute (P^t * A) * P*/
    B = matrix_mul(PA, P, n);
    /*free the extra memory we used*/
    for ( i = 0; i < n; i++) {
        free(P_transpose[i]);
        free(PA[i]);
    }
    free(P_transpose);
    free(PA);
    return B;
}

/*
# get a cluster
# return the avg vector
*/
struct vector* avg_clust(struct vector *vec1)
{
    struct vector *avg_head_vec, *curr_vec;
    struct cord *avg_head_cord, *avg_curr_cord, *curr_cord;

    int size = 0;

    avg_head_cord = calloc(1, sizeof(struct cord));
    avg_curr_cord = avg_head_cord;

    avg_head_vec = calloc(1 ,sizeof(struct vector));
    avg_head_vec->next = NULL;
    
    curr_vec = vec1;
    curr_cord = curr_vec->cords;
    /* sums all the cords*/
    while (curr_vec != NULL)
    {
        size++;
        if (size == 1){
            while (curr_cord != NULL)
        {
        avg_curr_cord->value += curr_cord->value;
        avg_curr_cord->next = calloc(1, sizeof(struct cord));
        avg_curr_cord = avg_curr_cord->next;
        avg_curr_cord->next = NULL;
        curr_cord = curr_cord->next;
        }
        }else{
            
        while (curr_cord != NULL)
        {
        avg_curr_cord->value += curr_cord->value;
        avg_curr_cord = avg_curr_cord->next;
        curr_cord = curr_cord->next;
        }
        }
        
        curr_vec = curr_vec->next;
        if (curr_vec != NULL)
        {
            curr_cord = curr_vec->cords;
        }
        avg_curr_cord = avg_head_cord;
        
    }
    /*calculate the avg*/
    avg_head_vec->cords = avg_head_cord;
    avg_curr_cord = avg_head_cord;
    while (avg_curr_cord != NULL)
    {   
        avg_curr_cord->value = avg_curr_cord->value / size;
        if(avg_curr_cord->next->value == 0.0000){ avg_curr_cord->next = NULL;break;}
        avg_curr_cord = avg_curr_cord->next;

    }
    return avg_head_vec;

}
/*
# get all clusters, number j and vector
# add the vector to cluster j
*/
void add_to_cluster(struct vector *clusters[], struct vector *vec, int j)
{
    /*struct vector clus;*/
    struct vector *temp_vec;
    temp_vec = copy_vec(vec);
    if (clusters[j] != NULL){
       temp_vec->next = clusters[j]; 
    }else{
        temp_vec->next = NULL;        
    }
    clusters[j] = temp_vec;
    
    
}
/*
# get all means, all data vectors, int K and all clusters
# return clusters sorted by euclidian distance from means
*/
void clusters_maker(struct vector *clus [],
struct vector *means[], struct vector *data, int K )
{
    struct vector *curr_vec = data;
    int j,i;
    double min , dist;
    
    while (curr_vec != NULL)
    {
        j = -1;
        min = 100*100;
        for (i = 0; i < K; i++)
        {   
            dist = euclidian_distance(curr_vec, means[i]);
            if (dist<min){
                min = dist;
                j = i;
            }
        }
        add_to_cluster(clus, curr_vec, j);
        curr_vec = curr_vec->next;
    }
    
}


void free_mat(double **A, int N){
    int i;
    for ( i = 0; i < N; i++)
    {
        free(A[i]);
    }
}
/*
    # gets double matrix and print it, separate by comma
*/
void printMatrix(double **matrix, int rows, int cols) {
    int i,j;
    for ( i = 0; i < rows; i++) {
        for ( j = 0; j < cols-1; j++) {
            printf("%.4f,", matrix[i][j]);
        }
        printf("%.4f", matrix[i][cols-1]);
        printf("\n");
    
    }
}
/*
    # delete first line of matrix A
*/
double **delete_first_line(double **A, int n, int m){
    int i,j;
    double **mat = (double**)malloc((n) * sizeof(double*));
    for ( i = 0; i < n; i++)
    {
        mat[i] = (double*)malloc(m * sizeof(double));
    }
    for ( i = 0; i < n; i++)
    {
        for ( j = 0; j < m; j++)
        {
            mat[i][j] = A[i+1][j];
        }
    }
    return mat;
}
/*
    # gets double matrix X of N 'd' dimension vectors
    # return new matrix W: the wheights matrix
*/
void compute_w(double** X, int N, int d, double** W) {
    /*Compute W matrix*/
    int i = 0, j = 0, k = 0;
    double distance, diff, val;
    for (i = 0; i < N; i++) 
    {
        for (j = 0; j < N; j++) 
        {
            if (i == j) {
                W[i][j] = 0.0;
            } else {
                distance = 0.0;
                for (k = 0; k < d; k++) {
                    diff = X[i][k] - X[j][k];
                    distance += diff * diff;
                }
                val = exp(-distance / 2.0);
                W[i][j] = val;
            }
        }
    }
}
/*
    # gets the W matrix
    # return diagonal matrix D (D[i][i] is sum of row i in W, else 0)
*/
double **compute_diagonal_matrix_D(double** W, int n) {
    int i,j;
    double** D = (double**) malloc(n * sizeof(double*));
    for ( i = 0; i < n; i++) 
    {
        D[i] = (double*) calloc(n, sizeof(double));
    }
    for ( i = 0; i < n; i++) 
    {
        double sum = 0.0;
        for ( j = 0; j < n; j++) 
        {
            sum += W[i][j];
        }
        D[i][i] = sum;
    }
    return D;
}
/*
    # gets matrix A
    # finds the largest element off diagonal
    # return int array size 2 with the indexes of this element
*/
int *l_off_diag(double** A, int n) {
    int *data, i, j;
    double max_off_diag = 0.0;
    double val;
    data = malloc(2 * sizeof(int));
    for ( i = 0; i < n; i++) {
        for ( j = 0; j < n; j++) {
            val = A[i][j];
            if (i != j){
                if (val > 0 && val > max_off_diag){
                    max_off_diag = val;
                    data[0] = i;
                    data[1] = j;
                }
                if (val < 0 && (-1 * val) > max_off_diag){
                    max_off_diag = -1 * val;
                    data[0] = i;
                    data[1] = j;
                }
            }
        }
    }
    return data;
}
/*
    # gets 2 matrices A and B
    # return A - B
*/
double **compute_matrix_L(double** D, double** W, int n) {
    int i, j;
    double** L = (double**) malloc(n * sizeof(double*));
    for ( i = 0; i < n; i++) 
    {
        L[i] = (double*) malloc(n * sizeof(double));
    }
    for ( i = 0; i < n; i++) 
    {
        for ( j = 0; j < n; j++) {
            L[i][j] = D[i][j] - W[i][j];
        }
    }

    return L;
}
/*
    # gets matrix A size N*N.
    # A[k][l] is the largest element off diagonal
    # return rotation matrix P
*/
double **p_mat_compute(double **A, int N, int k, int l){
    int i,j;
    double theta = 0, t = 0, c = 0, s = 0, sign = 0;
    double **P = (double **) malloc(N * sizeof(double *));
    for ( i = 0; i < N; i++) {
        P[i] = (double *) malloc(N * sizeof(double));
    }
    theta = (A[l][l] - A[k][k])/(2 * A[k][l]);
    sign = theta >= 0? 1 :-1;
    t = (sign) / (fabs(theta) + sqrt(theta * theta + 1));
    c = 1 / (sqrt(t * t +1));
    s = t * c;
    for ( i = 0; i < N; i++)
    {
        for ( j = 0; j < N; j++)
        {
            if (i == j){
                P[i][j] = 1;
            }
            else{
                P[i][j] = 0;
            }
        }
    }
    P[k][k] = c;
    P[l][l] = c;
    P[k][l] = s;
    P[l][k] = -1 * s;
    return P;
}
/*
    # gets matrix A size N*N.
    # A[k][l] is the largest element off diagonal
    # return A' matrix (A after rotation with P)
*/
double **a_tag_mat_compute(double **A, int N, int k, int l){
    int i, j;
    double theta = 0, t = 0, c = 0, s = 0, sign = 0;
    double **B = (double **) malloc(N * sizeof(double *));
    for ( i = 0; i < N; i++) {
        B[i] = (double *) malloc(N * sizeof(double));
    }

    /* compute theta, t, c, s */
    theta = (A[l][l] - A[k][k])/(2 * A[k][l]);
    sign = theta >= 0? 1 :-1;
    t = (sign) / (fabs(theta) + sqrt((theta * theta) + 1));
    c = 1 / (sqrt((t * t) +1));
    s = t * c;

    /* compute A' */ 
    for ( i = 0; i < N; i++)
    {
        for ( j = 0; j < N; j++)
        {
            if(i != k && i != l && j == k){
                B[i][j] = c * A[i][k] - s * A[i][l];
            }else{
                if(i != k && i != l && j == l){
                B[i][j] = c * A[i][l] + s * A[i][k];
            }else{
                B[i][j] = A[i][j];
            }
            }
        }
    }
    B[k][k] = c * c * A[k][k] + s * s * A[l][l] - 2.0 * s * c * A[k][l];
    B[l][l] = s * s * A[k][k] + c * c * A[l][l] + 2.0 * s * c * A[k][l];
    B[k][l] = 0;
    return B;
}

/*
    # gets matrix A size N*N.
    # return sum squars all off diagonal elements
*/
double sum_squares_off_diag(double** A, int n) {
    double sum = 0.0;
    int i,j;
    for ( i = 0; i < n; i++) 
    {
        for ( j = 0; j < n; j++) 
        {
            if (i != j) {
                sum += A[i][j] * A[i][j];
            }
        }
    }
    return sum;
}
/*
    # gets matrix A
    # return directly diagonal D matrix using other functions
*/
double **d_matrix(double **A, int N, int d){
    double **D, **W;
    int i;
    W = (double **) malloc(N * sizeof(double *));
    for ( i = 0; i < N; i++) 
    {
        W[i] = (double *) malloc(N * sizeof(double));
    }
    compute_w(A, N, d, W);
    D = compute_diagonal_matrix_D(W, N);
    return D;

}
/*
    # gets matrix A
    # returns its Laplacian matrix
*/
double **laplacian_matrix(double **A, int N, int d){
    double **D, **W, **L;
    int i;
    W = (double **) malloc(N * sizeof(double *));
    for ( i = 0; i < N; i++) 
    {
        W[i] = (double *) malloc(N * sizeof(double));
    }
    compute_w(A, N, d, W);
    D = compute_diagonal_matrix_D(W, N);
    L = compute_matrix_L(D, W, N);
    return L;

}

/*multiply matrices C = A*B
*/
double **multiply_matrices(double **A, double **B, int N) {
    double **C = malloc(N * sizeof(double *));
    int i = 0, j = 0, k = 0;
    for (i = 0; i < N; i++) 
    {
        C[i] = malloc(N * sizeof(double));
    }
    for (i = 0; i < N; i++) 
    {
        for (j = 0; j < N; j++) 
        {
            C[i][j] = 0;
            for (k = 0; k < N; k++) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return C;
}
/*
    # deep matrices copy
*/
void copy_mat_val(double **A, double **B, int N){
    int i,j;
    for ( i = 0; i < N; i++)
    {
        for ( j = 0; j < N; j++)
        {
            A[i][j] = B[i][j];
        }
        
    }
}
/*
    # gets matrix A, size N*N , int K.
    # return the first K columns as a new matrix
*/
double **final_points(double **A, int N, int K){
    double **B = malloc(N * sizeof(double*));
    int i, j;
    for ( i = 0; i < N; i++)
    {
        B[i] = malloc(K * sizeof(double));
    }
    for ( i = 0; i < N; i++)
    {
        for ( j = 0; j < K; j++)
        {
            B[i][j] = A[i][j];
        }
    }
    return B;
}
/*
    # gets double matrix size N*M and double array size N
    # return new matrix size (N+2) * N that first line is the array and then the matrix last 
    # line is for getting k
*/
double **add_line_to_mat(double **A, double *arr, int N, int M){
    double **mat = (double**)malloc((N+2) * sizeof(double*));
    int i, j;
    for ( i = 0; i < N + 2; i++)
    {
        mat[i] = (double*)malloc(M *sizeof(double));
    }
    for ( i = 0; i < M; i++)
    {
        mat[0][i] = arr[i];
    }
    for ( i = 1; i < N + 1; i++)
    {
        for ( j = 0; j < M; j++)
        {
            mat[i][j] = A[i-1][j];
        }
    }
    mat[N+1][0] = M;
    return mat;
}
/*
    # gets N*N simmetry matrix A and number k;
    # compute jacobi matrix of A
    # if k == 0 , return first K eigenvectors, computed using eigengap heuristic
    # else , return first k eigenvectors using given k
    # if all_or_k == 1 : return as above
    # else : return all jacobi matrix
*/
double **jacobi_mat(double **A, int N, int k_opt, int all_or_k){
    double **V, **curr_P, eps, **A_tag, *eigenvalues, **ret;
    int iter = 1, i=0;
    int *vals;
    eigenvalues = malloc(N * sizeof(double));
    /* first compute */
    vals = l_off_diag(A, N);
    /* P mat compute */
    curr_P = p_mat_compute(A, N, vals[0], vals[1]);
    /* A' compute */
    A_tag = transform_matrix(A, curr_P, N);
    /* current eps */
    eps = sum_squares_off_diag(A, N) - sum_squares_off_diag(A_tag, N);
    /* current V is the first rotation matrix */
    V = curr_P;
    
    /* main loop of compute eigenvectors and values */
    while (fabs(eps) > 0.00001 && iter < 100)
    {   
        /* curr A is old A' */
        copy_mat_val(A, A_tag, N);
        /* finds the largest off diagonal element indexes */
        vals = l_off_diag(A, N);
        if (A[vals[0]][vals[1]] == 0.0000){break;};
        /* compute as above */
        curr_P = p_mat_compute(A, N, vals[0], vals[1]);
        A_tag = transform_matrix(A, curr_P, N);
        eps = sum_squares_off_diag(A, N) - sum_squares_off_diag(A_tag, N);    
        V = multiply_matrices(V, curr_P, N);   
        iter++;
    }
    /* finds N eigenvalues */
    for ( i = 0; i < N; i++)
    {
        eigenvalues[i] = A_tag[i][i];
    }
    

    
    /* return all vectors or juse k of them */
    i = k_opt;
    i = all_or_k;
    ret = add_line_to_mat(V, eigenvalues, N, N);
    return ret;
}
double euclid_dist(double *vec1, double *vec2, int k){
    int i = 0;
    double dist = 0;
    for ( i = 0; i < k; i++)
    {
        dist += (vec1[i] - vec2[i]) * (vec1[i] - vec2[i]);
    }
    return sqrt(dist);
}
int min_dist_clus(double *vector, double **clusters, int k){
    int min_idx = 0, i = 0;
    double min_val = 10*4, curr_val = 0;
    for ( i = 0; i < k; i++)
    {
        curr_val = euclid_dist(vector, clusters[i], k);
        if (curr_val < min_val){
            min_val = curr_val;
            min_idx = i;
        }
    }
    return min_idx;
}
double *avg_vec(double *vec, int k){
    double *avg = (double*)calloc(k, sizeof(double));
    int size = vec[k] , i = 0;
    if (size == 0){
        return NULL;
    }
    for ( i = 0; i < k; i++)
    {
        avg[i] = vec[i] / size;
    }
    return avg;
}
double *copy_dvec(double *vec, int k){
   double *cop_vec = (double*)calloc(k, sizeof(double)); 
   int i = 0;
   for ( i = 0; i < k; i++)
   {
    cop_vec[i] = vec[i];
   }
   return cop_vec;
}
void add_vec(double**clusters, int k, int idx, double *vec){
    int i = 0;
    for ( i = 0; i < k; i++)
    {
        clusters[idx][i] += vec[i];
    }
    clusters[idx][k] += 1;
}
/*
    # final function
*/
double **final_centroids(double **vectors, int n, int k, int *init_clus){
    int i = 0, iter = 100, count = 0, idx = 0, j = 0;
    double eps = 0.00001, *vec;
    double **new_avg_vectors = (double**)calloc(k, sizeof(double*));
    double **old_avg_vectors = (double**)calloc(k, sizeof(double*));
    double **clusters = (double**)calloc(k, sizeof(double*));
    for (i = 0; i < k; i++)
    {
        new_avg_vectors[i] = (double*)calloc(k, sizeof(double));
        old_avg_vectors[i] = (double*)calloc(k, sizeof(double));
        clusters[i] = (double*)calloc((k + 1), sizeof(double));
    }
    for ( i = 0; i < k; i++)
    {
        idx = init_clus[i];
        old_avg_vectors[i] = copy_dvec(vectors[i], k);
    }
    for ( i = 0; i < iter; i++)
    {
        for ( j = 0; j < n; j++)
        {
            idx = min_dist_clus(vectors[j], old_avg_vectors, k);
            add_vec(clusters, k, idx, vectors[j]);
        }
        for (j = 0; j < k; j++){
            vec = avg_vec(clusters[j], k);
            if (vec == NULL){
                new_avg_vectors[j] = copy_dvec(old_avg_vectors[j], k);
            }else{
                new_avg_vectors[j] = vec;
            }
            if (euclid_dist(old_avg_vectors[j], new_avg_vectors[j], k) < eps){
                count ++;
            }
            old_avg_vectors[j] = copy_dvec(new_avg_vectors[j], k);
        }
        if(count == k){
            break;
        }
        count = 0;
    
    }
    return new_avg_vectors;
}
int main(int argc, char **argv){
    char *goal = argv[1], *filename = argv[2];
    double **matrix = mat_from_file(filename), **X, **goal_mat;
    int N = matrix[0][0];
    int d = matrix[0][1];
    int i;
    printf(" ");
    argc++;
    X = delete_first_line(matrix, N, d);
    if (strcmp(goal,"wam") == 0){
        goal_mat = (double**)malloc((N) * sizeof(double*));
        for ( i = 0; i < N; i++)
        {
            goal_mat[i] = (double*)malloc(d * sizeof(double));
        }
        compute_w(X, N, d, goal_mat);
        printMatrix(goal_mat, N, N);
        free_mat(X, N);
        return 0;
    }
    if (strcmp(goal,"ddg") == 0){
        goal_mat = d_matrix(X, N, d);
        printMatrix(goal_mat, N, N);  
        free_mat(X, N);
        return 0;
      
    }
    if (strcmp(goal,"gl") == 0)
    {
        goal_mat = laplacian_matrix(X, N, d);
        printMatrix(goal_mat, N, N);
        free_mat(X, N);
        return 0;       
    }
    if (strcmp(goal,"jacobi") == 0)
    {
        goal_mat = jacobi_mat(X, N, 0, 0);
        printMatrix(goal_mat, N+1, N);
        free_mat(X, N);
        return 0;
    }  
    printf("An Error Has Occurred");
    return 0;
}
