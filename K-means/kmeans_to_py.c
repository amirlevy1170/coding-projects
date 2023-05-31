# include <stdio.h>
# include <stdlib.h>
# include "kmeans.h"
# include "vector_init.c"


/*
    # input:clusters, number j and vector
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
    # input: means, data vectors, number K and clusters
    # sort clusters by euclidian distance from means
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
/*
    # input: file name, head veactor
    # write file data to vector
    # output: size number of vector in file, 0 if fopen failed
*/
int parseFile(const char *infile_name,struct vector *head_vec, struct vector *curr_vec)
{

    FILE *infile = NULL;
    int size = 0;
    struct cord *head_cord = NULL, *curr_cord = NULL;
    double n = 0.;
    char c = 0;
    

    infile = fopen(infile_name, "r");

    if (infile == NULL)
    {
        return 0;
    }
    head_cord = calloc(1, sizeof(struct cord));
    curr_cord = head_cord;
    curr_cord->next = NULL;

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
        curr_cord->value = n;
        curr_cord->next = calloc(1, sizeof(struct cord));
        curr_cord = curr_cord->next;
        curr_cord->next = NULL;    
    }
    fclose(infile);
    infile = NULL;


    return size;
}
/*
    # input: means matrix, head vector, array of indexes, size of array, number K
    # initialize means matrix to the vectors corresponding the indexes in the array
    # output: first kmneans vectors
*/
struct vector** kmeans_init(struct vector *curr_vec, int *vec_idxs, int vec_idxs_size, int K){
    struct vector **k_mean;
    int i = 0, j = 0, h = 0;
    
    k_mean = (struct vector **)calloc(K,sizeof(struct vector*));

    while (i < K)
    {   
        for ( h = 0; h < vec_idxs_size; h++)
        {
            if(j == vec_idxs[h]){
                k_mean[i] = copy_vec(curr_vec);
                i++;
            }
        }
        curr_vec = curr_vec->next;
        j++;
    }
    return k_mean;
}

/*
    # input: clusters, means, number of iterations, head vector, number K, epsilon
    # calculate the final K means vectors
*/
void calculateMeans(struct vector **k_mean, int iter,struct vector *head_vec,int K, double eps )
{
    struct vector *curr_vec, *temp_vec, **clus;
    int i = 0, h = 0, counter = 0;

    clus = (struct vector **)calloc(K,sizeof(struct vector*));

    for ( h = 0; h < iter; h++)
    {
        counter = 0;
        for ( i = 0; i < K; i++)
        {
            clus[i] = NULL;
        }
        clusters_maker(clus,k_mean,head_vec,K);
        clus[0] = clus[0]->next;
        for (i = 0; i < K; i++)
        {   
            curr_vec = k_mean[i];
            temp_vec = avg_clust(clus[i]);
            if (euclidian_distance(curr_vec,temp_vec) <= eps)
            {
                counter++;
            }
            free_vec(k_mean[i]);
            k_mean[i] = temp_vec;

        }
        if (counter == K){
            break;}
        for (i = 0; i < K; i++)
        {
            free_vec(clus[i]);
        }
    }
    for (i = 0; i < K; i++)
        {
            free_vec(clus[i]);
        }

}

/*
    # input: empty double matrix, len of cords, number K, final k means
    # put the final k means in the double matrix
    # output: final double matrix
*/
double** kmeansToDoubleArray(int num_cord, int K, struct vector **k_mean)
{
    struct cord *curr_cord = NULL;
    int i = 0, j = 0;
    double **dKmeans;

    dKmeans = (double**)calloc(K, sizeof(double*));
    for (i = 0; i < K; i++)
    {
        dKmeans[i] = (double*)calloc(num_cord, sizeof(double));
    }
    
    for ( i = 0; i < K; i++)
    {
        curr_cord = k_mean[i]->cords;
        j=0;
        while (curr_cord != NULL)
        {
            dKmeans[i][j] = curr_cord->value;
            curr_cord = curr_cord->next;
            j++;
        }
    }
    return dKmeans;
}
/*
    # free all used memmory
*/
void cleanup(struct vector *head_vec,struct vector *curr_vec,struct vector **k_mean, int K )
{
    int i = 0;

    for ( i = 0; i < K; i++)
    {
        free_vec(k_mean[i]);
    }

    free(k_mean);
    free_vec(head_vec);
}

/* main function */
double** kmeans_main(int K, int iter, int *vec_idxs, int vec_idxs_size, const char *infile_name, double eps)
{
    /* read all data*/
    int size = 0,num_cord = 0;
    struct vector *head_vec = NULL, *curr_vec = NULL, **k_mean = NULL;
    double **dKmeans = NULL;
    

    head_vec = calloc(1 ,sizeof(struct vector));
    curr_vec = head_vec;
    curr_vec->next = NULL;

    /* validate number of iterations*/
    if (iter < 2 || iter >1000){
        printf("Invalid maximum iteration!");
        return NULL;
    }

    /* parsing all file to head vector */
    size = parseFile(infile_name,head_vec, curr_vec);
    
    /* validate size of data is at least K+1*/
    if (size <= K)
    {
        printf("Invalid number of clusters!");
        return NULL;
    }

    curr_vec = head_vec;
    num_cord = get_vec_len(curr_vec);

    /* initialize first k means as given from the kmeans ++ algorithm */
    k_mean = kmeans_init(curr_vec, vec_idxs, vec_idxs_size, K);

    /*main loop runs "iter" times or delta distance < 0.001 for all clusters*/
    calculateMeans(k_mean, iter,head_vec, K, eps);

    /* move final k means to double matrix*/
    dKmeans = kmeansToDoubleArray(num_cord, K, k_mean);

    /* free memory */
    cleanup(head_vec, curr_vec, k_mean, K);

    return dKmeans;
}
