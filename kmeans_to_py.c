# include <stdio.h>
# include <stdlib.h>
# include "kmeans.h"

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



double** kmeans_main(int K, int iter, int *vec_idxs, int vec_idxs_size, const char *infile_name, double eps)
{
    /* read all data*/
    int i = 0, h = 0, counter = 0, size = 0, j = 0, num_cord = 0;
    FILE *infile = NULL;
    struct vector *head_vec = NULL, *curr_vec = NULL, *temp_vec = NULL, **k_mean = NULL, **clus = NULL;
    struct cord *head_cord = NULL, *curr_cord = NULL;
    double n = 0.;
    double **dKmeans;
    char c = 0;
    infile = fopen(infile_name, "r");
    if (infile == NULL)
    {
        return NULL;
    }
    k_mean = (struct vector **)calloc(K,sizeof(struct vector));
    clus = (struct vector **)calloc(K,sizeof(struct vector));
    if (iter < 2 || iter >1000){
        printf("Invalid maximum iteration!");
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
        curr_cord->value = n;
        curr_cord->next = calloc(1, sizeof(struct cord));
        curr_cord = curr_cord->next;
        curr_cord->next = NULL;    
    }
    fclose(infile);
    infile = NULL;
    /* 
    build first K means with the centroids we got from python 
    */
    
    if (size <= K)
    {
        printf("Invalid number of clusters!");
        return NULL;
    }
    
    curr_vec = head_vec;
    curr_cord = curr_vec->cords;
    num_cord = get_vec_len(curr_vec);
    /*
        build the double matrix to return K means
    */
    dKmeans = (double**)calloc(K, sizeof(double*));
    for (i = 0; i < K; i++)
    {
        dKmeans[i] = (double*)calloc(num_cord, sizeof(double));
    }
    
    i = 0;
    j = 0;
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
    
    
    /*main loop runs "iter" times or delta distance < 0.001 for all clusters*/
    
    
    
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
        if (counter == K){break;}
        for (i = 0; i < K; i++)
        {
            free_vec(clus[i]);
        }
    }
    for (i = 0; i < K; i++)
        {
            free_vec(clus[i]);
        }
    i = 0;
    /* move final k means to double matrix*/
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
    for ( i = 0; i < K; i++)
    {
        free_vec(k_mean[i]);
    }
    free(k_mean);
    free(clus);
    free_vec(head_vec);
    free_cord(head_cord);
    free_cord(curr_cord);
    return dKmeans;
}
