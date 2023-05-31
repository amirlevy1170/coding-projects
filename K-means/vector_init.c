# include <stdio.h>
# include <stdlib.h>
# include "vector_init.h"


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