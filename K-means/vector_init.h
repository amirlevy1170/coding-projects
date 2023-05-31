#ifndef __VECTOR_INIT_H__
#define __VECTOR_INIT_H__

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

void free_cord (struct cord *cords);
void free_vec (struct vector *vec);
struct vector* copy_vec(struct vector *vec);
double euclidian_distance(struct vector *vec1,struct vector *vec2);
int get_vec_len(struct vector *vec);
struct vector* avg_clust(struct vector *vec1);




#endif