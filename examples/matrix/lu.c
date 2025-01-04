#include "matrix_header.h"

#include <stdio.h>
#include <time.h>

int main(void)
{
    srand(time(NULL));

    cml_matrix *a = cml_matrix_alloc(3, 3);
    matrix_random_fill(&a, 10);
    a->print(a);

    cml_matrix *p = NULL, *l = NULL, *u = NULL;
    a->lu(a, &p, &l, &u);
    if(p != NULL && l != NULL && u != NULL)
    {
        p->print(p);
        l->print(l);
        u->print(u);

        printf("==== check LU = PA ====\n");
        cml_matrix *lu = cml_matrix_prod(l, u);
        cml_matrix *pa = cml_matrix_prod(p, a);
        lu->print(lu);
        pa->print(pa);
        lu->free(&lu);
        pa->free(&pa);

        p->free(&p);
        l->free(&l);
        u->free(&u);
    }

    a->free(&a);
    return EXIT_SUCCESS;
}