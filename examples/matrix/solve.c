#include "matrix_header.h"

#include <stdio.h>
#include <time.h>

int main(void)
{
    srand(time(NULL));

    cml_matrix *a = cml_matrix_alloc(3, 3);
    matrix_random_fill(&a, 5);
    a->print(a);

    cml_matrix *b = cml_matrix_alloc(3, 1);
    matrix_random_fill(&b, 5);
    b->print(b);

    printf("==== solution of Ax=b =====\n");
    cml_matrix *x = cml_matrix_solve(a, b);
    if (x != NULL)
    {
        x->print(x);

        printf("==== compute A^-1*b ====\n");
        cml_matrix *bprime = cml_matrix_prod(a, x);
        bprime->print(bprime);
        bprime->free(&bprime);

        x->free(&x);
    }

    a->free(&a);
    b->free(&b);
    return EXIT_SUCCESS;
}