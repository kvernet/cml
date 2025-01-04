#include "matrix_header.h"

#include <stdio.h>
#include <time.h>

int main(void)
{
    srand(time(NULL));

    cml_matrix *a = cml_matrix_alloc(4, 4);
    matrix_random_fill(&a, 5);
    a->print(a);

    printf("==== inverse ====\n");

    cml_matrix *ainv = a->inv(a);
    if(ainv != NULL)
    {
        ainv->print(ainv);

        printf("==== A*A^-1 ====\n");
        cml_matrix *eye = cml_matrix_prod(a, ainv);
        eye->print(eye);
        eye->free(&eye);

        printf("==== A^-1*A ====\n");
        eye = cml_matrix_prod(ainv, a);
        eye->print(eye);
        eye->free(&eye);

        ainv->free(&ainv);
    }

    a->free(&a);
    return EXIT_SUCCESS;
}