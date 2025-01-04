#include "matrix_header.h"

#include <stdio.h>
#include <time.h>

int main(void)
{
    srand(time(NULL));

    cml_matrix *a = cml_matrix_alloc(3, 3);
    matrix_random_fill(&a, 10);
    a->print(a);

    printf("det = %lg\n", a->det(a));

    a->free(&a);
    return EXIT_SUCCESS;
}