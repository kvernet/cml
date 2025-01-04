#include "matrix_header.h"

#include <time.h>

int main(void)
{
    srand(time(NULL));

    cml_matrix *a = cml_matrix_alloc(3, 4);
    matrix_random_fill(&a, 10);
    a->print(a);

    a->free(&a);
    return EXIT_SUCCESS;
}