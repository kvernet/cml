#include "matrix_header.h"

#include <stdio.h>
#include <time.h>

int main(void)
{
    srand(time(NULL));

    cml_matrix *a = cml_matrix_alloc(3, 2);
    matrix_random_fill(&a, 10);
    a->print(a);

    printf("==== tanspose =====\n");
    cml_matrix *at = NULL;
    a->transpose(a, &at);
    if (at != NULL)
    {
        at->print(at);
        at->free(&at);
    }

    a->free(&a);
    return EXIT_SUCCESS;
}