#include "matrix_header.h"

#include <stdio.h>
#include <time.h>

int main(void)
{
    srand(time(NULL));

    cml_matrix *a = cml_matrix_alloc(3, 4);
    matrix_random_fill(&a, 5);
    a->print(a);

    printf("*\n");

    cml_matrix *b = cml_matrix_alloc(4, 3);
    matrix_random_fill(&b, 5);
    b->print(b);

    printf("==== product =====\n");
    cml_matrix *p = cml_matrix_prod(a, b);
    if (p != NULL)
    {
        p->print(p);
        p->free(&p);
    }

    a->free(&a);
    b->free(&b);
    return EXIT_SUCCESS;
}