#include "matrix_header.h"

#include <stdio.h>
#include <time.h>

int main(void)
{
    srand(time(NULL));

    cml_matrix *a = cml_matrix_alloc(3, 4);
    matrix_random_fill(&a, 5);
    a->print(a);

    printf("+\n");

    cml_matrix *b = cml_matrix_alloc(3, 4);
    matrix_random_fill(&b, 5);
    b->print(b);

    printf("==== sum =====\n");
    cml_matrix *s = cml_matrix_sum(a, b);
    if (s != NULL)
    {
        s->print(s);
        s->free(&s);
    }

    a->free(&a);
    b->free(&b);
    return EXIT_SUCCESS;
}