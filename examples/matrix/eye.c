#include "matrix_header.h"

int main(void)
{
    cml_matrix *a = cml_matrix_eye(3);
    a->print(a);

    a->free(&a);
    return EXIT_SUCCESS;
}