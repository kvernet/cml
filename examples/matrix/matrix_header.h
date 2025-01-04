#include "cml_matrix.h"

#include <stdlib.h>

void matrix_random_fill(cml_matrix **a, const lgint max)
{
    if (*a == NULL)
        return;

    for (lgint i = 0; i < (*a)->m; i++)
    {
        for (lgint j = 0; j < (*a)->n; j++)
        {
            (*a)->set(a, i, j, rand() % max);
        }
    }
}