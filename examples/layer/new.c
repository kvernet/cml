#include "cml_layer.h"
#include "../matrix/matrix_header.h"

#include <stdlib.h>
#include <time.h>

int main(void)
{
    srand(time(NULL));

    const lgint units = 4;
    const lgint n_inputs = 3;
    cml_layer *layer = cml_layer_create(units, LINEAR);
    layer->compile(layer, n_inputs);
    layer->print(layer);

    cml_matrix *x = cml_matrix_alloc(5, n_inputs);
    matrix_random_fill(&x, 10);
    x->print(x);

    cml_matrix *y = layer->eval(layer, x);
    if (y != NULL)
    {
        y->print(y);
        y->free(&y);
    }

    x->free(&x);

    layer->free(&layer);
    return EXIT_SUCCESS;
}