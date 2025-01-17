#include "cml_sequential.h"
#include "../matrix/matrix_header.h"
#include "cml_prng.h"

#include <stdlib.h>

int main(void)
{
    cml_prng *prng = cml_prng_init(NULL);
    
    const lgint n_inputs = 3;
    const lgint n_layers = 4;

    cml_layer *layers[] = {
        cml_layer_create(256, LINEAR),
        cml_layer_create(128, RELU),
        cml_layer_create(64, RELU),
        cml_layer_create(1, SIGMOID)};

    cml_sequential *model = cml_sequential_create(layers, n_layers, n_inputs);
    model->compile(model, prng);
    model->summary(model);

    model->free(&model);
    prng->free(&prng);
    
    return EXIT_SUCCESS;
}