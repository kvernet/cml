#include "cml_sequential.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void)
{
    srand(time(0));

    cml_matrix *x = cml_matrix_alloc(4, 2);
    x->set(&x, 0, 0, 0);
    x->set(&x, 0, 1, 0);
    x->set(&x, 1, 0, 0);
    x->set(&x, 1, 1, 1);
    x->set(&x, 2, 0, 1);
    x->set(&x, 2, 1, 0);
    x->set(&x, 3, 0, 1);
    x->set(&x, 3, 1, 1);

    cml_matrix *y = cml_matrix_alloc(4, 1);
    y->set(&y, 0, 0, 0);
    y->set(&y, 1, 0, 1);
    y->set(&y, 2, 0, 1);
    y->set(&y, 3, 0, 0);

    cml_layer *layers[] = {
        cml_layer_create(4, TANH),
        cml_layer_create(1, SIGMOID)};
    const lgint n_layers = sizeof(layers) / sizeof(layers[0]);
    cml_sequential *model = cml_sequential_create(layers, n_layers, x->n);
    model->compile(model);
    model->summary(model);

    const fdouble alpha = 0.001;
    const lgint epochs = 500000;
    model->fit(model, x, y, alpha, epochs);

    cml_matrix *yhat = model->predict(model, x);
    if (yhat)
    {
        yhat->print(yhat);
        yhat->free(&yhat);
    }

    x->free(&x);
    y->free(&y);
    model->free(&model);

    return EXIT_SUCCESS;
}