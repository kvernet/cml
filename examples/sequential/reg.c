#include "cml_sequential.h"
#include "../matrix/matrix_header.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int, char **argv)
{
    srand(time(NULL));

    const int epochs = (argv[1] != NULL) ? strtold(argv[1], NULL) : 1000;

    const lgint n_inputs = 2;
    const lgint n_layers = 1;
    const lgint n = 3;

    cml_layer *layers[] = {
        cml_layer_create(1, LINEAR)};

    cml_sequential *model = cml_sequential_create(layers, n_layers, n_inputs);
    model->compile(model);
    model->summary(model);

    cml_matrix *x = cml_matrix_alloc(n, n_inputs);
    matrix_random_fill(&x, 10);

    cml_matrix *y = cml_matrix_alloc(n, 1);
    matrix_random_fill(&y, 10);

    const fdouble alpha = 0.01;
    model->fit(model, x, y, alpha, epochs);

    printf("x = ");
    x->print(x);

    printf("y = ");
    y->print(y);

    cml_matrix *yhat = model->predict(model, x);
    printf("yhat = ");
    yhat->print(yhat);

    x->free(&x);
    y->free(&y);
    yhat->free(&yhat);

    model->free(&model);
    return EXIT_SUCCESS;
}