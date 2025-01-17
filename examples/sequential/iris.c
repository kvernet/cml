#include "cml_sequential.h"
#include "cml_prng.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void get_iris_data(cml_matrix **x, cml_matrix **y, const char *path);

int main(void)
{
    cml_prng *prng = cml_prng_init(NULL);

    cml_matrix *x = cml_matrix_alloc(150, 4);
    cml_matrix *y = cml_matrix_alloc(150, 1);
    const char *path = "data/iris.data";
    get_iris_data(&x, &y, path);
    // x->print(x);
    // y->print(y);

    cml_layer *layers[] = {
        cml_layer_create(56, LINEAR),
        cml_layer_create(28, LINEAR),
        cml_layer_create(1, RELU)};
    const lgint n_layers = sizeof(layers) / sizeof(layers[0]);
    cml_sequential *model = cml_sequential_create(layers, n_layers, x->n);
    model->compile(model, prng);
    model->summary(model);

    const fdouble alpha = 0.01;
    const lgint epochs = 10000;
    model->fit(model, x, y, alpha, epochs);

    cml_matrix *x_test = cml_matrix_alloc(1, 4);
    x_test->set(&x_test, 0, 0, 6.2);
    x_test->set(&x_test, 0, 1, 2.2);
    x_test->set(&x_test, 0, 2, 4.5);
    x_test->set(&x_test, 0, 3, 1.5);

    cml_matrix *yhat = model->predict(model, x_test);
    yhat->print(yhat);
    x_test->free(&x_test);

    printf("seed = %d\n", prng->seed);

    x->free(&x);
    y->free(&y);
    model->free(&model);
    yhat->free(&yhat);
    prng->free(&prng);

    return EXIT_SUCCESS;
}

void get_iris_data(cml_matrix **x, cml_matrix **y, const char *path)
{
    FILE *stream = fopen(path, "r");
    if (stream == NULL)
    {
        fprintf(stderr, "Error (get_iris_data): file %s does not exist\n", path);
        return;
    }
    char dummy[255];

    fdouble sepal_length, sepal_width, petal_length, petal_width;
    char name[100];

    lgint i = 0;
    while (fgets(dummy, 255, stream))
    {
        if (5 == sscanf(dummy, "%lg,%lg,%lg,%lg,%s\n", &sepal_length, &sepal_width, &petal_length, &petal_width, name))
        {
            (*x)->set(x, i, 0, sepal_length);
            (*x)->set(x, i, 1, sepal_width);
            (*x)->set(x, i, 2, petal_length);
            (*x)->set(x, i, 3, petal_width);

            if (strcmp(name, "Iris-setosa") == 0)
            {
                (*y)->set(y, i, 0, 1);
            }
            else if (strcmp(name, "Iris-versicolor") == 0)
            {
                (*y)->set(y, i, 0, 2);
            }
            else if (strcmp(name, "Iris-virginica") == 0)
            {
                (*y)->set(y, i, 0, 3);
            }
            else
            {
                (*y)->set(y, i, 0, 0);
            }

            i++;
        }
    }
    fclose(stream);
}