#include "cml_sequential.h"

#include <stdio.h>
#include <stdlib.h>

static fdouble learning_rate(const fdouble alpha)
{
    return 1.00001 * alpha;
}

int main(void)
{
    cml_prng *prng = cml_prng_init(NULL);

    cml_matrix *x = cml_matrix_alloc(4, 2);
    x->set(&x, 0, 0, 0);
    x->set(&x, 0, 1, 0);
    x->set(&x, 1, 0, 0);
    x->set(&x, 1, 1, 1);
    x->set(&x, 2, 0, 1);
    x->set(&x, 2, 1, 0);
    x->set(&x, 3, 0, 1);
    x->set(&x, 3, 1, 1);

    cml_matrix *y = cml_matrix_alloc(4, 2);
    y->set(&y, 0, 0, 1);
    y->set(&y, 0, 1, 0);
    y->set(&y, 1, 0, 0);
    y->set(&y, 1, 1, 1);
    y->set(&y, 2, 0, 0);
    y->set(&y, 2, 1, 1);
    y->set(&y, 3, 0, 1);
    y->set(&y, 3, 1, 0);

    cml_layer *layers[] = {
        cml_layer_create(45, RELU),
        cml_layer_create(2, SOFTMAX)};
    const lgint n_layers = sizeof(layers) / sizeof(layers[0]);
    cml_sequential *model = cml_sequential_create(layers, n_layers, x->n, MULTI_CLASS_CROSS_ENTROPY);
    model->compile(model, prng);
    model->summary(model);

    const fdouble alpha = 0.001;
    const lgint epochs = 100000;
    model->fit(model, x, y, &learning_rate, alpha, epochs);

    y->print(y);
    cml_matrix *yhat = model->predict(model, x);
    if (yhat)
    {
        yhat->softmax(&yhat);
        yhat->print(yhat);

        cml_matrix *conf = cml_matrix_confusion(yhat, y);
        conf->print(conf);
        conf->free(&conf);

        cml_matrix *prec = NULL, *accur = NULL, *f1_score = NULL;
        cml_class_metrics(&prec, &accur, &f1_score, yhat, y);
        prec->print(prec);
        accur->print(accur);
        f1_score->print(f1_score);
        prec->free(&prec);
        accur->free(&accur);
        f1_score->free(&f1_score);

        yhat->free(&yhat);
    }

    x->free(&x);
    y->free(&y);
    model->free(&model);
    prng->free(&prng);

    return EXIT_SUCCESS;
}