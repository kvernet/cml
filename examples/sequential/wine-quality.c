#include "cml_data.h"
#include "cml_sequential.h"
#include "cml_prng.h"

#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static fdouble learning_rate(const fdouble alpha)
{
    return 1.001 * alpha;
}

int main(void)
{
    cml_prng *prng = cml_prng_init(NULL);

    cml_matrix *x = cml_matrix_zeros(4898, 11);
    cml_matrix *y = cml_matrix_zeros(4898, 11);
    const char *file_path = "data/winequality-white.data";
    const char *delimiter = ",";
    bool has_header = true;
    cml_data_read(&x, &y, file_path, delimiter, has_header);
    // normalize the data
    cml_matrix *x_normalized = x->normalize(x);

    // split data into train, validation & test
    cml_matrix *train_data_x = NULL, *train_data_y = NULL;
    cml_matrix *val_data_x = NULL, *val_data_y = NULL;
    cml_matrix *test_data_x = NULL, *test_data_y = NULL;
    const fdouble val_percentage = 0.25;
    const fdouble test_percentage = 0.15;
    bool shuffle = false;
    cml_data_split(
        &train_data_x, &train_data_y,
        &val_data_x, &val_data_y,
        &test_data_x, &test_data_y,
        x_normalized, y,
        val_percentage, test_percentage, shuffle);

    cml_layer *layers[] = {
        cml_layer_create(59, LEAKY_RELU),
        cml_layer_create(y->n, SOFTMAX)};
    const lgint n_layers = sizeof(layers) / sizeof(layers[0]);
    cml_sequential *model = cml_sequential_create(layers, n_layers, train_data_x->n, MULTI_CLASS_CROSS_ENTROPY);
    model->compile(model, prng);
    model->summary(model);

    const fdouble alpha = 0.01;
    const lgint epochs = 500;
    model->fit(model, train_data_x, train_data_y, &learning_rate, alpha, epochs);

    printf("Cross check validation\n");
    cml_matrix *yhat = model->predict(model, val_data_x);
    if(yhat)
    {
        yhat->softmax(&yhat);

        printf("Confusion matrix=");
        cml_matrix *conf = cml_matrix_confusion(yhat, val_data_y);
        conf->print(conf);
        conf->free(&conf);

        cml_matrix *prec = NULL, *accur = NULL, *f1_score = NULL;
        cml_class_metrics(&prec, &accur, &f1_score, yhat, val_data_y);
        printf("Precision matrix=");
        prec->print(prec);
        printf("Accuracy matrix=");
        accur->print(accur);
        printf("F1 score matrix=");
        f1_score->print(f1_score);
        prec->free(&prec);
        accur->free(&accur);
        f1_score->free(&f1_score);

        yhat->free(&yhat);
    }

    printf("Metrics\n");
    yhat = model->predict(model, test_data_x);
    if(yhat)
    {
        yhat->softmax(&yhat);

        printf("Confusion matrix=");
        cml_matrix *conf = cml_matrix_confusion(yhat, test_data_y);
        conf->print(conf);
        conf->free(&conf);

        cml_matrix *prec = NULL, *accur = NULL, *f1_score = NULL;
        cml_class_metrics(&prec, &accur, &f1_score, yhat, test_data_y);
        printf("Precision matrix=");
        prec->print(prec);
        printf("Accuracy matrix=");
        accur->print(accur);
        printf("F1 score matrix=");
        f1_score->print(f1_score);
        prec->free(&prec);
        accur->free(&accur);
        f1_score->free(&f1_score);

        yhat->free(&yhat);
    }

    train_data_x->free(&train_data_x);
    train_data_y->free(&train_data_y);
    val_data_x->free(&val_data_x);
    val_data_y->free(&val_data_y);
    test_data_x->free(&test_data_x);
    test_data_y->free(&test_data_y);
    x->free(&x);
    y->free(&y);
    x_normalized->free(&x_normalized);

    model->free(&model);
    prng->free(&prng);

    return EXIT_SUCCESS;
}