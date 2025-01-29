#include "cml_data.h"
#include "cml_sequential.h"
#include "cml_prng.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

cml_prng *prng = NULL;

// f(x) = 3x + 2 + error
static fdouble f(const fdouble x)
{
    const fdouble err = prng->normal(prng, 0., 0.02);
    return 3 * x + 2 + err;
}

static fdouble learning_rate(const fdouble alpha)
{
    return alpha;
}

int main(void)
{
    prng = cml_prng_init(NULL);

    cml_matrix *x = cml_matrix_zeros(100, 1);
    cml_matrix *y = cml_matrix_zeros(100, 1);
    for (lgint i = 0; i < x->m; i++)
    {
        for (lgint j = 0; j < x->n; j++)
        {
            const fdouble xij = prng->normal(prng, 7, 10);
            const fdouble yij = f(xij);
            x->set(&x, i, j, xij);
            y->set(&y, i, j, yij);
        }
    }

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
        x, y,
        val_percentage, test_percentage, shuffle);

    cml_layer *layers[] = {
        cml_layer_create(32, RELU),
        cml_layer_create(1, LINEAR)};
    const lgint n_layers = sizeof(layers) / sizeof(layers[0]);
    cml_sequential *model = cml_sequential_create(layers, n_layers, train_data_x->n, SQUARED_ERROR_LOSS);
    model->compile(model, prng);
    model->summary(model);

    const fdouble alpha = 0.05;
    const lgint epochs = 15000;
    model->fit(model, train_data_x, train_data_y, &learning_rate, alpha, epochs);

    cml_matrix *yhat = model->predict(model, test_data_x);
    if (yhat)
    {
        //yhat->print(yhat);

        fdouble mae, mse, rmse, rsquared, arsquared, mape, smape, hloss, evars, medae;
        const lgint k = train_data_x->n; // number of predictors (independent variables)
        const fdouble threshold = 0.2;
        printf("\nRegression metrics\n");
        printf("-------------------------------------------------------------------\n");
        //cml_reg_metrics(&mae, &mse, &rmse, &rsquared, &arsquared, yhat, y_test, k, threshold);
        cml_reg_metrics(&mae, &mse, &rmse, &rsquared, &arsquared, &mape, &smape, &hloss, &evars, &medae, yhat, test_data_y, k, threshold);

        printf("Mean Absolute Error (MAE)                       : %lg\n", mae);
        printf("Mean Squared Error (MSE)                        : %lg\n", mse);
        printf("Root Mean Squared Error (RMSE)                  : %lg\n", rmse);
        printf("R-Squared (R²)                                  : %lg\n", rsquared);
        printf("Adjusted R-Squared (aR²)                        : %lg\n", arsquared);
        printf("Mean Absolute Percentage Error (MAPE)           : %lg\n", mape);
        printf("Symmetric Mean Absolute Percentage Error (sMAPE): %lg\n", smape);
        printf("Huber Loss (HLoss)                              : %lg\n", hloss);
        printf("Explained Variance Score(EVS)                   : %lg\n", evars);
        printf("Median Absolute Error(MAE)                      : %lg\n", medae);
        printf("-------------------------------------------------------------------\n");

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

    model->free(&model);
    prng->free(&prng);

    return EXIT_SUCCESS;
}