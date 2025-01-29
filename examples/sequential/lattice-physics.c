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

    // get train data
    cml_matrix *x_train = cml_matrix_zeros(24000, 40);
    cml_matrix *y_train = cml_matrix_zeros(24000, 1);
    const char *file_path = "data/lattice-physics+(pwr+fuel+assembly+neutronics+simulation+results)/raw.data";
    const char *delimiter = ",";
    bool has_header = false;
    cml_data_read(&x_train, &y_train, file_path, delimiter, has_header);
    cml_matrix *x_train_normalized = x_train->normalize(x_train);

    // get test data
    cml_matrix *x_test = cml_matrix_zeros(360, 40);
    cml_matrix *y_test = cml_matrix_zeros(360, 1);
    file_path = "data/lattice-physics+(pwr+fuel+assembly+neutronics+simulation+results)/test.data";
    cml_data_read(&x_test, &y_test, file_path, delimiter, has_header);
    cml_matrix *x_test_normalized = x_train->normalize(x_test);

    cml_layer *layers[] = {
        cml_layer_create(64, LEAKY_RELU),
        cml_layer_create(32, LEAKY_RELU),
        cml_layer_create(y_train->n, LINEAR)};
    const lgint n_layers = sizeof(layers) / sizeof(layers[0]);
    cml_sequential *model = cml_sequential_create(layers, n_layers, x_train->n, SQUARED_ERROR_LOSS);
    model->compile(model, prng);
    model->summary(model);

    const fdouble alpha = 0.1;
    const lgint epochs = 2;//1000;
    model->fit(model, x_train_normalized, y_train, &learning_rate, alpha, epochs);

    //y_test->print(y_test);
    cml_matrix *yhat = model->predict(model, x_test_normalized);
    if (yhat)
    {
        //yhat->print(yhat);

        fdouble mae, mse, rmse, rsquared, arsquared, mape, smape, hloss, evars, medae;
        const lgint k = x_train->n; // number of predictors (independent variables)
        const fdouble threshold = 0.2;
        printf("\nRegression metrics\n");
        printf("-------------------------------------------------------------------\n");
        //cml_reg_metrics(&mae, &mse, &rmse, &rsquared, &arsquared, yhat, y_test, k, threshold);
        cml_reg_metrics(&mae, &mse, &rmse, &rsquared, &arsquared, &mape, &smape, &hloss, &evars, &medae, yhat, y_test, k, threshold);

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

    x_train->free(&x_train);
    y_train->free(&y_train);
    x_test->free(&x_test);
    y_test->free(&y_test);
    x_train_normalized->free(&x_train_normalized);
    x_test_normalized->free(&x_test_normalized);

    model->free(&model);
    prng->free(&prng);

    return EXIT_SUCCESS;
}