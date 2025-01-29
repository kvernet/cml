#include "cml_data.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void)
{
    srand(time(0));

    cml_matrix *train_data_x, *train_data_y;
    cml_matrix *val_data_x, *val_data_y;
    cml_matrix *test_data_x, *test_data_y;

    cml_matrix *x = cml_matrix_eye(15);
    cml_matrix *y = cml_matrix_zeros(15, 1);

    const fdouble val_percentage = 0.2;
    const fdouble test_percentage = 0.15;
    bool shuffle = true;

    cml_data_split(
        &train_data_x, &train_data_y,
        &val_data_x, &val_data_y,
        &test_data_x, &test_data_y,
        x, y,
        val_percentage, test_percentage, shuffle);
    
    train_data_x->print(train_data_x);
    val_data_x->print(val_data_x);
    test_data_x->print(test_data_x);

    train_data_x->free(&train_data_x);
    train_data_y->free(&train_data_y);
    val_data_x->free(&val_data_x);
    val_data_y->free(&val_data_y);
    test_data_x->free(&test_data_x);
    test_data_y->free(&test_data_y);
    x->free(&x);
    y->free(&y);

    return EXIT_SUCCESS;
}