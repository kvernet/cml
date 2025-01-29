#include "cml_data.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void cml_data_read(cml_matrix **x, cml_matrix **y, const char *file_path, const char *delimiter, bool has_header)
{
    FILE *stream = fopen(file_path, "r");
    if (stream == NULL)
    {
        fprintf(stderr, "Error (cml_read_csv): file %s does not exist.\n", file_path);
        return;
    }
    char dummy[1024];

    // Read the header file
    if (has_header)
    {
        if (!fgets(dummy, sizeof(dummy), stream))
            return;
    }

    // Read each line from the CSV file
    lgint i = 0;
    while (fgets(dummy, sizeof(dummy), stream))
    {
        char *token;

        fdouble value = 0.;
        lgint j = 0;
        // Tokenize the line based on the delimiter
        token = strtok(dummy, delimiter);
        value = strtod(token, NULL);

        (*x)->set(x, i, j, value);
        j++;

        while (token != NULL)
        {
            token = strtok(NULL, delimiter);
            if (token == NULL)
                break;

            value = strtod(token, NULL);

            if (j < (*x)->n)
            {
                (*x)->set(x, i, j, value);
            }
            else
            {
                (*y)->set(y, i, j - (*x)->n, value);
            }

            j++;
        }
        i++;
    }

    // Close the file after reading
    fclose(stream);
}

static bool index_is_present(lgint *indices, const lgint max, const lgint index)
{
    for (lgint i = 0; i < max; i++)
    {
        if (indices[i] == index)
            return true;
    }
    return false;
}

static void sufle_indices(lgint *indices, const lgint max)
{
    for (lgint i = 0; i < max; i++)
    {
        indices[i] = INT_MAX;
    }

    lgint i = 0;
    while (i < max)
    {
        lgint index = rand() % max;
        while (index_is_present(indices, max, index))
        {
            index = rand() % max;
        }
        indices[i] = index;
        i++;
    }
}

void cml_data_split(
    cml_matrix **train_data_x, cml_matrix **train_data_y,
    cml_matrix **val_data_x, cml_matrix **val_data_y,
    cml_matrix **test_data_x, cml_matrix **test_data_y,
    cml_matrix *const x, cml_matrix *const y,
    const fdouble val_percentage, const fdouble test_percentage,
    bool shuffle)
{
    if (x == NULL || y == NULL)
        return;

    if (x->m != y->m)
        return;

    if (val_percentage + test_percentage >= 1. || val_percentage * test_percentage < 0)
        return;

    const lgint val_size = val_percentage * x->m;
    const lgint test_size = test_percentage * x->m;
    const lgint train_size = x->m - (val_size + test_size);

    *train_data_x = cml_matrix_alloc(train_size, x->n);
    *train_data_y = cml_matrix_alloc(train_size, y->n);

    *val_data_x = cml_matrix_alloc(val_size, x->n);
    *val_data_y = cml_matrix_alloc(val_size, y->n);

    *test_data_x = cml_matrix_alloc(test_size, x->n);
    *test_data_y = cml_matrix_alloc(test_size, y->n);

    lgint indices[x->m];
    if (shuffle)
    {
        sufle_indices(indices, x->m);
    }
    else
    {
        for (lgint i = 0; i < x->m; i++)
            indices[i] = i;
    }

    for (lgint i = 0; i < x->m; i++)
    {
        if (i >= train_size + val_size)
        {
            for (lgint j = 0; j < x->n; j++)
            {
                (*test_data_x)->set(test_data_x, i - val_size - train_size, j, x->get(x, indices[i], j));
            }
            for (lgint j = 0; j < y->n; j++)
            {
                (*test_data_y)->set(test_data_y, i - val_size - train_size, j, y->get(y, indices[i], j));
            }
        }
        else if (i >= train_size)
        {
            for (lgint j = 0; j < x->n; j++)
            {
                (*val_data_x)->set(val_data_x, i - train_size, j, x->get(x, indices[i], j));
            }
            for (lgint j = 0; j < y->n; j++)
            {
                (*val_data_y)->set(val_data_y, i - train_size, j, y->get(y, indices[i], j));
            }
        }
        else
        {
            for (lgint j = 0; j < x->n; j++)
            {
                (*train_data_x)->set(train_data_x, i, j, x->get(x, indices[i], j));
            }
            for (lgint j = 0; j < y->n; j++)
            {
                (*train_data_y)->set(train_data_y, i, j, y->get(y, indices[i], j));
            }
        }
    }
}