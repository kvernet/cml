#include "cml_sequential.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define CML_EPSILON 1E-06

struct sequential
{
    /* Public interface */
    cml_sequential pub;

    /* Placeholder for data */
    bool is_compiled;
};

static void sequential_compile(cml_sequential *const model, cml_prng *const prng);
static void sequential_fit(cml_sequential *const model, cml_matrix *const x, cml_matrix *const y, const fdouble alpha, const lgint epochs);
static void sequential_free(cml_sequential **model);
static cml_matrix *sequential_predict(cml_sequential *const model, cml_matrix *const x);
static void sequential_summary(cml_sequential *const model);

cml_sequential *cml_sequential_create(cml_layer *layers[], const lgint n_layers, const lgint n_inputs)
{
    struct sequential *model = NULL;
    const size_t size = sizeof(*model);
    model = (struct sequential *)malloc(size);
    model->pub.layers = layers;
    *(lgint *)(&model->pub.n_layers) = n_layers;
    *(lgint *)(&model->pub.n_inputs) = n_inputs;

    model->pub.compile = &sequential_compile;
    model->pub.fit = &sequential_fit;
    model->pub.free = &sequential_free;
    model->pub.predict = &sequential_predict;
    model->pub.summary = &sequential_summary;

    model->is_compiled = false;

    return &model->pub;
}

void sequential_compile(cml_sequential *const model, cml_prng *const prng)
{
    if (model == NULL)
        return;
    for (lgint i = 0; i < model->n_layers; i++)
    {
        cml_layer *layer = model->layers[i];
        if (i == 0)
            layer->compile(layer, model->n_inputs, prng);
        else
            layer->compile(layer, model->layers[i - 1]->weight(model->layers[i - 1])->n, prng);
    }
    struct sequential *sequential = (struct sequential *)model;
    sequential->is_compiled = true;
}

static void sequential_forward(cml_sequential *const model, cml_matrix *const x, cml_matrix **inputs)
{
    for (lgint n = 0; n < model->n_layers; n++)
    {
        cml_layer *layer = model->layers[n];
        if (n == 0)
            inputs[n] = layer->eval(layer, x);
        else
            inputs[n] = layer->eval(layer, inputs[n - 1]);
    }
}

static cml_matrix *gradient_bias(cml_matrix *const err, const lgint m)
{
    cml_matrix *gradB = cml_matrix_alloc(err->n, 1);
    for (lgint i = 0; i < gradB->m; i++)
    {
        fdouble s = 0.;
        for (lgint j = 0; j < err->m; j++)
        {
            s += err->get(err, j, i);
        }
        gradB->set(&gradB, i, 0, s / m);
    }
    return gradB;
}

static cml_matrix *gradient_weight(cml_matrix *const input, cml_matrix *const err, const lgint m)
{
    cml_matrix *input_transpose = NULL;
    input->transpose(input, &input_transpose);
    cml_matrix *gradW = cml_matrix_alloc(input_transpose->m, err->n);
    for (lgint i = 0; i < gradW->m; i++)
    {
        for (lgint j = 0; j < gradW->n; j++)
        {
            fdouble s = 0.;
            for (lgint k = 0; k < input_transpose->n; k++)
            {
                s += input_transpose->get(input_transpose, i, k) * err->get(err, k, j);
            }
            gradW->set(&gradW, i, j, s / m);
        }
    }
    input_transpose->free(&input_transpose);
    return gradW;
}

static void update_weight_bias(cml_matrix **weight, cml_matrix **bias, cml_matrix *const gradW, cml_matrix *const gradB, const fdouble alpha)
{
    for (lgint j = 0; j < (*weight)->n; j++)
    {
        for (lgint i = 0; i < (*weight)->m; i++)
        {
            (*weight)->set(
                weight,
                i, j,
                (*weight)->get(*weight, i, j) - alpha * gradW->get(gradW, i, j));
        }
        (*bias)->set(
            bias,
            j, 0,
            (*bias)->get(*bias, j, 0) - alpha * gradB->get(gradB, j, 0));
    }
}

static void sequential_backward(cml_sequential *const model, cml_matrix *const x, cml_matrix **inputs, cml_matrix *const y, const fdouble alpha)
{
    cml_matrix *err = cml_matrix_dif(inputs[model->n_layers - 1], y);
    const lgint m = x->m;

    for (long n = model->n_layers - 2; n >= 0; n--)
    {
        cml_matrix *input = inputs[n];

        cml_layer *layer = model->layers[n + 1];
        cml_matrix *W = layer->weight(layer);
        cml_matrix *b = layer->bias(layer);

        cml_matrix *gradW = gradient_weight(input, err, m);
        cml_matrix *gradB = gradient_bias(err, m);

        update_weight_bias(&W, &b, gradW, gradB, alpha);

        gradW->free(&gradW);
        gradB->free(&gradB);

        cml_matrix *WT = NULL;
        W->transpose(W, &WT);
        cml_matrix *prod = cml_matrix_prod(err, WT);
        err->free(&err);
        WT->free(&WT);
        layer = model->layers[n];
        cml_matrix *zp = NULL;
        if (n > 0)
            zp = layer->gradient(layer, inputs[n - 1]);
        else
            zp = layer->gradient(layer, x);

        err = prod->hadamard(prod, zp);
        zp->free(&zp);
        prod->free(&prod);
    }

    err->free(&err);
}

static fdouble sequential_mse(cml_sequential *const model, cml_matrix *const x, cml_matrix *const y)
{
    if (model == NULL)
        return DBL_MAX;
    cml_matrix *yhat = model->predict(model, x);
    if (yhat == NULL)
        return DBL_MAX;
    cml_matrix *dif = cml_matrix_dif(yhat, y);
    yhat->free(&yhat);
    cml_matrix *dif_t = NULL;
    dif->transpose(dif, &dif_t);
    cml_matrix *tmp = cml_matrix_prod(dif_t, dif);
    dif->free(&dif);
    dif_t->free(&dif_t);
    fdouble mse = 0.;
    for (lgint i = 0; i < tmp->m; i++)
    {
        for (lgint j = 0; j < tmp->n; j++)
        {
            mse += tmp->get(tmp, i, j);
        }
    }
    mse /= (2 * y->m);
    tmp->free(&tmp);
    return mse;
}

void sequential_fit(cml_sequential *const model, cml_matrix *const x, cml_matrix *const y, const fdouble alpha, const lgint epochs)
{
    if (model == NULL || x == NULL || y == NULL)
        return;
    struct sequential *sequential = (struct sequential *)model;
    if (!sequential->is_compiled)
    {
        fprintf(stderr, "Error (sequential_fit): the model should be compiled first.\n");
        return;
    }
    if (x == NULL)
    {
        fprintf(stderr, "Error (sequential_fit): the input [x] is null\n");
        return;
    }
    if (y == NULL)
    {
        fprintf(stderr, "Error (sequential_fit): the output [y] is null\n");
        return;
    }

    for (lgint e = 0; e < epochs; e++)
    {
        // feed forward
        cml_matrix *inputs[model->n_layers];
        sequential_forward(model, x, inputs);

        // backward
        sequential_backward(model, x, inputs, y, alpha);

        for (lgint n = 0; n < model->n_layers; n++)
        {
            inputs[n]->free(&inputs[n]);
        }

        const fdouble mse = sequential_mse(model, x, y);
        printf("epoch\t%ld/%ld\tmse %lg\n", e + 1, epochs, mse);
    }
}

void sequential_free(cml_sequential **model)
{
    if (*model == NULL)
        return;
    for (lgint i = 0; i < (*model)->n_layers; i++)
    {
        cml_layer *layer = (*model)->layers[i];
        if (layer != NULL)
            layer->free(&layer);
    }
    free(*model);
    *model = NULL;
}

cml_matrix *sequential_predict(cml_sequential *const model, cml_matrix *const x)
{
    if (model == NULL || x == NULL)
        return NULL;
    struct sequential *sequential = (struct sequential *)model;
    if (!sequential->is_compiled)
    {
        fprintf(stderr, "Error (sequential_predict): the model should be compiled first.\n");
        return NULL;
    }
    cml_matrix *mts[model->n_layers];
    cml_matrix *a = x;

    for (lgint i = 0; i < model->n_layers; i++)
    {
        cml_layer *layer = model->layers[i];
        mts[i] = layer->eval(layer, a);
        a = mts[i];
        if (i > 0)
            mts[i - 1]->free(&mts[i - 1]);
    }

    return mts[model->n_layers - 1];
}

void sequential_summary(cml_sequential *const model)
{
    if (model == NULL)
        return;
    struct sequential *sequential = (struct sequential *)model;
    if (!sequential->is_compiled)
    {
        fprintf(stderr, "Error (sequential_summary): the model should be compiled first.\n");
        return;
    }
    printf("Model summary:\n");
    printf("Layer\t\tActivation\t\tUnits\t\tVariables\n");
    printf("-------------------------------------------------------------------\n");
    for (lgint i = 0; i < model->n_layers; i++)
    {
        cml_layer *layer = model->layers[i];
        cml_matrix *weight = layer->weight(layer);
        // cml_matrix *bias = layer->bias(layer);
        const lgint nvar = weight->m * weight->n; // + bias->m * bias->n;
        printf("%ld\t\t%s\t\t\t%ld\t\t%ld\n", i + 1, cml_activation_name(&layer->activation), layer->units, nvar);
    }
    printf("-------------------------------------------------------------------\n");
}