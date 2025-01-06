#include "cml_sequential.h"

#include <float.h>
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

static void sequential_compile(cml_sequential *const model);
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

void sequential_compile(cml_sequential *const model)
{
    if (model == NULL)
        return;
    for (lgint i = 0; i < model->n_layers; i++)
    {
        cml_layer *layer = model->layers[i];
        if (i == 0)
            layer->compile(layer, model->n_inputs);
        else
            layer->compile(layer, model->layers[i - 1]->weight(model->layers[i - 1])->n);
    }
    struct sequential *sequential = (struct sequential *)model;
    sequential->is_compiled = true;
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

void sequential_update_weight(cml_sequential *const model, cml_matrix *const x, cml_matrix *const y, const fdouble alpha)
{
    cml_matrix *tmp_weight[model->n_layers];
    cml_matrix *tmp_bias[model->n_layers];

    for (lgint n = 0; n < model->n_layers; n++)
    {
        cml_layer *layer = model->layers[n];
        cml_matrix *weight = layer->weight(layer);
        cml_matrix *bias = layer->bias(layer);

        // gradient w.r.t weight
        tmp_weight[n] = weight->copy(weight);
        for (lgint i = 0; i < weight->m; i++)
        {
            for (lgint j = 0; j < weight->n; j++)
            {
                const fdouble wij = weight->get(weight, i, j);
                weight->set(&weight, i, j, wij - CML_EPSILON);
                fdouble y1 = sequential_mse(model, x, y);
                weight->set(&weight, i, j, wij + CML_EPSILON);
                fdouble y2 = sequential_mse(model, x, y);
                const fdouble grad = 0.5 * (y2 - y1) / CML_EPSILON;
                tmp_weight[n]->set(&tmp_weight[n], i, j, grad);
                // reset weight for future computation
                weight->set(&weight, i, j, wij);
            }
        }
        // gradient w.r.t bias
        tmp_bias[n] = bias->copy(bias);
        for (lgint i = 0; i < bias->m; i++)
        {
            for (lgint j = 0; j < bias->n; j++)
            {
                const fdouble bij = bias->get(bias, i, j);
                bias->set(&bias, i, j, bij - CML_EPSILON);
                fdouble y1 = sequential_mse(model, x, y);
                bias->set(&bias, i, j, bij + CML_EPSILON);
                fdouble y2 = sequential_mse(model, x, y);
                tmp_bias[n]->set(&tmp_bias[n], i, j, 0.5 * (y2 - y1) / CML_EPSILON);
                // reset bias for future computation
                bias->set(&bias, i, j, bij);
            }
        }
    }

    // simulaneously update the weight and bias
    for (lgint n = 0; n < model->n_layers; n++)
    {
        cml_layer *layer = model->layers[n];
        cml_matrix *weight = layer->weight(layer);
        cml_matrix *bias = layer->bias(layer);

        for (lgint i = 0; i < tmp_weight[n]->m; i++)
        {
            for (lgint j = 0; j < tmp_weight[n]->n; j++)
            {
                weight->set(&weight, i, j, weight->get(weight, i, j) - alpha * tmp_weight[n]->get(tmp_weight[n], i, j));
            }
        }

        for (lgint i = 0; i < tmp_bias[n]->m; i++)
        {
            for (lgint j = 0; j < tmp_bias[n]->n; j++)
            {
                bias->set(&bias, i, j, bias->get(bias, i, j) - alpha * tmp_bias[n]->get(tmp_bias[n], i, j));
            }
        }

        tmp_weight[n]->free(&tmp_weight[n]);
        tmp_bias[n]->free(&tmp_bias[n]);
    }
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
    for (lgint e = 0; e < epochs; e++)
    {
        sequential_update_weight(model, x, y, alpha);
        fdouble mse = sequential_mse(model, x, y);
        fprintf(stderr, "epoch\t%ld\t\tmse\t\t%lg\n", e + 1, mse);
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