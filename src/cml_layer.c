#include "cml_layer.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

struct layer
{
    /* Public interface */
    cml_layer pub;

    /* Placeholder for data */
    cml_matrix *weight;
    cml_matrix *bias;
};

const char *cml_activation_name(const cml_activation *const activation)
{
    switch (*activation)
    {
    case LINEAR:
        return "linear";
    case RELU:
        return "relu";
    case SIGMOID:
        return "sigmoid";
    default:
        return NULL;
    }
}

static cml_matrix *layer_bias(cml_layer *const layer);
static void layer_compile(cml_layer *const layer, const lgint n_inputs);
static cml_matrix *layer_eval(cml_layer *const layer, cml_matrix *const x);
static void layer_free(cml_layer **layer);
static cml_matrix *layer_gradient(cml_layer *const layer, cml_matrix *const x);
static void layer_print(cml_layer *const layer);
static cml_matrix *layer_weight(cml_layer *const layer);

cml_layer *cml_layer_create(const lgint units, const cml_activation activation)
{
    struct layer *layer = (struct layer *)malloc(sizeof(*layer));
    *(lgint *)(&layer->pub.units) = units;
    *(cml_activation *)(&layer->pub.activation) = activation;

    layer->pub.bias = &layer_bias;
    layer->pub.compile = &layer_compile;
    layer->pub.eval = &layer_eval;
    layer->pub.free = &layer_free;
    layer->pub.gradient = &layer_gradient;
    layer->pub.print = &layer_print;
    layer->pub.weight = &layer_weight;

    layer->weight = NULL;
    layer->bias = NULL;

    return &layer->pub;
}

cml_matrix *layer_bias(cml_layer *const self)
{
    if (self == NULL)
        return NULL;
    struct layer *layer = (struct layer *)self;
    return layer->bias;
}

static fdouble random_sign_number(const int max)
{
    const fdouble u = (fdouble)rand() / RAND_MAX;
    int sign = (u > 0.5) ? 1 : -1;

    return sign * (rand() % max);
}

void layer_compile(cml_layer *const self, const lgint n_inputs)
{
    if (self == NULL)
        return;
    struct layer *layer = (struct layer *)self;
    layer->weight = cml_matrix_alloc(n_inputs, self->units);
    layer->bias = cml_matrix_alloc(self->units, 1);

    const lgint max = 10;

    for (lgint j = 0; j < layer->weight->n; j++)
    {
        for (lgint i = 0; i < layer->weight->m; i++)
        {
            layer->weight->set(&layer->weight, i, j, random_sign_number(max));
        }
        layer->bias->set(&layer->bias, j, 0, random_sign_number(max));
    }
}

static fdouble eval_linear(const fdouble x)
{
    return x;
}
static fdouble eval_linear_grad(const fdouble)
{
    return 1.;
}

static fdouble eval_relu(const fdouble x)
{
    return (x > 0) ? x : 0;
}
static fdouble eval_relu_grad(const fdouble x)
{
    return (x > 0) ? 1 : 0;
}

static fdouble eval_sigmoid(const fdouble x)
{
    return 1. / (1. + exp(-x));
}
static fdouble eval_sigmoid_grad(const fdouble x)
{
    return eval_sigmoid(x) * (1 - eval_sigmoid(x));
}

static fdouble eval_tanh(const fdouble x)
{
    const fdouble e1 = exp(x);
    const fdouble e2 = exp(-x);

    return (e1 - e2) / (e1 + e2);
}
static fdouble eval_tanh_grad(const fdouble x)
{
    const fdouble y = eval_tanh(x);
    return 1 - y * y;
}

static fdouble layer_compute(cml_layer *const layer, const fdouble x)
{
    switch (layer->activation)
    {
    case LINEAR:
        return eval_linear(x);
    case RELU:
        return eval_relu(x);
    case SIGMOID:
        return eval_sigmoid(x);
    case TANH:
        return eval_tanh(x);
    default:
        return DBL_MAX;
    }
}

static fdouble layer_compute_grad(cml_layer *const layer, const fdouble x)
{
    switch (layer->activation)
    {
    case LINEAR:
        return eval_linear_grad(x);
    case RELU:
        return eval_relu_grad(x);
    case SIGMOID:
        return eval_sigmoid_grad(x);
    case TANH:
        return eval_tanh_grad(x);
    default:
        return DBL_MAX;
    }
}

// compute activation(z) = activation(X*w + b)
cml_matrix *layer_eval(cml_layer *const self, cml_matrix *const x)
{
    if (self == NULL || x == NULL)
        return NULL;
    struct layer *layer = (struct layer *)self;

    if (x == NULL)
    {
        fprintf(stderr, "error (layer_eval): the matrix X is null in X*w.\n");
        return NULL;
    }
    if (layer->weight == NULL)
    {
        fprintf(stderr, "error (layer_eval): the matrix w is null in X*w.\n");
        return NULL;
    }
    if (x->n != layer->weight->m)
    {
        fprintf(stderr, "error (layer_eval): the matrices (%ld, %ld) and (%ld, %ld) are not product compatible in X*w.\n", x->m, x->n, layer->weight->m, layer->weight->n);
        return NULL;
    }
    cml_matrix *z = cml_matrix_alloc(x->m, layer->weight->n);
    for (lgint i = 0; i < z->m; i++)
    {
        for (lgint j = 0; j < z->n; j++)
        {
            fdouble s = 0.;
            for (lgint k = 0; k < x->n; k++)
            {
                s += x->get(x, i, k) * layer->weight->get(layer->weight, k, j);
            }
            const fdouble x = s + layer->bias->get(layer->bias, j, 0);
            z->set(&z, i, j, layer_compute(self, x));
        }
    }
    return z;
}

void layer_free(cml_layer **self)
{
    if (*self == NULL)
        return;
    struct layer *layer = (struct layer *)(*self);
    if (layer->weight != NULL)
        layer->weight->free(&layer->weight);
    if (layer->bias != NULL)
        layer->bias->free(&layer->bias);
    free(layer);
    *self = NULL;
}

// compute activation_prime(z) = activation_prime(X*w + b)
cml_matrix *layer_gradient(cml_layer *const self, cml_matrix *const x)
{
    if (self == NULL || x == NULL)
        return NULL;
    struct layer *layer = (struct layer *)self;

    if (x == NULL)
    {
        fprintf(stderr, "error (layer_gradient): the matrix X is null in X*w.\n");
        return NULL;
    }
    if (layer->weight == NULL)
    {
        fprintf(stderr, "error (layer_gradient): the matrix w is null in X*w.\n");
        return NULL;
    }
    if (x->n != layer->weight->m)
    {
        fprintf(stderr, "error (layer_gradient): the matrices (%ld, %ld) and (%ld, %ld) are not product compatible in X*w.\n", x->m, x->n, layer->weight->m, layer->weight->n);
        return NULL;
    }
    cml_matrix *z = cml_matrix_alloc(x->m, layer->weight->n);
    for (lgint i = 0; i < z->m; i++)
    {
        for (lgint j = 0; j < z->n; j++)
        {
            fdouble s = 0.;
            for (lgint k = 0; k < x->n; k++)
            {
                s += x->get(x, i, k) * layer->weight->get(layer->weight, k, j);
            }
            const fdouble x = s + layer->bias->get(layer->bias, j, 0);
            z->set(&z, i, j, layer_compute_grad(self, x));
        }
    }
    return z;
}

void layer_print(cml_layer *const self)
{
    if (self == NULL)
        return;
    printf("=== Layer ===\n");
    printf("units: %ld\n", self->units);
    printf("activation: %s\n", cml_activation_name(&self->activation));
    struct layer *layer = (struct layer *)self;
    if (layer->weight != NULL)
        layer->weight->print(layer->weight);
    else
        printf("weight=null\n");
    if (layer->bias != NULL)
        layer->bias->print(layer->bias);
    else
        printf("bias=null\n");
}

cml_matrix *layer_weight(cml_layer *const self)
{
    if (self == NULL)
        return NULL;
    struct layer *layer = (struct layer *)self;
    return layer->weight;
}