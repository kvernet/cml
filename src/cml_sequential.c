#include "cml_sequential.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define CML_EPSILON 1E-06

static fdouble matrix_sum(cml_matrix *const a)
{
    if (a == NULL)
        return 0;

    fdouble s = 0;
    for (lgint i = 0; i < a->m; i++)
    {
        for (lgint j = 0; j < a->n; j++)
        {
            s += a->get(a, i, j);
        }
    }
    return s;
}

void cml_class_metrics(cml_matrix **prec, cml_matrix **accur, cml_matrix **f1_score, cml_matrix *const yhat, cml_matrix *const y)
{
    cml_matrix *conf = cml_matrix_confusion(yhat, y);
    if (conf == NULL)
        return;

    fdouble total = matrix_sum(conf);

    *prec = cml_matrix_alloc(conf->m, conf->n);
    *accur = cml_matrix_alloc(conf->m, conf->n);
    *f1_score = cml_matrix_alloc(conf->m, conf->n);

    for (lgint i = 0; i < conf->m; i++)
    {
        const fdouble tp = conf->get(conf, i, i);
        for (lgint j = 0; j < conf->n; j++)
        {
            fdouble fn = 0, fp = 0;
            for (lgint k = 0; k < conf->n; k++)
            {
                fn += conf->get(conf, i, k);
                fp += conf->get(conf, k, i);
            }
            fn -= tp;
            fp -= tp;
            fdouble tn = total - (tp + fn + fp);
            fdouble p = 0;
            if (tp + fp != 0)
                p = tp / (tp + fp);

            fdouble a = 0;
            if (total != 0)
                a = (tp + tn) / total;

            fdouble rec = 0;
            if (tp + fn != 0)
                rec = tp / (tp + fn);

            fdouble score = 0;
            if (p + rec != 0)
                score = 2 * p * rec / (p + rec);

            (*prec)->set(prec, i, j, p);
            (*accur)->set(accur, i, j, a);
            (*f1_score)->set(f1_score, i, j, score);
        }
    }
    conf->free(&conf);
}

static fdouble cml_huber_loss(const fdouble yhat_i, const fdouble y_i, const fdouble threshold)
{
    const fdouble delta = yhat_i - y_i;
    return (fabs(delta) <= threshold) ? 0.5 * delta * delta : threshold * (fabs(delta) - 0.5 * threshold);
}

// comparison function for qsort (necessary for sorting doubles)
static int cml_compare(const void *a, const void *b)
{
    double val_a = *((double *)a);
    double val_b = *((double *)b);
    return (val_a > val_b) - (val_a < val_b); // Return -1, 0, or 1
}

static fdouble cml_median_abs_error(cml_matrix *const yhat, cml_matrix *const y)
{
    const lgint size = yhat->m * yhat->n;
    fdouble med[size];
    for (lgint i = 0; i < yhat->m; i++)
    {
        for (lgint j = 0; j < yhat->n; j++)
        {
            med[i * yhat->n + j] = fabs(yhat->get(yhat, i, j) - y->get(y, i, j));
        }
    }
    // sorting the array using qsort
    qsort(med, size, sizeof(*med), cml_compare);

    return (size % 2 == 0) ? 0.5 * (med[size / 2 - 1] + med[size / 2]) : med[size / 2];
}

void cml_reg_metrics(fdouble *mae, fdouble *mse, fdouble *rmse,
                     fdouble *rsquared, fdouble *arsquared, fdouble *mape,
                     fdouble *smape, fdouble *hloss, fdouble *evars, fdouble *medae,
                     cml_matrix *const yhat, cml_matrix *const y,
                     const lgint k, const fdouble threshold)
{
    *mae = *mse = *rmse = *rsquared = *arsquared = 0;
    *mape = *smape = *hloss = *evars = *medae = 0;
    if (yhat->m != y->m || yhat->n != y->n)
    {
        fprintf(stderr, "Error (cml_reg_metrics): the matrices should be of same dimension.\n");
        return;
    }
    const lgint N = yhat->m;
    // compute the mean of the actual(true) values
    fdouble y_mean = 0., y_mean2 = 0., y_delta_mean = 0., y_delta_mean2 = 0.;
    for (lgint i = 0; i < yhat->m; i++)
    {
        for (lgint j = 0; j < yhat->n; j++)
        {
            const fdouble yhat_ij = yhat->get(yhat, i, j);
            const fdouble y_ij = y->get(y, i, j);
            const fdouble delta = yhat_ij - y_ij;
            y_mean += y_ij;
            y_mean2 += y_ij * y_ij;
            y_delta_mean += delta;
            y_delta_mean2 += delta * delta;
        }
    }
    y_mean /= N;
    y_mean2 /= N;
    y_delta_mean /= N;
    y_delta_mean2 /= N;
    const fdouble y_var = y_mean2 - y_mean * y_mean;
    const fdouble mean_delta_var = y_delta_mean2 - y_delta_mean * y_delta_mean;

    fdouble sum_delta_mean = 0;
    for (lgint i = 0; i < yhat->m; i++)
    {
        for (lgint j = 0; j < yhat->n; j++)
        {
            const fdouble yhat_ij = yhat->get(yhat, i, j);
            const fdouble y_ij = y->get(y, i, j);
            const fdouble delta = yhat_ij - y_ij;
            *mae += fabs(delta);
            *mse += delta * delta;
            sum_delta_mean += (y_ij - y_mean) * (y_ij - y_mean);
            *mape += fabs(delta / y_ij);
            *smape += 2 * fabs(delta) / (fabs(y_ij) + fabs(yhat_ij));
            *hloss += cml_huber_loss(yhat_ij, y_ij, threshold);
        }
    }

    *mae /= N;
    *mse /= N;
    *rmse = sqrt(*mse);
    sum_delta_mean /= N;
    *rsquared = 1 - *mse / sum_delta_mean;
    *arsquared = 1 - (1 - *rsquared) * (N - 1) / (N - k - 1);
    *mape = *mape * 100 / N;
    *smape = *smape * 100 / N;
    *evars = 1 - mean_delta_var / y_var;
    *medae = cml_median_abs_error(yhat, y);
}

struct sequential
{
    /* Public interface */
    cml_sequential pub;

    /* Placeholder for data */
    bool is_compiled;
};

static void sequential_compile(cml_sequential *const model, cml_prng *const prng);
static void sequential_fit(cml_sequential *const model, cml_matrix *const x, cml_matrix *const y, fdouble (*learning_rate)(fdouble alpha), const fdouble alpha, const lgint epochs);
static void sequential_free(cml_sequential **model);
static cml_matrix *sequential_predict(cml_sequential *const model, cml_matrix *const x);
static void sequential_summary(cml_sequential *const model);

cml_sequential *cml_sequential_create(cml_layer *layers[], const lgint n_layers, const lgint n_inputs, const cml_loss loss)
{
    struct sequential *model = NULL;
    const size_t size = sizeof(*model);
    model = (struct sequential *)malloc(size);
    model->pub.layers = layers;
    *(lgint *)(&model->pub.n_layers) = n_layers;
    *(lgint *)(&model->pub.n_inputs) = n_inputs;
    *(cml_loss *)(&model->pub.loss) = loss;

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

static fdouble sequential_softmax_entropy(cml_sequential *const model, cml_matrix *const x, cml_matrix *const y)
{
    if (model == NULL)
        return DBL_MAX;
    cml_matrix *prob = model->predict(model, x);
    if (prob == NULL)
        return DBL_MAX;
    cml_matrix *lprob = cml_matrix_alloc(prob->m, prob->n);
    for (lgint i = 0; i < lprob->m; i++)
    {
        for (lgint j = 0; j < lprob->n; j++)
        {
            lprob->set(&lprob, i, j, log(prob->get(prob, i, j)));
        }
    }
    prob->free(&prob);
    cml_matrix *ytranspose = NULL;
    y->transpose(y, &ytranspose);
    cml_matrix *prod = cml_matrix_prod(ytranspose, lprob);
    ytranspose->free(&ytranspose);
    lprob->free(&lprob);
    const fdouble trace = prod->trace(prod);
    prod->free(&prod);
    return -trace / x->m;
}

void sequential_fit(cml_sequential *const model, cml_matrix *const x, cml_matrix *const y, fdouble (*learning_rate)(fdouble alpha), const fdouble alpha, const lgint epochs)
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

    fdouble rate = alpha;
    for (lgint e = 0; e < epochs; e++)
    {
        // feed forward
        cml_matrix *inputs[model->n_layers];
        sequential_forward(model, x, inputs);

        // backward
        rate = learning_rate(rate);
        sequential_backward(model, x, inputs, y, rate);

        for (lgint n = 0; n < model->n_layers; n++)
        {
            inputs[n]->free(&inputs[n]);
        }

        fdouble loss = 0;
        if (model->loss == MULTI_CLASS_CROSS_ENTROPY)
        {
            loss = sequential_softmax_entropy(model, x, y);
        }
        else
        {
            loss = sequential_mse(model, x, y);
        }
        printf("epoch\t%ld/%ld\tlearning rate\t%5.7E\tloss %5.7E\n", e + 1, epochs, rate, loss);
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