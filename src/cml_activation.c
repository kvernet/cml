#include "cml_activation.h"

#include <math.h>
#include <stdlib.h>

#define LEAKY_RELU_COEF 0.01

const char *cml_activation_name(const cml_activation *const activation)
{
    switch (*activation)
    {
    case LINEAR:
        return "linear";
    case RELU:
        return "relu";
    case LEAKY_RELU:
        return "lrelu";
    case SIGMOID:
        return "sigmoid";
    case TANH:
        return "tanh";
    case SOFTMAX:
        return "softmax";
    default:
        return NULL;
    }
}

fdouble eval_linear(const fdouble x)
{
    return x;
}
fdouble eval_linear_grad(const fdouble)
{
    return 1.;
}

fdouble eval_relu(const fdouble x)
{
    return (x > 0) ? x : 0;
}
fdouble eval_relu_grad(const fdouble x)
{
    return (x > 0) ? 1 : 0;
}

fdouble eval_leaky_relu(const fdouble x)
{
    return (x > 0) ? x : LEAKY_RELU_COEF * x;
}
fdouble eval_leaky_relu_grad(const fdouble x)
{
    return (x > 0) ? 1 : LEAKY_RELU_COEF;
}

fdouble eval_sigmoid(const fdouble x)
{
    return 1. / (1. + exp(-x));
}
fdouble eval_sigmoid_grad(const fdouble x)
{
    return eval_sigmoid(x) * (1 - eval_sigmoid(x));
}

fdouble eval_tanh(const fdouble x)
{
    const fdouble e1 = exp(x);
    const fdouble e2 = exp(-x);

    return (e1 - e2) / (e1 + e2);
}
fdouble eval_tanh_grad(const fdouble x)
{
    const fdouble y = eval_tanh(x);
    return 1 - y * y;
}
