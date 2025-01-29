#ifndef cml_activation_h
#define cml_activation_h

#include "cml_matrix.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef enum cml_activation
    {
        LINEAR = 0,
        RELU,
        LEAKY_RELU,
        SIGMOID,
        TANH,
        SOFTMAX
    } cml_activation;

    const char *cml_activation_name(const cml_activation *const activation);

    fdouble eval_linear(const fdouble x);
    fdouble eval_linear_grad(const fdouble);

    fdouble eval_relu(const fdouble x);
    fdouble eval_relu_grad(const fdouble x);

    fdouble eval_leaky_relu(const fdouble x);
    fdouble eval_leaky_relu_grad(const fdouble x);

    fdouble eval_sigmoid(const fdouble x);
    fdouble eval_sigmoid_grad(const fdouble x);

    fdouble eval_tanh(const fdouble x);
    fdouble eval_tanh_grad(const fdouble x);

#ifdef __cplusplus
}
#endif

#endif