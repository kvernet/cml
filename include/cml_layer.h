#ifndef cml_layer_h
#define cml_layer_h

#include "cml_matrix.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef enum cml_activation
    {
        LINEAR = 0,
        RELU = 1,
        SIGMOID = 2,
        TANH = 3
    } cml_activation;

    const char *cml_activation_name(const cml_activation *const activation);

    typedef struct cml_layer cml_layer;

    typedef cml_matrix *cml_layer_bias(cml_layer *const layer);

    typedef void cml_layer_compile(cml_layer *const layer, const lgint n_inputs);

    typedef cml_matrix *cml_layer_eval(cml_layer *const layer, cml_matrix *const x);

    typedef void cml_layer_free(cml_layer **layer);

    typedef cml_matrix *cml_layer_gradient(cml_layer *const layer, cml_matrix *const x);

    typedef void cml_layer_print(cml_layer *const layer);

    typedef cml_matrix *cml_layer_weight(cml_layer *const layer);

    struct cml_layer
    {
        const lgint units;
        const cml_activation activation;

        cml_layer_bias *bias;
        cml_layer_compile *compile;
        cml_layer_eval *eval;
        cml_layer_free *free;
        cml_layer_gradient *gradient;
        cml_layer_print *print;
        cml_layer_weight *weight;
    };

    cml_layer *cml_layer_create(const lgint units, const cml_activation activation);

#ifdef __cplusplus
}
#endif

#endif