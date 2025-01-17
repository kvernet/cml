#ifndef cml_sequential_h
#define cml_sequential_h

#include "cml_layer.h"
#include "cml_matrix.h"
#include "cml_prng.h"

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct cml_sequential cml_sequential;

    typedef void cml_sequential_compile(cml_sequential *const model, cml_prng *const prng);

    typedef void cml_sequential_fit(cml_sequential *const model, cml_matrix *const x, cml_matrix *const y, const fdouble alpha, const lgint epochs);

    typedef void cml_sequential_free(cml_sequential **model);

    typedef cml_matrix *cml_sequential_predict(cml_sequential *const model, cml_matrix *const x);

    typedef void cml_sequential_summary(cml_sequential *const model);

    struct cml_sequential
    {
        cml_layer **layers;
        const lgint n_layers;
        const lgint n_inputs;

        cml_sequential_compile *compile;
        cml_sequential_fit *fit;
        cml_sequential_free *free;
        cml_sequential_predict *predict;
        cml_sequential_summary *summary;
    };

    cml_sequential *cml_sequential_create(cml_layer *layers[], const lgint n_layers, const lgint n_inputs);

#ifdef __cplusplus
}
#endif

#endif