#ifndef cml_prng_h
#define cml_prng_h

#include "cml_matrix.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef struct cml_prng cml_prng;

    typedef void cml_prng_free(cml_prng **prng);

    typedef fdouble cml_prng_normal(cml_prng *const prng, const fdouble mu, const fdouble sigma);

    typedef fdouble cml_prng_uniform01(cml_prng *const prng);

    typedef fdouble cml_prng_uniform(cml_prng *const prng, const fdouble a, const fdouble b);

    struct cml_prng
    {
        const unsigned int seed;

        cml_prng_free *free;
        cml_prng_normal *normal;
        cml_prng_uniform01 *uniform01;
        cml_prng_uniform *uniform;
    };

    cml_prng *cml_prng_init(unsigned int *seed);

#ifdef __cplusplus
}
#endif

#endif