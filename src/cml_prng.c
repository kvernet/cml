#include "cml_prng.h"

#include <fcntl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.1415
#endif

static unsigned int cml_get_seed();

static void prng_free(cml_prng **prng);

static fdouble prng_normal(cml_prng *const prng, const fdouble mu, const fdouble sigma);

static fdouble prng_uniform01(cml_prng *const prng);

static fdouble prng_uniform(cml_prng *const prng, const fdouble a, const fdouble b);

cml_prng *cml_prng_init(unsigned int *seed)
{
    cml_prng *prng = (cml_prng *)malloc(sizeof(*prng));

    if (seed == NULL)
    {
        *(unsigned int *)(&prng->seed) = cml_get_seed();
    }
    else
    {
        *(unsigned int *)(&prng->seed) = *seed;
    }

    prng->free = &prng_free;
    prng->normal = &prng_normal;
    prng->uniform01 = &prng_uniform01;
    prng->uniform = &prng_uniform;

    // set seed
    srand(prng->seed);

    return prng;
}

unsigned int cml_get_seed()
{
    int fd = open("/dev/urandom", O_RDONLY);
    if (fd == -1)
    {
        return time(0);
    }

    unsigned int seed;
    ssize_t result = read(fd, &seed, sizeof(seed));
    if (result != sizeof(seed))
    {
        close(fd);
        return time(0);
    }
    close(fd);

    return seed;
}

void prng_free(cml_prng **prng)
{
    if (*prng == NULL)
        return;
    free(*prng);
    *prng = NULL;
}

fdouble prng_normal(cml_prng *const prng, const fdouble mu, const fdouble sigma)
{
    if (sigma == 0)
    {
        return mu;
    }

    const fdouble u = prng->uniform01(prng);
    const fdouble v = prng->uniform01(prng);

    const double two_pi = 2 * M_PI;
    const double mag = sqrt(-2 * log(u));

    return sigma * mag * cos(two_pi * v) + mu;
}

fdouble prng_uniform01(cml_prng *)
{
    return (fdouble)rand() / (fdouble)RAND_MAX;
}

fdouble prng_uniform(cml_prng *prng, const fdouble a, const fdouble b)
{
    fdouble u = prng->uniform01(prng);
    return a + u * (b - a);
}