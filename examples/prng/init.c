#include "cml_prng.h"

#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    unsigned int seed = 1234;
    cml_prng *prng = cml_prng_init(&seed);

    printf("seed: %d\n", prng->seed);

    prng->free(&prng);

    return EXIT_SUCCESS;
}