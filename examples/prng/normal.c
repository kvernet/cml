#include "cml_prng.h"

#include <stdio.h>
#include <stdlib.h>

int main(void)
{
    cml_prng *prng = cml_prng_init(NULL);

    printf("seed: %d\n", prng->seed);

    for (int i = 0; i < 10; i++)
        printf("random_%d = %lg\n", i, prng->normal(prng, 0, 1));

    prng->free(&prng);

    return EXIT_SUCCESS;
}