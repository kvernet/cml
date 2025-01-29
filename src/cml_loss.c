#include "cml_loss.h"

#include <stdlib.h>

const char *cml_loss_name(cml_loss *const loss)
{
    switch (*loss)
    {
    case SQUARED_ERROR_LOSS:
        return "Squared Loss";
    case BINARY_CROSS_ENTROPY:
        return "Binary Cross Entropy";
    case MULTI_CLASS_CROSS_ENTROPY:
        return "Multi Class Cross Entropy";
    default:
        return NULL;
    }
}