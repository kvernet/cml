#include "cml_optimizer.h"

#include <stdlib.h>

const char *cml_optimizer_name(cml_optimizer *const optimizer)
{
    switch (*optimizer)
    {
    case GRADIENT_DESCENT:
        return "Gradient Descent";
    case STOCHASTIC_GRADIENT_DESCENT:
        return "Stochastic Gradient Descent";
    case ADAGRAD:
        return "Adagrad";
    case RMSPROP:
        return "RMS Prop";
    case ADAM:
        return "Adaptive Moment";
    default:
        return NULL;
    }
}