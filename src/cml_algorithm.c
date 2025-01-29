#include "cml_algorithm.h"

#include <stdlib.h>

const char *cml_algorithm_name(cml_algorithm *const algorithm)
{
    switch (*algorithm)
    {
    case LINEAR_CLASSIF:
        return "Linear classification";
    case SVM:
        return "Support Vector Machine";
    case K_NEAREST:
        return "K-Nearest";
    case RANDOM_FOREST:
        return "Random Forest";
    case LINEAR_REG:
        return "Linear Regression";
    case LASSO_REG:
        return "Lasso Regression";
    case MULTIVARIATE_REG:
        return "Multivariate Regression";
    case K_MEANS:
        return "K-Means";
    case EXPECTATION_MAX:
        return "Expectation Max";
    case HCA:
        return "Hierarchical Cluster Analysis";
    default:
        return NULL;
    }
}