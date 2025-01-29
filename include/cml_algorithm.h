#ifndef cml_algorithm_h
#define cml_algorithm_h

#ifdef __cplusplus
extern "C"
{
#endif

    typedef enum cml_algorithm
    {
        /* classification */
        LINEAR_CLASSIF = 0,
        SVM,
        K_NEAREST,
        RANDOM_FOREST,
        /* regression */
        LINEAR_REG,
        LASSO_REG,
        MULTIVARIATE_REG,
        /* clustering */
        K_MEANS,
        EXPECTATION_MAX,
        HCA /* Hierarchical Cluster Analysis */
    } cml_algorithm;

    const char *cml_algorithm_name(cml_algorithm *const algorithm);

#ifdef __cplusplus
}
#endif

#endif