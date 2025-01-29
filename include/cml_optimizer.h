#ifndef cml_optimiser_h
#define cml_optimiser_h

#ifdef __cplusplus
extern "C"
{
#endif

    typedef enum cml_optimizer
    {
        GRADIENT_DESCENT = 0,
        STOCHASTIC_GRADIENT_DESCENT,
        ADAGRAD,
        RMSPROP,
        ADAM
    } cml_optimizer;

    const char *cml_optimizer_name(cml_optimizer *const optimizer);

#ifdef __cplusplus
}
#endif

#endif