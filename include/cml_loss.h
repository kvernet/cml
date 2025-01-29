#ifndef cml_loss_h
#define cml_loss_h

#ifdef __cplusplus
extern "C"
{
#endif

    typedef enum cml_loss
    {
        SQUARED_ERROR_LOSS = 0,
        BINARY_CROSS_ENTROPY,
        MULTI_CLASS_CROSS_ENTROPY
    } cml_loss;

    const char *cml_loss_name(cml_loss *const loss);

#ifdef __cplusplus
}
#endif

#endif