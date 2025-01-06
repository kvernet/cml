#ifndef cml_matrix_h
#define cml_matrix_h

#include <stddef.h>

#ifdef __cplusplus
extern "C"
{
#endif

    typedef double fdouble;

    typedef size_t lgint;

    typedef struct cml_matrix cml_matrix;

    typedef cml_matrix *cml_matrix_copy(cml_matrix *const a);

    typedef fdouble cml_matrix_det(cml_matrix *const a);

    typedef void cml_matrix_free(cml_matrix **a);

    typedef fdouble cml_matrix_get(cml_matrix *const a, const lgint i, const lgint j);

    typedef cml_matrix *cml_matrix_inv(cml_matrix *const a);

    typedef void cml_matrix_lu(cml_matrix *const a, cml_matrix **p, cml_matrix **l, cml_matrix **u);

    typedef void cml_matrix_print(cml_matrix *const a);

    typedef void cml_matrix_set(cml_matrix **a, const lgint i, const lgint j, const fdouble value);

    typedef fdouble cml_matrix_trace(cml_matrix *const a);

    typedef void cml_matrix_transpose(cml_matrix *const a, cml_matrix **at);

    struct cml_matrix
    {
        const lgint m;
        const lgint n;

        cml_matrix_copy *copy;
        cml_matrix_det *det;
        cml_matrix_free *free;
        cml_matrix_get *get;
        cml_matrix_inv *inv;
        cml_matrix_lu *lu;
        cml_matrix_print *print;
        cml_matrix_set *set;
        cml_matrix_trace *trace;
        cml_matrix_transpose *transpose;
    };

    cml_matrix *cml_matrix_alloc(const lgint m, const lgint n);

    cml_matrix *cml_matrix_dif(cml_matrix *const a, cml_matrix *const b);

    cml_matrix *cml_matrix_eye(const lgint n);

    cml_matrix *cml_matrix_solve(cml_matrix *const a, cml_matrix *const b);

    cml_matrix *cml_matrix_sum(cml_matrix *const a, cml_matrix *const b);

    cml_matrix *cml_matrix_prod(cml_matrix *const a, cml_matrix *const b);

    cml_matrix *cml_matrix_zeros(const lgint m, const lgint n);

#ifdef __cplusplus
}
#endif

#endif