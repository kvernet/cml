#include "cml_matrix.h"

#include <float.h>
#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>

#define CML_MATRIX_TOLERANCE 1E-09

struct matrix
{
    /* Public interface */
    cml_matrix pub;

    /* Placeholder for data */
    fdouble data[];
};

static cml_matrix *matrix_copy(cml_matrix *const a);
static fdouble matrix_det(cml_matrix *const a);
static void matrix_free(cml_matrix **a);
static fdouble matrix_get(cml_matrix *const a, const lgint i, const lgint j);
static cml_matrix *matrix_hadamard(cml_matrix *const a, cml_matrix *const b);
static cml_matrix *matrix_inv(cml_matrix *const a);
static void matrix_lu(cml_matrix *const a, cml_matrix **p, cml_matrix **l, cml_matrix **u);
static cml_matrix *matrix_normalize(cml_matrix *const a);
static void matrix_print(cml_matrix *const a);
static void matrix_set(cml_matrix **a, const lgint i, const lgint j, const fdouble value);
static void matrix_softmax(cml_matrix **a);
static fdouble matrix_trace(cml_matrix *const a);
static void matrix_transpose(cml_matrix *const a, cml_matrix **at);

cml_matrix *matrix_create(const lgint m, const lgint n, void *(*alloc)(size_t))
{
    struct matrix *mat = NULL;
    const size_t size = sizeof(*mat) + m * n * sizeof(*mat->data);
    mat = (struct matrix *)alloc(size);
    *(lgint *)(&mat->pub.m) = m;
    *(lgint *)(&mat->pub.n) = n;

    mat->pub.copy = &matrix_copy;
    mat->pub.det = &matrix_det;
    mat->pub.free = &matrix_free;
    mat->pub.get = &matrix_get;
    mat->pub.hadamard = &matrix_hadamard;
    mat->pub.inv = &matrix_inv;
    mat->pub.lu = &matrix_lu;
    mat->pub.normalize = &matrix_normalize;
    mat->pub.print = &matrix_print;
    mat->pub.set = &matrix_set;
    mat->pub.softmax = &matrix_softmax;
    mat->pub.trace = &matrix_trace;
    mat->pub.transpose = &matrix_transpose;

    return &mat->pub;
}

cml_matrix *cml_matrix_alloc(const lgint m, const lgint n)
{
    return matrix_create(m, n, &malloc);
}

cml_matrix *cml_matrix_confusion(cml_matrix *const yhat, cml_matrix *const y)
{
    if (yhat->m != y->m || yhat->n != y->n)
    {
        fprintf(stderr, "Error (cml_matrix_confusion): the matrices are not of same dimension\n");
        return NULL;
    }

    cml_matrix *report = cml_matrix_zeros(y->n, y->n);
    for (lgint i = 0; i < y->m; i++)
    {
        // get actual/target class of the ith row
        lgint target_class = 0;
        for (lgint j = 0; j < y->n; j++)
        {
            if (y->get(y, i, j) == 1)
            {
                target_class = j;
                break;
            }
        }

        // get predicted class of the ith row
        lgint predicted_class = 0;
        for (lgint j = 0; j < yhat->n; j++)
        {
            if (yhat->get(yhat, i, j) == 1)
            {
                predicted_class = j;
                break;
            }
        }
        const lgint k = (lgint)report->get(report, target_class, predicted_class);
        report->set(&report, target_class, predicted_class, k + 1);
    }

    return report;
}

cml_matrix *cml_matrix_dif(cml_matrix *const a, cml_matrix *const b)
{
    if (a == NULL)
    {
        fprintf(stderr, "error (cml_matrix_dif): the matrix A is null in A-B.\n");
        return NULL;
    }
    if (b == NULL)
    {
        fprintf(stderr, "error (cml_matrix_dif): the matrix B is null in A-B.\n");
        return NULL;
    }
    if (a->m != b->m || a->n != b->n)
    {
        fprintf(stderr, "error (cml_matrix_dif): the matrices should be of same dimension in A-B.\n");
        return NULL;
    }
    cml_matrix *d = cml_matrix_alloc(a->m, a->n);
    for (lgint i = 0; i < d->m; i++)
    {
        for (lgint j = 0; j < d->n; j++)
        {
            d->set(&d, i, j, a->get(a, i, j) - b->get(b, i, j));
        }
    }
    return d;
}

cml_matrix *cml_matrix_eye(const lgint n)
{
    cml_matrix *a = cml_matrix_zeros(n, n);
    for (lgint i = 0; i < a->m; i++)
    {
        a->set(&a, i, i, 1.);
    }
    return a;
}

cml_matrix *cml_matrix_prod(cml_matrix *const a, cml_matrix *const b)
{
    if (a == NULL)
    {
        fprintf(stderr, "error (cml_matrix_prod): the matrix A is null in A*B.\n");
        return NULL;
    }
    if (b == NULL)
    {
        fprintf(stderr, "error (cml_matrix_prod): the matrix B is null in A*B.\n");
        return NULL;
    }
    if (a->n != b->m)
    {
        fprintf(stderr, "error (cml_matrix_prod): the matrices should be product compatible in A*B.\n");
        return NULL;
    }
    cml_matrix *p = cml_matrix_alloc(a->m, b->n);
    for (lgint i = 0; i < p->m; i++)
    {
        for (lgint j = 0; j < p->n; j++)
        {
            fdouble s = 0.;
            for (lgint k = 0; k < a->n; k++)
            {
                s += a->get(a, i, k) * b->get(b, k, j);
            }
            p->set(&p, i, j, s);
        }
    }
    return p;
}

cml_matrix *cml_matrix_solve(cml_matrix *const a, cml_matrix *const b)
{
    if (a == NULL)
    {
        fprintf(stderr, "error (cml_matrix_solve): the matrix A is null in resolution of Ax=b.\n");
        return NULL;
    }
    if (b == NULL)
    {
        fprintf(stderr, "error (cml_matrix_solve): the matrix b is null in resolution of Ax=b.\n");
        return NULL;
    }
    cml_matrix *ainv = a->inv(a);
    if (ainv == NULL)
    {
        fprintf(stderr, "error (cml_matrix_solve): Ax=b has no solution.\n");
        return NULL;
    }
    cml_matrix *x = cml_matrix_prod(ainv, b);
    ainv->free(&ainv);
    return x;
}

cml_matrix *cml_matrix_sum(cml_matrix *const a, cml_matrix *const b)
{
    if (a == NULL)
    {
        fprintf(stderr, "error (cml_matrix_sum): the matrix A is null in A+B.\n");
        return NULL;
    }
    if (b == NULL)
    {
        fprintf(stderr, "error (cml_matrix_sum): the matrix B is null in A+B.\n");
        return NULL;
    }
    if (a->m != b->m || a->n != b->n)
    {
        fprintf(stderr, "error (cml_matrix_sum): the matrices should be of same dimension in A+B.\n");
        return NULL;
    }
    cml_matrix *s = cml_matrix_alloc(a->m, a->n);
    for (lgint i = 0; i < s->m; i++)
    {
        for (lgint j = 0; j < s->n; j++)
        {
            s->set(&s, i, j, a->get(a, i, j) + b->get(b, i, j));
        }
    }
    return s;
}

static void *matrix_alloc(size_t size)
{
    return calloc(size, 1);
}

cml_matrix *cml_matrix_zeros(const lgint m, const lgint n)
{
    return matrix_create(m, n, &matrix_alloc);
}

cml_matrix *matrix_copy(cml_matrix *const a)
{
    if (a == NULL)
        return NULL;
    cml_matrix *b = cml_matrix_alloc(a->m, a->n);
    for (lgint i = 0; i < b->m; i++)
    {
        for (lgint j = 0; j < b->n; j++)
        {
            b->set(&b, i, j, a->get(a, i, j));
        }
    }
    return b;
}

static lgint cml_lu(cml_matrix *const a, cml_matrix **p, cml_matrix **l, cml_matrix **u);

fdouble matrix_det(cml_matrix *const a)
{
    if (a == NULL)
    {
        fprintf(stderr, "error (matrix_det): the matrix is null.\n");
        return DBL_MAX;
    }
    if (a->m == a->n && a->m == 1)
        return a->get(a, 0, 0);

    cml_matrix *p = NULL, *l = NULL, *u = NULL;
    lgint n = cml_lu(a, &p, &l, &u);
    if (p == NULL || l == NULL || u == NULL)
    {
        return 0.;
    }

    fdouble det = 1.;
    for (lgint i = 0; i < a->m; i++)
    {
        det *= u->get(u, i, i);
    }
    p->free(&p);
    l->free(&l);
    u->free(&u);

    return pow(-1., n) * det;
}

void matrix_free(cml_matrix **a)
{
    if (*a == NULL)
        return;
    free(*a);
    *a = NULL;
}

fdouble matrix_get(cml_matrix *const a, const lgint i, const lgint j)
{
    if (a == NULL)
        return DBL_MAX;
    if (a->m <= i || a->n <= j)
    {
        fprintf(stderr, "Error (matrix_get): the index (%ld, %ld) is outside of the matrix dimension (%ld, %ld)\n", i, j, a->m, a->n);
        return DBL_MAX;
    }
    struct matrix *mat = (struct matrix *)a;
    return mat->data[i * a->n + j];
}

cml_matrix *matrix_hadamard(cml_matrix *const a, cml_matrix *const b)
{
    if (a->m != b->m || a->n != b->n)
    {
        fprintf(stderr, "Error (matrix_hadamard): the matrices are not element-wise product.\n");
        return NULL;
    }
    cml_matrix *prod = cml_matrix_alloc(a->m, a->n);
    for (lgint i = 0; i < a->m; i++)
    {
        for (lgint j = 0; j < a->n; j++)
        {
            prod->set(
                &prod,
                i, j,
                a->get(a, i, j) * b->get(b, i, j));
        }
    }
    return prod;
}

static lgint matrix_get_pivot(cml_matrix *const a, const lgint i)
{
    if (a == NULL)
    {
        return -1;
    }
    lgint pivot = i;
    fdouble max = fabs(a->get(a, i, i));
    for (lgint j = i + 1; j < a->n; j++)
    {
        if (max < fabs(a->get(a, j, i)))
        {
            max = fabs(a->get(a, j, i));
            pivot = j;
        }
    }
    return pivot;
}

static void matrix_pivoting(cml_matrix *a[], const lgint n, const lgint pivot, const lgint i)
{
    if (n <= 0 || a == NULL)
        return;
    for (lgint j = 0; j < a[0]->n; j++)
    {
        for (lgint k = 0; k < n; k++)
        {
            const fdouble tmp = a[k]->get(a[k], pivot, j);
            a[k]->set(&a[k], pivot, j, a[k]->get(a[k], i, j));
            a[k]->set(&a[k], i, j, tmp);
        }
    }
}

// row_i <= coef_j*row_j + coef_k*row_k for all the n matrices A
static void matrix_linear_oper(cml_matrix **a, const int n,
                               const lgint row_i,
                               const fdouble coef_j, const lgint row_j,
                               const fdouble coef_k, const lgint row_k)
{
    if (a == NULL || n < 1)
        return;

    for (lgint j = 0; j < a[0]->n; j++)
    {
        for (int i = 0; i < n; i++)
        {
            a[i]->set(
                &a[i], row_i, j,
                coef_j * a[i]->get(a[i], row_j, j) + coef_k * a[i]->get(a[i], row_k, j));
        }
    }
}

cml_matrix *matrix_inv(cml_matrix *const a)
{
    if (a == NULL)
    {
        fprintf(stderr, "error (matrix_inv): the matrix is null.\n");
        return NULL;
    }
    if (a->m != a->n)
    {
        fprintf(stderr, "error (matrix_inv): the matrix should be square.\n");
        return NULL;
    }

    cml_matrix *ainv = cml_matrix_eye(a->m);
    if (ainv == NULL)
    {
        fprintf(stderr, "error (matrix_inv): the allocation memory has failed.");
        return NULL;
    }

    cml_matrix *b = a->copy(a);
    if (b == NULL)
    {
        fprintf(stderr, "error (matrix_inv) - the allocation memory has failed.");
        ainv->free(&ainv);
        return NULL;
    }
    cml_matrix *mts[] = {b, ainv};

    for (lgint i = 0; i < b->m; i++)
    {
        const lgint pivot = matrix_get_pivot(b, i);
        if (pivot != i)
        {
            matrix_pivoting(mts, 2, pivot, i);
        }
        fdouble coef = b->get(b, i, i);
        if (fabs(coef) < CML_MATRIX_TOLERANCE)
        {
            ainv->free(&ainv);
            b->free(&b);
            fprintf(stderr, "error (matrix_inv) - the matrix is singular - it can not be inverted.\n");
            return NULL;
        }
        else
        {
            matrix_linear_oper(mts, 2, i, 1 / coef, i, 0., 0);
            for (lgint it = 0; it < b->m; it++)
            {
                if (it == i)
                    continue;

                const fdouble coef_j = b->get(b, it, i);
                const fdouble coef_k = b->get(b, i, i);

                matrix_linear_oper(mts, 2, it, -coef_j, i, coef_k, it);
            }
        }
    }
    b->free(&b);
    return ainv;
}

lgint cml_lu(cml_matrix *const a, cml_matrix **p, cml_matrix **l, cml_matrix **u)
{
    if (a == NULL)
    {
        fprintf(stderr, "error (cml_lu): the matrix is null.\n");
        return 0;
    }
    if (a->m != a->n)
    {
        fprintf(stderr, "error (cml_lu): the matrix should be square.\n");
        return 0;
    }
    lgint i, j, k;
    //  initialize permutation matrix `p`, `l` and `u`
    *p = cml_matrix_eye(a->m);
    *l = cml_matrix_zeros(a->m, a->n);
    *u = a->copy(a);

    lgint permutations = 0;
    for (i = 0; i < (*u)->m; i++)
    {
        const lgint pivot = matrix_get_pivot(*u, i);

        // swap rows pivot and `i`
        if (pivot != i)
        {
            cml_matrix *at[] = {*p, *l, *u};
            matrix_pivoting(at, 3, pivot, i);
            permutations++;
        }

        // perform the decomposition
        for (j = i + 1; j < a->m; j++)
        {
            const fdouble coef = (*u)->get(*u, i, i);
            if (fabs(coef) < CML_MATRIX_TOLERANCE)
            {
                fprintf(stderr, "error (cml_lu): the matrix is singular.\n");
                (*p)->free(p);
                (*l)->free(l);
                (*u)->free(u);
                return 0;
            }
            (*l)->set(l, j, i, (*u)->get(*u, j, i) / coef);
            for (k = i; k < a->m; k++)
            {
                (*u)->set(u, j, k, (*u)->get(*u, j, k) - (*l)->get(*l, j, i) * (*u)->get(*u, i, k));
            }
        }
    }

    // check last row
    bool is_zeros = true;
    for (j = 0; j < (*u)->n; j++)
    {
        if (fabs((*u)->get(*u, (*u)->m - 1, j)) > CML_MATRIX_TOLERANCE)
        {
            is_zeros = false;
            break;
        }
    }
    if (is_zeros)
    {
        fprintf(stderr, "error (cml_lu): the matrix is singular.\n");
        (*p)->free(p);
        (*l)->free(l);
        (*u)->free(u);
        return 0;
    }

    // set the diagonal of `l` to 1
    for (i = 0; i < a->m; i++)
    {
        (*l)->set(l, i, i, 1.);
    }

    return permutations;
}

void matrix_lu(cml_matrix *const a, cml_matrix **p, cml_matrix **l, cml_matrix **u)
{
    cml_lu(a, p, l, u);
}

cml_matrix *matrix_normalize(cml_matrix *const a)
{
    if (a == NULL)
        return NULL;

    cml_matrix *a_norm = cml_matrix_alloc(a->m, a->n);
    for (lgint j = 0; j < a->n; j++)
    {
        // get the max of jth column
        lgint i_max = 0;
        fdouble fmax = DBL_MIN;
        for (lgint i = 0; i < a->m; i++)
        {
            if (fmax < a->get(a, i, j))
            {
                fmax = a->get(a, i, j);
                i_max = i;
            }
        }
        // divide by the max
        const fdouble coef = a->get(a, i_max, j);
        if (coef != 0)
        {
            for (lgint i = 0; i < a_norm->m; i++)
            {
                a_norm->set(&a_norm, i, j, a->get(a, i, j) / coef);
            }
        }
    }
    return a_norm;
}

void matrix_print(cml_matrix *const a)
{
    if (a == NULL)
        return;
    printf("Matrix(%ld, %ld) = \n", a->m, a->n);
    printf("[\n");
    for (lgint i = 0; i < a->m; i++)
    {
        printf("   [");
        for (lgint j = 0; j < a->n; j++)
        {
            if (j == 0)
                printf("%lg", a->get(a, i, j));
            else
                printf(", %lg", a->get(a, i, j));
        }
        printf("]\n");
    }
    printf("]\n");
}

void matrix_set(cml_matrix **a, const lgint i, const lgint j, const fdouble value)
{
    if (*a == NULL)
        return;
    if (i >= (*a)->m || j >= (*a)->n)
    {
        fprintf(stderr, "Error (matrix_set): the index (%ld, %ld) is outside of the matrix dimension (%ld, %ld)\n", i, j, (*a)->m, (*a)->n);
        return;
    }
    struct matrix *mat = (struct matrix *)(*a);
    mat->data[i * (*a)->n + j] = value;
}

void matrix_softmax(cml_matrix **a)
{
    for (lgint i = 0; i < (*a)->m; i++)
    {
        lgint index = 0;
        fdouble fmax = DBL_MIN;
        for (lgint j = 0; j < (*a)->n; j++)
        {
            const fdouble aij = (*a)->get(*a, i, j);
            if (fmax < aij)
            {
                fmax = aij;
                index = j;
            }
        }

        for (lgint j = 0; j < (*a)->n; j++)
        {
            (*a)->set(a, i, j, index == j);
        }
    }
}

fdouble matrix_trace(cml_matrix *const a)
{
    if (a == NULL)
        return DBL_MAX;
    fdouble trace = 0.;
    const lgint n = (a->m < a->n) ? a->m : a->n;
    for (lgint i = 0; i < n; i++)
    {
        trace += a->get(a, i, i);
    }
    return trace;
}

void matrix_transpose(cml_matrix *const a, cml_matrix **at)
{
    if (a == NULL)
        return;
    if (*at == NULL)
        *at = cml_matrix_alloc(a->n, a->m);
    if ((*at)->m != a->n || (*at)->n != a->m)
    {
        fprintf(stderr, "error (matrix_transpose): the matrix transpose has bad dimension.\n");
        return;
    }
    for (lgint i = 0; i < (*at)->m; i++)
    {
        for (lgint j = 0; j < (*at)->n; j++)
        {
            (*at)->set(at, i, j, a->get(a, j, i));
        }
    }
}