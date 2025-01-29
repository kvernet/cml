#ifndef cml_data_h
#define cml_data_h

#include "cml_matrix.h"

#include <stdbool.h>

#ifdef __cplusplus
extern "C"
{
#endif

    void cml_data_read(cml_matrix **x, cml_matrix **y, const char *file_path, const char *delimiter, bool has_header);

    void cml_data_split(
        cml_matrix **train_data_x, cml_matrix **train_data_y,
        cml_matrix **val_data_x, cml_matrix **val_data_y,
        cml_matrix **test_data_x, cml_matrix **test_data_y,
        cml_matrix *const x, cml_matrix *const y,
        const fdouble val_percentage, const fdouble test_percentage,
        bool shuffle);

#ifdef __cplusplus
}
#endif

#endif