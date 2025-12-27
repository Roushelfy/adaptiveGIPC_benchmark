///////////////////////////////////////////////////////////////////////////////////////////
// cuSPARSE Compatibility Layer for CUDA 12
// Maps deprecated CUDA 10 cuSPARSE API to CUDA 12 API
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef __CUSPARSE_COMPAT_H__
#define __CUSPARSE_COMPAT_H__

#include <cusparse.h>

// The old cusparseSolveAnalysisInfo_t type is removed in CUDA 11+
// We create a dummy pointer type for compatibility
typedef void* cusparseSolveAnalysisInfo_t;

// Dummy functions for compatibility (these are no-ops in modern cuSPARSE)
inline cusparseStatus_t cusparseCreateSolveAnalysisInfo(cusparseSolveAnalysisInfo_t* info) {
    // Modern cuSPARSE doesn't need pre-analysis for triangular solves
    // Just allocate a small dummy structure to avoid null pointer issues
    *info = malloc(sizeof(int));
    return (*info) ? CUSPARSE_STATUS_SUCCESS : CUSPARSE_STATUS_ALLOC_FAILED;
}

inline cusparseStatus_t cusparseDestroySolveAnalysisInfo(cusparseSolveAnalysisInfo_t info) {
    if (info) {
        free(info);
    }
    return CUSPARSE_STATUS_SUCCESS;
}

// Compatibility wrapper for deprecated cusparseScsr2csc (float version)
// Converts CSR to CSC format using modern cuSPARSE API
inline cusparseStatus_t cusparseScsr2csc(
    cusparseHandle_t handle,
    int m, int n, int nnz,
    const float *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    float *cscVal,
    int *cscRowInd,
    int *cscColPtr,
    cusparseAction_t copyValues,
    cusparseIndexBase_t idxBase)
{
    // In CUDA 12, use cusparseCsr2cscEx2
    size_t bufferSize = 0;
    void* buffer = NULL;

    // Get buffer size
    cusparseStatus_t status = cusparseCsr2cscEx2_bufferSize(
        handle, m, n, nnz,
        csrVal, csrRowPtr, csrColInd,
        cscVal, cscColPtr, cscRowInd,
        CUDA_R_32F, copyValues, idxBase,
        CUSPARSE_CSR2CSC_ALG1, &bufferSize);

    if (status != CUSPARSE_STATUS_SUCCESS) return status;

    // Allocate buffer
    if (bufferSize > 0) {
        cudaMalloc(&buffer, bufferSize);
    }

    // Perform conversion
    status = cusparseCsr2cscEx2(
        handle, m, n, nnz,
        csrVal, csrRowPtr, csrColInd,
        cscVal, cscColPtr, cscRowInd,
        CUDA_R_32F, copyValues, idxBase,
        CUSPARSE_CSR2CSC_ALG1, buffer);

    // Free buffer
    if (buffer) {
        cudaFree(buffer);
    }

    return status;
}

// Compatibility wrapper for deprecated cusparseDcsr2csc (double version)
inline cusparseStatus_t cusparseDcsr2csc(
    cusparseHandle_t handle,
    int m, int n, int nnz,
    const double *csrVal,
    const int *csrRowPtr,
    const int *csrColInd,
    double *cscVal,
    int *cscRowInd,
    int *cscColPtr,
    cusparseAction_t copyValues,
    cusparseIndexBase_t idxBase)
{
    size_t bufferSize = 0;
    void* buffer = NULL;

    cusparseStatus_t status = cusparseCsr2cscEx2_bufferSize(
        handle, m, n, nnz,
        csrVal, csrRowPtr, csrColInd,
        cscVal, cscColPtr, cscRowInd,
        CUDA_R_64F, copyValues, idxBase,
        CUSPARSE_CSR2CSC_ALG1, &bufferSize);

    if (status != CUSPARSE_STATUS_SUCCESS) return status;

    if (bufferSize > 0) {
        cudaMalloc(&buffer, bufferSize);
    }

    status = cusparseCsr2cscEx2(
        handle, m, n, nnz,
        csrVal, csrRowPtr, csrColInd,
        cscVal, cscColPtr, cscRowInd,
        CUDA_R_64F, copyValues, idxBase,
        CUSPARSE_CSR2CSC_ALG1, buffer);

    if (buffer) {
        cudaFree(buffer);
    }

    return status;
}

// Compatibility wrapper for deprecated cusparseScsrmv (float version)
// Performs sparse matrix-vector multiplication: y = alpha * A * x + beta * y
inline cusparseStatus_t cusparseScsrmv(
    cusparseHandle_t handle,
    cusparseOperation_t transA,
    int m, int n, int nnz,
    const float *alpha,
    const cusparseMatDescr_t descrA,
    const float *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const float *x,
    const float *beta,
    float *y)
{
    // Create sparse matrix descriptor
    cusparseSpMatDescr_t matA;
    cusparseStatus_t status = cusparseCreateCsr(
        &matA, m, n, nnz,
        (void*)csrRowPtrA, (void*)csrColIndA, (void*)csrValA,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    if (status != CUSPARSE_STATUS_SUCCESS) return status;

    // Create dense vector descriptors
    cusparseDnVecDescr_t vecX, vecY;
    status = cusparseCreateDnVec(&vecX, n, (void*)x, CUDA_R_32F);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroySpMat(matA);
        return status;
    }

    status = cusparseCreateDnVec(&vecY, m, (void*)y, CUDA_R_32F);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroyDnVec(vecX);
        cusparseDestroySpMat(matA);
        return status;
    }

    // Get buffer size
    size_t bufferSize = 0;
    status = cusparseSpMV_bufferSize(
        handle, transA,
        alpha, matA, vecX, beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroyDnVec(vecY);
        cusparseDestroyDnVec(vecX);
        cusparseDestroySpMat(matA);
        return status;
    }

    // Allocate buffer
    void* buffer = NULL;
    if (bufferSize > 0) {
        cudaMalloc(&buffer, bufferSize);
    }

    // Perform SpMV
    status = cusparseSpMV(
        handle, transA,
        alpha, matA, vecX, beta, vecY,
        CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer);

    // Cleanup
    if (buffer) cudaFree(buffer);
    cusparseDestroyDnVec(vecY);
    cusparseDestroyDnVec(vecX);
    cusparseDestroySpMat(matA);

    return status;
}

// Compatibility wrapper for deprecated cusparseDcsrmv (double version)
inline cusparseStatus_t cusparseDcsrmv(
    cusparseHandle_t handle,
    cusparseOperation_t transA,
    int m, int n, int nnz,
    const double *alpha,
    const cusparseMatDescr_t descrA,
    const double *csrValA,
    const int *csrRowPtrA,
    const int *csrColIndA,
    const double *x,
    const double *beta,
    double *y)
{
    cusparseSpMatDescr_t matA;
    cusparseStatus_t status = cusparseCreateCsr(
        &matA, m, n, nnz,
        (void*)csrRowPtrA, (void*)csrColIndA, (void*)csrValA,
        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
        CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

    if (status != CUSPARSE_STATUS_SUCCESS) return status;

    cusparseDnVecDescr_t vecX, vecY;
    status = cusparseCreateDnVec(&vecX, n, (void*)x, CUDA_R_64F);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroySpMat(matA);
        return status;
    }

    status = cusparseCreateDnVec(&vecY, m, (void*)y, CUDA_R_64F);
    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroyDnVec(vecX);
        cusparseDestroySpMat(matA);
        return status;
    }

    size_t bufferSize = 0;
    status = cusparseSpMV_bufferSize(
        handle, transA,
        alpha, matA, vecX, beta, vecY,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize);

    if (status != CUSPARSE_STATUS_SUCCESS) {
        cusparseDestroyDnVec(vecY);
        cusparseDestroyDnVec(vecX);
        cusparseDestroySpMat(matA);
        return status;
    }

    void* buffer = NULL;
    if (bufferSize > 0) {
        cudaMalloc(&buffer, bufferSize);
    }

    status = cusparseSpMV(
        handle, transA,
        alpha, matA, vecX, beta, vecY,
        CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buffer);

    if (buffer) cudaFree(buffer);
    cusparseDestroyDnVec(vecY);
    cusparseDestroyDnVec(vecX);
    cusparseDestroySpMat(matA);

    return status;
}

#endif // __CUSPARSE_COMPAT_H__
