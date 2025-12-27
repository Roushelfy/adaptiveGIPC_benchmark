#ifndef HELPER_CUDA_H
#define HELPER_CUDA_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Simple error checking macro for CUDA calls
#ifndef checkCudaErrors
#define checkCudaErrors(call) call
#endif

#endif // HELPER_CUDA_H
