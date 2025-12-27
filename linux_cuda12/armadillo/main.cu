///////////////////////////////////////////////////////////////////////////////////////////
// Linux CUDA 12 version - Armadillo (Tetrahedral Mesh) Simulation
// Converted from Windows CUDA 10.0
///////////////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "OPENGL_DRIVER_LINUX.h"

int main(int argc, char *argv[])
{
    printf("PainlessMG - Armadillo Simulation (Linux CUDA 12)\n");
    printf("=================================================\n");

    // Initialize CUDA
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        printf("ERROR: No CUDA devices found!\n");
        return 1;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Using GPU: %s\n", deviceProp.name);
    printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("=================================================\n\n");

    OPENGL_DRIVER(&argc, argv);
    return 0;
}
