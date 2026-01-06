///////////////////////////////////////////////////////////////////////////////////////////
// Linux CUDA 12 version - Armadillo (Tetrahedral Mesh) Simulation
// Converted from Windows CUDA 10.0
///////////////////////////////////////////////////////////////////////////////////////////
#include <stdio.h>
#include <string.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "OPENGL_DRIVER_LINUX.h"

int main(int argc, char *argv[])
{
    printf("PainlessMG - Armadillo Simulation (Linux CUDA 12)\n");
    printf("=================================================\n");

    // Parse command-line arguments
    bool use_gui = true;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--no-gui") == 0) {
            use_gui = false;
            printf("Running in headless mode (no GUI)\n");
        }
    }

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

    OPENGL_DRIVER(&argc, argv, use_gui);
    return 0;
}
