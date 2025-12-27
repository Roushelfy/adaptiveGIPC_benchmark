///////////////////////////////////////////////////////////////////////////////////////////
// CUDA 12 Wrapper for CUDA_PROJECTIVE_TET_MESH.h
// Includes compatibility layer for deprecated cuSPARSE APIs
///////////////////////////////////////////////////////////////////////////////////////////
#ifndef __CUDA_PROJECTIVE_TET_MESH_WRAPPER_H__
#define __CUDA_PROJECTIVE_TET_MESH_WRAPPER_H__

// First include the original header
#include "../lib/CUDA_PROJECTIVE_TET_MESH.h"

// Then include our compatibility layer
#include "cusparse_compat.h"

#endif
