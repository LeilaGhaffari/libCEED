// Copyright (c) 2017, Lawrence Livermore National Security, LLC. Produced at
// the Lawrence Livermore National Laboratory. LLNL-CODE-734707. All Rights
// reserved. See files LICENSE and NOTICE for details.
//
// This file is part of CEED, a collection of benchmarks, miniapps, software
// libraries and APIs for efficient high-order finite element and spectral
// element discretizations for exascale applications. For more information and
// source code availability see http://github.com/ceed.
//
// The CEED research is supported by the Exascale Computing Project 17-SC-20-SC,
// a collaborative effort of two U.S. Department of Energy organizations (Office
// of Science and the National Nuclear Security Administration) responsible for
// the planning and preparation of a capable exascale ecosystem, including
// software, applications, hardware, advanced system engineering and early
// testbed platforms, in support of the nation's exascale computing imperative.

#include <ceed.h>
#include <ceed-backend.h>
#include <hip/hip_runtime.h>

const int sizeMax = 16;
__constant__ double c_B[sizeMax*sizeMax];
__constant__ double c_G[sizeMax*sizeMax];

//------------------------------------------------------------------------------
// Interp device initalization
//------------------------------------------------------------------------------
CEED_INTERN int CeedHipInitInterp(CeedScalar *d_B, CeedInt P1d, CeedInt Q1d,
                                  CeedScalar **c_B_ptr) {
  const int Bsize = P1d*Q1d*sizeof(CeedScalar);
  hipMemcpyToSymbol(HIP_SYMBOL(c_B), d_B, Bsize, 0, hipMemcpyDeviceToDevice);
  hipGetSymbolAddress((void **)c_B_ptr, HIP_SYMBOL(c_B));

  return 0;
}

//------------------------------------------------------------------------------
// Grad device initalization
//------------------------------------------------------------------------------
CEED_INTERN int CeedHipInitInterpGrad(CeedScalar *d_B, CeedScalar *d_G,
                                      CeedInt P1d, CeedInt Q1d,
                                      CeedScalar **c_B_ptr,
                                      CeedScalar **c_G_ptr) {
  const int Bsize = P1d*Q1d*sizeof(CeedScalar);
  hipMemcpyToSymbol(HIP_SYMBOL(c_B), d_B, Bsize, 0, hipMemcpyDeviceToDevice);
  hipGetSymbolAddress((void **)c_B_ptr, HIP_SYMBOL(c_B));
  hipMemcpyToSymbol(HIP_SYMBOL(c_G), d_G, Bsize, 0, hipMemcpyDeviceToDevice);
  hipGetSymbolAddress((void **)c_G_ptr, HIP_SYMBOL(c_G));

  return 0;
}
//------------------------------------------------------------------------------
