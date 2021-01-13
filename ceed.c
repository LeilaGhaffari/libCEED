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

#define _POSIX_C_SOURCE 200112 // LG: If you define this macro to a value greater 
                               //     than or equal to 200112L, then the functionality  
                               //     from the 2001 edition of the POSIX standard 
                               //     (IEEE Standard 1003.1-2001) is made available. 

                               // The Portable Operating System Interface (POSIX) 
                               // is a family of standards specified by the IEEE 
                               // Computer Society for maintaining compatibility 
                               // between operating systems.
#include <ceed-impl.h>

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/// @cond DOXYGEN_SKIP
static CeedRequest ceed_request_immediate;

static struct {
  char prefix[CEED_MAX_RESOURCE_LEN];
  int (*init)(const char *resource, Ceed f);
} backends[32];
static size_t num_backends;
/// @endcond

/// @file
/// Implementation of core components of Ceed library
///
/// @defgroup Ceed Ceed: core components
/// @{

/// Request immediate completion
///
/// This predefined constant is passed as the \ref CeedRequest argument to
/// interfaces when the caller wishes for the operation to be performed
/// immediately.  The code
///
/// @code
///   CeedOperatorApply(op, ..., CEED_REQUEST_IMMEDIATE);
/// @endcode
///
/// is semantically equivalent to
///
/// @code
///   CeedRequest request;
///   CeedOperatorApply(op, ..., &request);
///   CeedRequestWait(&request);
/// @endcode
CeedRequest *const CEED_REQUEST_IMMEDIATE = &ceed_request_immediate;

/// Error handling implementation; use \ref CeedError instead.
int CeedErrorImpl(Ceed ceed, const char *filename, int lineno, const char *func,
                  int ecode, const char *format, ...) {
  va_list args; // LG: initializes an object of this type in such a way that subsequent 
                //     calls to va_arg sequentially retrieve the additional arguments passed to the function.
  va_start(args, format);
  if (ceed) return ceed->Error(ceed, filename, lineno, func, ecode, format, args);
  return CeedErrorAbort(ceed, filename, lineno, func, ecode, format, args);
}

/// Error handler that returns without printing anything.
///
/// Pass this to CeedSetErrorHandler() to obtain this error handling behavior.
///
/// @sa CeedErrorAbort
int CeedErrorReturn(Ceed ceed, const char *filename, int lineno,
                    const char *func, int ecode, const char *format,
                    va_list args) {
  return ecode;
}

/// Error handler that prints to stderr and aborts
///
/// Pass this to CeedSetErrorHandler() to obtain this error handling behavior.
///
/// @sa CeedErrorReturn
int CeedErrorAbort(Ceed ceed, const char *filename, int lineno,
                   const char *func, int ecode,
                   const char *format, va_list args) {
  fprintf(stderr, "%s:%d in %s(): ", filename, lineno, func);
  vfprintf(stderr, format, args); // Writes the C string pointed by format to the stream, 
                                  // replacing any format specifier in the same way as printf does, 
                                  // but using the elements in the variable argument list identified 
                                  // by arg instead of additional function arguments.
                                  // stderr: standard error
  fprintf(stderr, "\n");
  abort();
  return ecode;
}

/// Set error handler
///
/// A default error handler is set in CeedInit().  Use this function to change
/// the error handler to CeedErrorReturn(), CeedErrorAbort(), or a user-defined
/// error handler.
int CeedSetErrorHandler(Ceed ceed,
                        int (eh)(Ceed, const char *, int, const char *,
                                 int, const char *, va_list)) {
  ceed->Error = eh;
  return 0;
}

/**
  Register a Ceed backend

  @param prefix Prefix of resources for this backend to respond to.  For
                example, the reference backend responds to "/cpu/self".
  @param init   Initialization function called by CeedInit() when the backend
                is selected to drive the requested resource.
  @return 0 on success
 */
int CeedRegister(const char *prefix,
                 int (*init)(const char *, Ceed)) {
  if (num_backends >= sizeof(backends) / sizeof(backends[0])) {
    return CeedError(NULL, 1, "Too many backends");
  }
  strncpy(backends[num_backends].prefix, prefix, CEED_MAX_RESOURCE_LEN);
  backends[num_backends].init = init;
  num_backends++;
  return 0;
}

/// Allocate an array on the host; use CeedMalloc()
///
/// Memory usage can be tracked by the library.  This ensures sufficient
/// alignment for vectorization and should be used for large allocations.
///
/// @param n Number of units to allocate
/// @param unit Size of each unit
/// @param p Address of pointer to hold the result.
/// @sa CeedFree()
int CeedMallocArray(size_t n, size_t unit, void *p) {
  int ierr = posix_memalign((void **)p, CEED_ALIGN, n*unit);  // allocate aligned memory
  if (ierr)
    return CeedError(NULL, ierr,
                     "posix_memalign failed to allocate %zd members of size %zd\n", n, unit);
                     // LG: Here is the place where more arguments are need since we want to 
                     //     report errors with strings and numbers.
  return 0;
}

/// Allocate a cleared (zeroed) array on the host; use CeedCalloc()
///
/// Memory usage can be tracked by the library.
///
/// @param n Number of units to allocate
/// @param unit Size of each unit
/// @param p Address of pointer to hold the result.
/// @sa CeedFree()
int CeedCallocArray(size_t n, size_t unit, void *p) {
  *(void **)p = calloc(n, unit);
  if (n && unit && !*(void **)p)
    return CeedError(NULL, 1, "calloc failed to allocate %zd members of size %zd\n",
                     n, unit);  // LG: Why it is allocated in a different way?
  return 0;
}

/// Free memory allocated using CeedMalloc() or CeedCalloc()
///
/// @param p address of pointer to memory.  This argument is of type void* to avoid needing a cast, 
             // but is the address of the pointer (which is zeroed) rather than the pointer.

int CeedFree(void *p) { // LG: void pointer => has no associated data type with it
                        //     void pointers cannot be dereferenced
  
  free(*(void **)p);   // LG: A void** is a pointer to a void*. 
                           // void** is not special in any way - it's just a pointer to something, where that something happens to be a void*.
                           // (void**)something casts something to a void**.
                           // *something dereferences something.
                           // Therefore, *(void**)something casts something to a void**, 
                           // and then dereferences it (yielding a void*).
 
  *(void **)p = NULL;   // LG: NULL pointer => 1- Initialize a pointer
                                            // 2- Check for NULL pointer. 
                                            //    perform error handling in pointer related code e.g. 
                                            //    dereference pointer variable only if it’s not NULL.
                                            // 3- To pass a null pointer to a function argument when we 
                                            //    don’t want to pass any valid memory address.
  return 0;
}

/**
  Wait for a CeedRequest to complete.

  Calling CeedRequestWait on a NULL request is a no-op.

  @param req Address of CeedRequest to wait for; zeroed on completion.
  @return 0 on success
 */
int CeedRequestWait(CeedRequest *req) {
  if (!*req) return 0;
  return CeedError(NULL, 2, "CeedRequestWait not implemented");
}

/// Initialize a \ref Ceed to use the specified resource.
///
/// @param resource  Resource to use, e.g., "/cpu/self"
/// @param ceed The library context
/// @sa CeedRegister() CeedDestroy()
int CeedInit(const char *resource, Ceed *ceed) {
  int ierr;
  size_t matchlen = 0, matchidx;  
  // LG: size_t= unsigned integer data type which can assign only 0 
  //     and greater than 0 integer values. It measure bytes of any object's size and 
  //     returned by sizeof operator. const is the syntax representation of size_t , 
  //     but without const you can run the programm.

  if (!resource) return CeedError(NULL, 1, "No resource provided");
  for (size_t i=0; i<num_backends; i++) {
    size_t n;
    const char *prefix = backends[i].prefix;
    for (n = 0; prefix[n] && prefix[n] == resource[n]; n++) {}  // LG: What is this condition?
    if (n > matchlen) {
      matchlen = n;
      matchidx = i;
    }
  }
  if (!matchlen) return CeedError(NULL, 1, "No suitable backend");
  ierr = CeedCalloc(1,ceed); CeedChk(ierr);
  (*ceed)->Error = CeedErrorAbort;
  (*ceed)->data = NULL;
  ierr = backends[matchidx].init(resource, *ceed); CeedChk(ierr);
  return 0;
}

/**
  Destroy a Ceed context

  @param ceed Address of Ceed context to destroy
  @return 0 on success
 */
int CeedDestroy(Ceed *ceed) {
  int ierr;

  if (!*ceed) return 0;
  if ((*ceed)->Destroy) {
    ierr = (*ceed)->Destroy(*ceed); CeedChk(ierr);
  }
  ierr = CeedFree(ceed); CeedChk(ierr);
  return 0;
}

/// @}

/**
  Private printf-style debugging with color for the terminal

  @param format printf-style format string
  @param ... arguments as specified in format string
 */
void CeedDebug(const char *format,...) {
  // real slow, should use NDEBUG to ifdef the body
  if (!getenv("CEED_DEBUG")) return;
  va_list args;
  va_start(args, format);
  fflush(stdout);
  fprintf(stdout,"\033[32m");
  vfprintf(stdout,format,args);
  fprintf(stdout,"\033[m");
  fprintf(stdout,"\n");
  fflush(stdout);
  va_end(args);
}
