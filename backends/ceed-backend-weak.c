#include <ceed-backend.h>
#include <stdlib.h>

// This function provides a debug target for weak symbols
static int CeedRegister_Weak(const char *name) {
  if (getenv("CEED_DEBUG")) fprintf(stderr, "Weak %s\n", name);
  return 0;
}

#define MACRO(name)                                                     \
  CEED_INTERN int name(void) __attribute__((weak));                     \
  int name(void) { return CeedRegister_Weak(__func__); }
#include "ceed-backend-list.h"
#undef MACRO
