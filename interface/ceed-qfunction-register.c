#include <ceed-impl.h>
#include <stdio.h>

static bool register_all_called;

#define MACRO(name) CEED_INTERN int name(void);
#include "../gallery/ceed-gallery-list.h"
#undef MACRO

/**
  @brief Register the gallery of preconfigured QFunctions.

  This is called automatically by CeedQFunctionCreateInteriorByName() and thus normally need not be called by users.
  Users can call CeedQFunctionRegister() to register additional backends.

  @return An error code: 0 - success, otherwise - failure

  @sa CeedQFunctionRegister()

  @ref Backend
**/
int CeedQFunctionRegisterAll() {
  if (register_all_called) return 0;
  register_all_called = true;

#define MACRO(name) CeedChk(name());
#include "../gallery/ceed-gallery-list.h"
#undef MACRO
  return 0;
}
