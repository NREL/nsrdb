/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by  The HDF Group and                                           *
 *               The Board of Trustees of the University of Illinois.        *
 * All rights reserved.                                                      *
 *                                                                           *
 * This file is part of H4H5TOOLS. The full H4H5TOOLS copyright notice,      *
 * including terms governing use, modification, and redistribution, is       *
 * contained in the files COPYING and Copyright.html.  COPYING can be found  *
 * at the root of the source code distribution tree; Copyright.html can be   *
 * found at the root level of an installed copy of the electronic H4H5TOOLS  *
 * document set, is linked from the top-level documents page, and can be     *
 * found at http://www.hdfgroup.org/h4toh5/Copyright.html.  If you do not    *
 * have access to either file, you may request a copy from help@hdfgroup.org.*
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#ifndef H4toH5API_ADPT_H
#define H4toH5API_ADPT_H

/* This will only be defined if HDF4 was built with CMake */
#ifdef H4H5_BUILT_AS_DYNAMIC_LIB

#if defined(h4toh5_EXPORTS)
  #if defined (_MSC_VER)  /* MSVC Compiler Case */
    #define __DLL425__ __declspec(dllexport)
  #elif (__GNUC__ >= 4)  /* GCC 4.x has support for visibility options */
    #define __DLL425__ __attribute__ ((visibility("default")))
  #endif
#else
  #if defined (_MSC_VER)  /* MSVC Compiler Case */
    #define __DLL425__ __declspec(dllimport)
  #elif (__GNUC__ >= 4)  /* GCC 4.x has support for visibility options */
    #define __DLL425__ __attribute__ ((visibility("default")))
  #endif
#endif

#ifndef __DLL425__
  #define __DLL425__
#endif /* h4h5_EXPORTS */

#elif (H4H5_BUILT_AS_STATIC_LIB)
  #define __DLL425__

#else
/* This is the original HDFGroup defined preprocessor code which should still work
 * with the VS projects that are maintained by "The HDF Group"
 * This will be removed after the next release.
 */

#if defined(WIN32)
# if defined(_H4TOH5DLL_)
#  pragma warning(disable: 4273)	/* Disable the dll linkage warnings */
#  define __DLL425__ __declspec(dllexport)
/*#define __DLLVARH425__ __declspec(dllexport)*/
# elif defined(_H4TOH5USEDLL_)
#  define __DLL425__ __declspec(dllimport)
/*#define __DLLVARH425__ __declspec(dllimport)*/
# else
#  define __DLL425__
/*#define __DLLVARH425__ extern*/
# endif /* _H4TOH5DLL_ */

#else /*WIN32*/
# define __DLL425__
/*#define __DLLVAR__ extern*/
#endif

#endif /*H4H5_BUILT_AS_DYNAMIC_LIB  */

#endif /* H4to5API_ADPT_H */
