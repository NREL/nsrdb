/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 * Copyright by The HDF Group.                                               *
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

/* This is C++ vector-like structure and functions.
 * It was inspired by Dan. Bernstein's stralloc functions in qmail and djbdns.
 */

/* to implement "stralloc.h"
 *
 *		#ifndef _STRALLOC_H
 *		#define _STRALLOC_H
 *		
 *		#ifdef _STRALLOC_H_IMPL
 *		#define _C_VECTOR_IMPL
 *		#endif
 *		
 *		#define USE_INTERNAL_BUFFER
 *		#define INIT_BUFFER (100)
 *		
 *		#include "cvector.h"
 *		DECLARE_VECTOR_ALL(stralloc, char)
 *		  or DECLARE_VECTOR_BIN(stralloc, char)
 *		
 *		#undef USE_INTERNAL_BUFFER
 *		#undef INIT_BUFFER
 *		
 *		#undef _C_VECTOR_IMPL
 *		
 *		#endif
 */

/* If USE_INTERNAL_BUFFER is defined, short string optimization is enabled.
 * INIT_BUFFER is used to specify the size of the initial buffer.
 */

/* Some compilers do not support inlining. If the following macro is defined,
 * this library will not generate any inline functions. Instead, all functions
 * become normal functions.
 */
#define _FOR_NONSTD_COMPILER_

/* By default, all normal functions; i.e., non-inline functions, will be
 * just 'declared'. If the following macro is defined, all functions will
 * be 'defined'.
 */
#define _C_VECTOR_IMPL

/* You can let this library use your own allocation methods. By default,
 * malloc(), realloc() and free() will be used.
 */
#ifdef USER_SPECIFIC_ALLOC
void * _malloc(size_t);
void * _realloc(void *, size_t);
void _free(void *);
void _status(void);
#else
#define _malloc malloc
#define _realloc realloc
#define _free free
#endif
#ifndef FMT_BUFFER
#define FMT_BUFFER 200
#endif



/*
 *   INTERFACE
 */

/* There are four ways to generate C-vector. This will be explained
 * in detail in a cvector.docx.
 */

#define DECLARE_VECTOR_ALL(type_name, internal)									\
	DECLARE_VECTOR_STRUCTURE(type_name, internal)								\
	DECLARE_VECTOR_DATA(type_name, type_name ##_data, internal)				\
	DECLARE_VECTOR_INIT(type_name, type_name ##_init, internal)				\
	DECLARE_VECTOR_READY(type_name, type_name ##_ready, internal)			\
	DECLARE_VECTOR_READYPLUS(type_name, type_name ##_readyplus)				\
	DECLARE_VECTOR_FREE(type_name, type_name ##_free, internal)				\
	DECLARE_VECTOR_EMPTY(type_name, type_name ##_empty, internal)			\
	DECLARE_VECTOR_COPYB(type_name, type_name ##_copyb, internal)			\
	DECLARE_VECTOR_COPY(type_name, type_name ##_copy)							\
	DECLARE_VECTOR_COPYS(type_name, type_name ##_copys, internal)			\
	DECLARE_VECTOR_CATB(type_name, type_name ##_catb, internal)				\
	DECLARE_VECTOR_CAT(type_name, type_name ##_cat)								\
	DECLARE_VECTOR_CATS(type_name, type_name ##_cats, internal)				\
	DECLARE_VECTOR_COPY_FORMAT(type_name, type_name ##_copyf, internal)	\
	DECLARE_VECTOR_CAT_FORMAT(type_name, type_name ##_catf, internal)		\
	DECLARE_VECTOR_APPEND(type_name, type_name ##_append, internal)		\
	DECLARE_VECTOR_CAT_ULONG(type_name, type_name ##_catulong, internal)	\
	DECLARE_VECTOR_CAT_LONG(type_name, type_name ##_catlong, internal)	\
	DECLARE_VECTOR_0(type_name, type_name ##_0, internal)						\
	DECLARE_VECTOR_DUMP(type_name, type_name ##_dump)							\
	DECLARE_VECTOR_INFO(type_name, type_name ##_info)							\
	DECLARE_VECTOR_REF_STRUCTURE(type_name ##_ref, internal)					\
	DECLARE_VECTOR_REF_POINT(type_name ##_ref, type_name ##_point, internal)

#define DECLARE_VECTOR_BIN(type_name, internal)									\
	DECLARE_VECTOR_STRUCTURE(type_name, internal)								\
	DECLARE_VECTOR_DATA(type_name, type_name ##_data, internal)				\
	DECLARE_VECTOR_INIT2(type_name, type_name ##_init, internal)			\
	DECLARE_VECTOR_READY(type_name, type_name ##_ready, internal)			\
	DECLARE_VECTOR_READYPLUS(type_name, type_name ##_readyplus)				\
	DECLARE_VECTOR_FREE2(type_name, type_name ##_free, internal)			\
	DECLARE_VECTOR_EMPTY2(type_name, type_name ##_empty, internal)			\
	DECLARE_VECTOR_COPYB2(type_name, type_name ##_copyb, internal)			\
	DECLARE_VECTOR_COPY(type_name, type_name ##_copy)							\
	DECLARE_VECTOR_CATB2(type_name, type_name ##_catb, internal)			\
	DECLARE_VECTOR_CAT(type_name, type_name ##_cat)								\
	DECLARE_VECTOR_APPEND(type_name, type_name ##_append, internal)		\
	DECLARE_VECTOR_INFO(type_name, type_name ##_info)							\
	DECLARE_VECTOR_REF_STRUCTURE(type_name ##_ref, internal)					\
	DECLARE_VECTOR_REF_POINT(type_name ##_ref, type_name ##_point, internal)

#define DECLARE_VECTOR_ALLOCATOR(type_name, internal)							\
	DECLARE_VECTOR_STRUCTURE(type_name, internal)								\
	DECLARE_VECTOR_DATA(type_name, type_name ##_data, internal)				\
	DECLARE_VECTOR_INIT2(type_name, type_name ##_init, internal)			\
	DECLARE_VECTOR_READY(type_name, type_name ##_ready, internal)			\
	DECLARE_VECTOR_READYPLUS(type_name, type_name ##_readyplus)				\
	DECLARE_VECTOR_FREE2(type_name, type_name ##_free, internal)			\
	DECLARE_VECTOR_EMPTY2(type_name, type_name ##_empty, internal)			\
	DECLARE_VECTOR_INFO(type_name, type_name ##_info)

#define DECLARE_VECTOR_ALLOCATOR_FREE(type_name, internal)							\
	DECLARE_VECTOR_STRUCTURE(type_name, internal)								\
	DECLARE_VECTOR_DATA(type_name, type_name ##_data, internal)				\
	DECLARE_VECTOR_INIT2(type_name, type_name ##_init, internal)			\
	DECLARE_VECTOR_READY(type_name, type_name ##_ready, internal)			\
	DECLARE_VECTOR_READYPLUS(type_name, type_name ##_readyplus)				\
	DECLARE_VECTOR_FREE3(type_name, type_name ##_free, internal, free_ ##internal)			\
	DECLARE_VECTOR_EMPTY3(type_name, type_name ##_empty, internal, free_ ##internal)	\
	DECLARE_VECTOR_INFO(type_name, type_name ##_info)


/*
 *   Avoid warnings caused by redefining macros.
 */

#ifdef DECLARE_VECTOR_STRUCTURE
#undef DECLARE_VECTOR_STRUCTURE
#undef DECLARE_VECTOR_DATA
#undef DECLARE_VECTOR_INIT
#undef DECLARE_VECTOR_READY
#undef DECLARE_VECTOR_READYPLUS
#undef DECLARE_VECTOR_FREE
#undef DECLARE_VECTOR_EMPTY
#undef DECLARE_VECTOR_COPYB
#undef DECLARE_VECTOR_COPY
#undef DECLARE_VECTOR_COPYS
#undef DECLARE_VECTOR_CATB
#undef DECLARE_VECTOR_CAT
#undef DECLARE_VECTOR_CATS
#undef DECLARE_VECTOR_COPY_FORMAT
#undef DECLARE_VECTOR_CAT_FORMAT
#undef DECLARE_VECTOR_APPEND
#undef DECLARE_VECTOR_CAT_ULONG
#undef DECLARE_VECTOR_CAT_LONG
#undef DECLARE_VECTOR_0
#undef DECLARE_VECTOR_DUMP
#undef DECLARE_VECTOR_INFO
#undef DECLARE_VECTOR_REF_STRUCTURE
#undef DECLARE_VECTOR_REF_POINT
#undef DECLARE_VECTOR_INIT2
#undef DECLARE_VECTOR_FREE2
#undef DECLARE_VECTOR_EMPTY2
#undef DECLARE_VECTOR_COPYB2
#undef DECLARE_VECTOR_CATB2
#undef DECLARE_VECTOR_FREE3
#undef DECLARE_VECTOR_EMPTY3
#endif



/*
 *   TYPE
 */

#ifdef _C_VECTOR_IMPL
#define INLINE  inline
#include <stdarg.h>
#else
#ifdef __GNUC__
/* GCC does not conform to the C99 standard by default */
#define INLINE  extern inline
#else
#define INLINE  inline
#endif
#endif

#ifdef USE_INTERNAL_BUFFER
#define DECLARE_VECTOR_STRUCTURE(type_name, internal)	\
	typedef struct {												\
		internal *s;												\
		int len;													\
		internal *dyn;												\
		int error;													\
		int capacity;									\
		int valid;													\
		internal sta[INIT_BUFFER + 1];						\
	} type_name;
#else /* USE_INTERNAL_BUFFER */
#define DECLARE_VECTOR_STRUCTURE(type_name, internal)	\
	typedef struct {												\
		internal *s;												\
		int len;													\
		internal *dyn;												\
		int error;													\
		int capacity;									\
	} type_name;
#endif

#define DECLARE_VECTOR_REF_STRUCTURE(type_name, internal)	\
	typedef struct {														\
		const internal *s;												\
		int len;														\
	} type_name;



/*
 *   MACRO
 *
 *     With _FOR_NONSTD_COMPILER_, inlined functions are expanded to normal
 *     functions. However, the most reliable solution is buying C++ compiler.
 *     It recognizes 'inline' keyword and Pro* C can be executed with CPP
 *     option. Also, this file is not necessary because STL has string,
 *     vector<>.
 */

#ifndef _FOR_NONSTD_COMPILER_

#ifdef USE_INTERNAL_BUFFER
#define DECLARE_VECTOR_DATA(type_name, func_name, internal)	\
INLINE internal * func_name(type_name *vc)						\
{																				\
	return vc->valid ? vc->dyn : vc->sta;							\
}
#else
#define DECLARE_VECTOR_DATA(type_name, func_name, internal)	\
INLINE internal * func_name(type_name *vc)						\
{																				\
	return vc->dyn;														\
}
#endif

#define DECLARE_VECTOR_EMPTY(type_name, func_name, internal)	\
INLINE void func_name(type_name *vc)									\
{																					\
	vc->len = 0;																\
	if (vc->s) vc->s[0] = (internal)0;													\
}

#define DECLARE_VECTOR_EMPTY2(type_name, func_name, internal)	\
INLINE void func_name(type_name *vc)									\
{																					\
	vc->len = 0;																\
}

#define DECLARE_VECTOR_READYPLUS(type_name, func_name)	\
INLINE int func_name(type_name *vc, int len)		\
{																			\
	return type_name ##_ready(vc, vc->len + len);			\
}

#define DECLARE_VECTOR_COPY(type_name, func_name)			\
INLINE int func_name(type_name *vc, const type_name *vs)	\
{																			\
	return type_name ##_copyb(vc, vs->s, vs->len);			\
}

#define DECLARE_VECTOR_COPYS(type_name, func_name, internal)			\
INLINE int func_name(type_name *vc, const internal *sz)					\
{																							\
	return type_name ##_copyb(vc, sz, strlen(sz) / sizeof(internal));	\
}

#define DECLARE_VECTOR_CAT(type_name, func_name)			\
INLINE int func_name(type_name *vc, const type_name *vs)	\
{																			\
	return type_name ##_catb(vc, vs->s, vs->len);			\
}

#define DECLARE_VECTOR_CATS(type_name, func_name, internal)				\
INLINE int func_name(type_name *vc, const internal *sz)					\
{																							\
	return type_name ##_catb(vc, sz, strlen(sz) / sizeof(internal));	\
}

#define DECLARE_VECTOR_APPEND(type_name, func_name, internal)	\
INLINE int func_name(type_name *vc, const internal *s)			\
{																					\
	return type_name ##_catb(vc, s, 1);									\
}

#define DECLARE_VECTOR_0(type_name, func_name, internal)	\
INLINE int func_name(type_name *vc)								\
{																			\
	return type_name ##_catb(vc, "\0", 1);						\
}

#define DECLARE_VECTOR_REF_POINT(ref_type_name, func_name, internal)					\
INLINE void func_name(ref_type_name *vc, const internal *sz, int len)	\
{																											\
	vc->s = sz;																							\
	vc->len = len;																						\
}

#endif



/*
 *   FUNCTION
 */

#ifndef _C_VECTOR_IMPL

#define DECLARE_VECTOR_INIT(type_name, func_name, internal)				\
void func_name(type_name *vc);
#define DECLARE_VECTOR_INIT2(type_name, func_name, internal)			\
void func_name(type_name *vc);
#define DECLARE_VECTOR_READY(type_name, func_name, internal)			\
int func_name(type_name *vc, int len);
#define DECLARE_VECTOR_EMPTY3(type_name, func_name, internal, free_func_name)	\
void func_name(type_name *vc);
#define DECLARE_VECTOR_FREE(type_name, func_name, internal)				\
int func_name(type_name *vc);
#define DECLARE_VECTOR_FREE2(type_name, func_name, internal)			\
int func_name(type_name *vc);
#define DECLARE_VECTOR_FREE3(type_name, func_name, internal, free_func_name)			\
int func_name(type_name *vc);
#define DECLARE_VECTOR_COPYB(type_name, func_name, internal)			\
int func_name(type_name *vc, const internal *src, int n);
#define DECLARE_VECTOR_COPYB2(type_name, func_name, internal)			\
int func_name(type_name *vc, const internal *src, int n);
#define DECLARE_VECTOR_CATB(type_name, func_name, internal)				\
int func_name(type_name *vc, const internal *src, int n);
#define DECLARE_VECTOR_CATB2(type_name, func_name, internal)			\
int func_name(type_name *vc, const internal *src, int n);
#define DECLARE_VECTOR_COPY_FORMAT(type_name, func_name, internal)	\
int func_name(type_name *vc, const char *fmt, ...);
#define DECLARE_VECTOR_CAT_FORMAT(type_name, func_name, internal)		\
int func_name(type_name *vc, const char *fmt, ...);
#define DECLARE_VECTOR_CAT_ULONG(type_name, func_name, internal)		\
int func_name(type_name *vc, unsigned long u, int len); 
#define DECLARE_VECTOR_CAT_LONG(type_name, func_name, internal)		\
int func_name(type_name *vc, long l, int len); 
#define DECLARE_VECTOR_DUMP(type_name, func_name)							\
void func_name(type_name *vc);
#define DECLARE_VECTOR_INFO(type_name, func_name)							\
void func_name(type_name *vc);

#ifdef _FOR_NONSTD_COMPILER_
#define DECLARE_VECTOR_DATA(type_name, func_name, internal)				\
internal * func_name(type_name *vc);
#define DECLARE_VECTOR_EMPTY(type_name, func_name, internal)			\
void func_name(type_name *vc);
#define DECLARE_VECTOR_EMPTY2(type_name, func_name, internal)			\
void func_name(type_name *vc);
#define DECLARE_VECTOR_EMPTY3(type_name, func_name, internal)			\
void func_name(type_name *vc);
#define DECLARE_VECTOR_READYPLUS(type_name, func_name)					\
int func_name(type_name *vc, int len);
#define DECLARE_VECTOR_COPY(type_name, func_name)							\
int func_name(type_name *vc, const type_name *vs);
#define DECLARE_VECTOR_COPYS(type_name, func_name, internal)			\
int func_name(type_name *vc, const internal *sz);
#define DECLARE_VECTOR_CAT(type_name, func_name)							\
int func_name(type_name *vc, const type_name *vs);
#define DECLARE_VECTOR_CATS(type_name, func_name, internal)				\
int func_name(type_name *vc, const internal *sz);
#define DECLARE_VECTOR_APPEND(type_name, func_name, internal)			\
int func_name(type_name *vc, const internal *s);
#define DECLARE_VECTOR_0(type_name, func_name, internal)					\
int func_name(type_name *vc);
#define DECLARE_VECTOR_REF_POINT(ref_type_name, func_name, internal)	\
void func_name(ref_type_name *vc, const internal *sz, int len);
#endif

#else /* _C_VECTOR_IMPL */

	/* vector_init() function */
#ifdef USE_INTERNAL_BUFFER
#define DECLARE_VECTOR_INIT(type_name, func_name, internal)	\
void func_name(type_name *vc)											\
{																				\
	vc->s = vc->sta;														\
	vc->dyn = (internal *)0;											\
	vc->error = 0;															\
	vc->len = 0;															\
	vc->capacity = INIT_BUFFER;										\
	vc->valid = 0;															\
	vc->sta[0] = (internal)0;											\
}
#else
#define DECLARE_VECTOR_INIT(type_name, func_name, internal)	\
void func_name(type_name *vc)											\
{																				\
	vc->s = (internal *)0;												\
	vc->dyn = (internal *)0;											\
	vc->error = 0;															\
	vc->len = 0;															\
	vc->capacity = 0;														\
}
#endif

#ifdef USE_INTERNAL_BUFFER
#define DECLARE_VECTOR_INIT2(type_name, func_name, internal)	\
void func_name(type_name *vc)												\
{																					\
	vc->s = vc->sta;															\
	vc->dyn = (internal *)0;												\
	vc->error = 0;																\
	vc->len = 0;																\
	vc->capacity = INIT_BUFFER;											\
	vc->valid = 0;																\
}
#else
#define DECLARE_VECTOR_INIT2(type_name, func_name, internal)	\
void func_name(type_name *vc)												\
{																					\
	vc->s = (internal *)0;													\
	vc->dyn = (internal *)0;												\
	vc->error = 0;																\
	vc->len = 0;																\
	vc->capacity = 0;															\
}
#endif

	/* vector_ready() function */
#ifdef USE_INTERNAL_BUFFER
#define DECLARE_VECTOR_READY(type_name, func_name, internal)									\
int func_name(type_name *vc, int len)														\
{																													\
	if (!vc->valid) {																							\
		if (len >= INIT_BUFFER) {																			\
			register int wanted = len + (len >> 3) + 30;												\
			register internal *tmp;																			\
																													\
			if (!(tmp = (internal *)_malloc(sizeof(internal) * (wanted + 1)))) {				\
				vc->error = 1;																					\
				return 0;																						\
			}																										\
																													\
			memcpy((void *)tmp, (void *)vc->sta, sizeof(internal) * vc->len);					\
																													\
			vc->s = vc->dyn = tmp;																			\
			vc->capacity = wanted;																			\
			vc->valid = 1;																						\
		}																											\
		else																										\
			return 1;																							\
	}																												\
																													\
	if (!vc->dyn || vc->capacity < len) {																\
		register int wanted = len + (len >> 3) + 30;													\
		register internal *tmp;																				\
																													\
		if (!(tmp = (internal *)_realloc(vc->dyn, sizeof(internal) * (wanted + 1)))) {	\
			vc->error = 1;																						\
			return 0;																							\
		}																											\
																													\
		vc->s = vc->dyn = tmp;																				\
		vc->capacity = wanted;																				\
	}																												\
	return 1;																									\
}
#else
#define DECLARE_VECTOR_READY(type_name, func_name, internal)									\
int func_name(type_name *vc, int len)														\
{																													\
	if (!vc->dyn || vc->capacity < len) {																\
		register int wanted = len + (len >> 3) + 30;													\
		register internal *tmp;																				\
																													\
		if (!(tmp = (internal *)_realloc(vc->dyn, sizeof(internal) * (wanted + 1)))) {	\
			vc->error = 1;																						\
			return 0;																							\
		}																											\
																													\
		vc->s = vc->dyn = tmp;																				\
		vc->capacity = wanted;																				\
	}																												\
	return 1;																									\
}
#endif

#define DECLARE_VECTOR_EMPTY3(type_name, func_name, internal, free_func_name)		\
void func_name(type_name *vc)									\
{																					\
	int i;																			\
	if (vc->s) {                                \
	for (i = 0; i < vc->len; ++i)													\
		free_func_name(vc->s + i);													\
	}                                           \
	vc->len = 0;																\
}

	/* vector_free() function */
#define DECLARE_VECTOR_FREE(type_name, func_name, internal)	\
void func_name(type_name *vc)											\
{																				\
	if (vc->dyn)															\
		_free(vc->dyn);													\
																				\
	vc->s = vc->dyn = (internal *)0;									\
	vc->error = 0;															\
	vc->len = 0;															\
	vc->capacity = 0;														\
}

#define DECLARE_VECTOR_FREE2(type_name, func_name, internal)	\
void func_name(type_name *vc)												\
{																					\
	if (vc->dyn)																\
		_free(vc->dyn);														\
																					\
	vc->s = vc->dyn = (internal *)0;										\
	vc->error = 0;																\
	vc->len = 0;																\
	vc->capacity = 0;															\
}

#define DECLARE_VECTOR_FREE3(type_name, func_name, internal, free_func_name)	\
void func_name(type_name *vc)												\
{																					\
	int i;																		\
	for (i = 0; i < vc->len; ++i)													\
		free_func_name(vc->s + i);												\
	if (vc->dyn)																\
		_free(vc->dyn);														\
																					\
	vc->s = vc->dyn = (internal *)0;										\
	vc->error = 0;																\
	vc->len = 0;																\
	vc->capacity = 0;															\
}

	/* vector_copyb() function */
#define DECLARE_VECTOR_COPYB(type_name, func_name, internal)		\
int func_name(type_name *vc, const internal *src, int n)	\
{																						\
	if (vc->error)																	\
		return 0;																	\
																						\
	if (!type_name ##_ready(vc, n))											\
		return 0;																	\
																						\
	memcpy((void *)vc->s, (void *)src, sizeof(internal) * n);		\
	vc->len = n;																	\
	vc->s[n] = (internal)0;														\
	return 1;																		\
}

#define DECLARE_VECTOR_COPYB2(type_name, func_name, internal)		\
int func_name(type_name *vc, const internal *src, int n)	\
{																						\
	if (vc->error)																	\
		return 0;																	\
																						\
	if (!type_name ##_ready(vc, n))											\
		return 0;																	\
																						\
	memcpy((void *)vc->s, (void *)src, sizeof(internal) * n);		\
	vc->len = n;																	\
	return 1;																		\
}

	/* vector_catb() function */
#define DECLARE_VECTOR_CATB(type_name, func_name, internal)			\
int func_name(type_name *vc, const internal *src, int n)	\
{																						\
	if (vc->error)																	\
		return 0;																	\
																						\
	if (!type_name ##_ready(vc, vc->len + n))								\
		return 0;																	\
																						\
	memcpy((void *)&vc->s[vc->len],											\
	       (void *)src, sizeof(internal) * n);							\
	vc->len += n;																	\
	vc->s[vc->len] = (internal)0;												\
	return 1;																		\
}

#define DECLARE_VECTOR_CATB2(type_name, func_name, internal)		\
int func_name(type_name *vc, const internal *src, int n)	\
{																						\
	if (vc->error)																	\
		return 0;																	\
																						\
	if (!type_name ##_ready(vc, vc->len + n))								\
		return 0;																	\
																						\
	memcpy((void *)&vc->s[vc->len],											\
	       (void *)src, sizeof(internal) * n);							\
	vc->len += n;																	\
	return 1;																		\
}

	/* vector_copyf() function */
#define DECLARE_VECTOR_COPY_FORMAT(type_name, func_name, internal)	\
int func_name(type_name *vc, const char *fmt, ...)							\
{																							\
	int done, n, i;																	\
	va_list args;																		\
																							\
	if (vc->error)																		\
		return 0;																		\
																							\
	va_start(args, fmt);																\
	for (done = 0, i = 1; !done; ++i) {											\
		if (!type_name ##_readyplus(vc, FMT_BUFFER * i)) {					\
			va_end(args);																\
			return 0;																	\
		}																					\
		n = vsnprintf(vc->s, vc->capacity - vc->len, fmt, args);			\
		done = n >= 0 && n < vc->capacity - vc->len;							\
	}																						\
	vc->len = n;																		\
	va_end(args);																		\
	return 1;																			\
}

	/* vector_catf() function */
#define DECLARE_VECTOR_CAT_FORMAT(type_name, func_name, internal)				\
int func_name(type_name *vc, const char *fmt, ...)									\
{																									\
	int done, n, i;																			\
	va_list args;																				\
																									\
	if (vc->error)																				\
		return 0;																				\
																									\
	va_start(args, fmt);																		\
	for (done = 0, i = 1; !done; ++i) {													\
		if (!type_name ##_readyplus(vc, FMT_BUFFER * i)) {							\
			va_end(args);																		\
			return 0;																			\
		}																							\
		n = vsnprintf(vc->s + vc->len, vc->capacity - vc->len, fmt, args);	\
		done = n >= 0 && n < vc->capacity - vc->len;									\
	}																								\
	vc->len += n;																				\
	va_end(args);																				\
	return 1;																					\
}

	/* vector_catulong() function */
#define DECLARE_VECTOR_CAT_ULONG(type_name, func_name, internal)	\
int func_name(type_name *vc, unsigned long u, int len)				\
{																						\
	int real;															\
	unsigned long q;																\
	char *s;																			\
																						\
	real = 1;																		\
	q = u;																			\
	while (q > 9) {																\
		++real;																		\
		q /= 10;																		\
	}																					\
	if (real < len)																\
		real = len;																	\
																						\
	if (!type_name ##_readyplus(vc, real))									\
		return 0;																	\
	s = vc->s + vc->len;															\
	vc->len += real;																\
	while (real) {																	\
		s[--real] = '0' + (char)(u % 10);											\
		u /= 10;																		\
	}																					\
	vc->s[vc->len] = (internal)0;												\
	return 1;																		\
}

	/* vector_catlong() function */
#define DECLARE_VECTOR_CAT_LONG(type_name, func_name, internal)	\
int func_name(type_name *vc, long l, int len)							\
{																						\
	if (l < 0) {																	\
		if (!type_name ##_append(vc, "-"))									\
			return 0;																\
		l = -l;																		\
	}																					\
	return type_name ##_catulong(vc, l, len);								\
}

	/* vector_dump() function */
#ifdef USE_INTERNAL_BUFFER
#define DECLARE_VECTOR_DUMP(type_name, func_name)						\
void func_name(type_name *vc)													\
{																						\
	if (type_name ##_data(vc) != vc->s)										\
		printf("!!! Assertion failure !!!\n");								\
	printf("DUMP (%s) %4d of %4d, [%s] %s %p\n",							\
	        vc->error ? "UNSAFE" : "SAFE", vc->len, vc->capacity,	\
	        type_name ##_data(vc),											\
	        vc->valid ? "" : "(INTERNAL)", vc);							\
}
#else
#define DECLARE_VECTOR_DUMP(type_name, func_name)						\
void func_name(type_name *vc)													\
{																						\
	if (type_name ##_data(vc) != vc->s)										\
		printf("!!! Assertion failure !!!\n");								\
	printf("DUMP (%s) %4d of %4d, [%s] %p\n",								\
	        vc->error ? "UNSAFE" : "SAFE", vc->len, vc->capacity,	\
	        vc->dyn, vc);														\
}
#endif

	/* vector_info() function */
#define DECLARE_VECTOR_INFO(type_name, func_name)								\
void func_name(type_name *vc)															\
{																								\
	if (type_name ##_data(vc) != vc->s)												\
		printf("!!! Assertion failure !!!\n");										\
	printf("INFO (%s) %4d of %4d, %p\n", vc->error ? "UNSAFE" : "SAFE",	\
	        vc->len, vc->capacity, vc);												\
}

#ifdef _FOR_NONSTD_COMPILER_
#ifdef USE_INTERNAL_BUFFER
#define DECLARE_VECTOR_DATA(type_name, func_name, internal)	\
internal * func_name(type_name *vc)									\
{																				\
	return vc->valid ? vc->dyn : vc->sta;							\
}
#else
#define DECLARE_VECTOR_DATA(type_name, func_name, internal)	\
internal * func_name(type_name *vc)									\
{																				\
	return vc->dyn;														\
}
#endif

#define DECLARE_VECTOR_EMPTY(type_name, func_name, internal)	\
void func_name(type_name *vc)												\
{																					\
	vc->len = 0;																\
	if (vc->s) vc->s[0] = (internal)0;													\
}

#define DECLARE_VECTOR_EMPTY2(type_name, func_name, internal)	\
void func_name(type_name *vc)												\
{																					\
	vc->len = 0;																\
}

#define DECLARE_VECTOR_READYPLUS(type_name, func_name)	\
int func_name(type_name *vc, int len)				\
{																			\
	return type_name ##_ready(vc, vc->len + len);			\
}

#define DECLARE_VECTOR_COPY(type_name, func_name)			\
int func_name(type_name *vc, const type_name *vs)			\
{																			\
	return type_name ##_copyb(vc, vs->s, vs->len);			\
}

#define DECLARE_VECTOR_COPYS(type_name, func_name, internal)			\
int func_name(type_name *vc, const internal *sz)							\
{																							\
	return type_name ##_copyb(vc, sz, strlen(sz) / sizeof(internal));	\
}

#define DECLARE_VECTOR_CAT(type_name, func_name)			\
int func_name(type_name *vc, const type_name *vs)			\
{																			\
	return type_name ##_catb(vc, vs->s, vs->len);			\
}

#define DECLARE_VECTOR_CATS(type_name, func_name, internal)				\
int func_name(type_name *vc, const internal *sz)							\
{																							\
	return type_name ##_catb(vc, sz, strlen(sz) / sizeof(internal));	\
}

#define DECLARE_VECTOR_APPEND(type_name, func_name, internal)	\
int func_name(type_name *vc, const internal *s)						\
{																					\
	return type_name ##_catb(vc, s, 1);									\
}

#define DECLARE_VECTOR_0(type_name, func_name, internal)	\
int func_name(type_name *vc)										\
{																			\
	return type_name ##_catb(vc, "\0", 1);						\
}

#define DECLARE_VECTOR_REF_POINT(ref_type_name, func_name, internal)		\
void func_name(ref_type_name *vc, const internal *sz, int len)	\
{																								\
	vc->s = sz;																				\
	vc->len = len;																			\
}
#endif

#endif /* _C_VECTOR_IMPL */

/* vim:set ts=3 sw=3: */
