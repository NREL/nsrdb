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

#ifndef H4TOH5PUBLIC_H
#define H4TOH5PUBLIC_H

#include "hdf.h"
#include "mfhdf.h"
#ifdef HAVE_LIBHDFEOS
#include "HdfEosDef.h"
#endif
#include "hfile.h"
/*#define  H5_USE_16_API 1 */
#include "hdf5.h"
#include <fcntl.h>
#include <errno.h>
#include <time.h>
#include <string.h>
#include "H4TOH5api_adpt.h"

#include "h4toh5apicompatible.h"

/* ----------------------------- Version Tags ----------------------------- */
/* Version information of the tool. It consists of:
 * Major version: Increases whenever there are addition of new features or
 *                major changes to existing feature. It also resets the Minor
 *                version and Release to zero.
 * Minor version: Increases whenever there are feature changes. This applies
 *                to minor feature changes or bug fixes that change the
 *                behavior of existing feature. It also resets the Release to
 *                zero.
 * Release:       Increases whenever officially released. This applies to
 *                external libraries (e.g. HDF5, HDF4, zlib, ...) version
 *                changes. Also bug fixes that do not change the behavior
 *                of existing features.
 */
#define H4TOH5_LIBVER_MAJOR    2
#define H4TOH5_LIBVER_MINOR    2
#define H4TOH5_LIBVER_RELEASE  2
#define H4TOH5_LIBVER_STRING   "H4toH5 converter library Version 2.2 Release 2, currently under development"
/* end of version tags */

/*. defination of public flags. */
/* File access flags. */
#define H425_CREATE  1
#define H425_OPEN    2
#define H425_CLOBBER 3

/* Attribute access flags. */
#define H425_NOATTRS     0
#define H425_ALLATTRS    1
#define H425_NONEWATTRS  2

/* VGROUP member conversion flags. */
#define H425_NOMEMBERS   0
#define H425_ALLMEMBERS  1

/* dimensional scale  conversion flags. */
#define H425_NODIMSCALE  0
#define H425_DIMSCALE    1

/* palette conversion flags. */
#define H425_NOPAL       0
#define H425_PAL         1

/* object reference appendix flags. */
#define H425_NOREF       0
#define H425_REF         1

/* Significant API changes in HDF5 1.8 ,
   to make sure the conversion tool can link with previous version
   of HDF5 library, the following macro is added KY 2007/9/27 
#if H5_VERS_MAJOR==1 && H5_VERS_MINOR<8
#define H5Gopen_abs(S,F) H5Gopen(S,F)
#define H5Gcreate_abs(S,F) H5Gcreate(S,F,0)
#else 
#define H5Gopen_abs(S,F) H5Gopen(S,F,H5P_DEFAULT)
#define H5Gcreate_abs(S,F) H5Gcreate(S,F,H5P_DEFAULT,H5P_DEFAULT,H5P_DEFAULT)
#endif

*/
__DLL425__ hid_t H4toH5open_id(char*,hid_t);
__DLL425__ hid_t H4toH5open(char*,char*,int);
__DLL425__ char* H4toH5check_object(hid_t,uint16,int32,int*);
__DLL425__ int H4toH5check_objname_in_use(hid_t,char*,const char*);
__DLL425__ int H4toH5close(hid_t);
__DLL425__ int H4toH5bas_vgroup(hid_t,int32,char*,char*,int,int);
__DLL425__ int H4toH5vgroup_attr_index(hid_t,int32,char*,char*,int);
__DLL425__ int H4toH5vgroup_attr_name(hid_t,int32,char*,char*,char*);
__DLL425__ int H4toH5adv_group(hid_t,int32,char*,char*,int);
__DLL425__ int H4toH5vdata(hid_t,int32,char*,char*,int);
__DLL425__ int H4toH5all_lone_vdata(hid_t,char*,int);
__DLL425__ int H4toH5vdata_attr_index(hid_t,int32,char*,char*,int);
__DLL425__ int H4toH5vdata_attr_name(hid_t,int32,char*,char*,char*);
__DLL425__ int H4toH5vdata_field_attr_index(hid_t,int32,char*,char*,int,int);
__DLL425__ int H4toH5vdata_field_attr_name(hid_t,int32,char*,char*,char*,char*);
__DLL425__ int H4toH5sds(hid_t,int32,char*,char*,char*,int,int);
__DLL425__ int H4toH5all_lone_sds(hid_t,char*,char*,int,int);
__DLL425__ int H4toH5sds_attr_index(hid_t,int32,char*,char*,int);
__DLL425__ int H4toH5sds_attr_name(hid_t,int32,char*,char*,char*);
__DLL425__ int H4toH5image(hid_t,int32,char*,char*,char*,char*,int,int); 
__DLL425__ int H4toH5image_attr_index(hid_t,int32,char*,char*,int);
__DLL425__ int H4toH5image_attr_name(hid_t,int32,char*,char*,char*);
__DLL425__ int H4toH5all_lone_image(hid_t,char*,char*,int,int);
__DLL425__ int H4toH5pal(hid_t,int32,char *,char*,char*,char*,int,int);
__DLL425__ int H4toH5anno_file_label(hid_t,char*,int);
__DLL425__ int H4toH5anno_file_desc(hid_t,char*,int);
__DLL425__ int H4toH5anno_file_all_labels(hid_t);
__DLL425__ int H4toH5anno_file_all_descs(hid_t);
__DLL425__ int H4toH5anno_obj_label(hid_t,char*,char*,uint16, int32,char*,int);
__DLL425__ int H4toH5anno_obj_desc(hid_t,char*,char*,uint16, int32,char*,int);
__DLL425__ int H4toH5anno_obj_all_labels(hid_t,const char*,const char*,uint16, int32);
__DLL425__ int H4toH5anno_obj_all_descs(hid_t,const char*,const char*,uint16, int32);
__DLL425__ int H4toH5all_dimscale(hid_t,int32,char*,char*,char*,
		      int,int);
__DLL425__ int H4toH5one_dimscale(hid_t,int32,char*,char*,char*,char*,int,int,int);
__DLL425__ int H4toH5glo_sds_attr(hid_t);
__DLL425__ int H4toH5glo_image_attr(hid_t);
__DLL425__ int H4toH5datatype(hid_t,const int32,hid_t*, hid_t*, size_t*, 
				   size_t*);
__DLL425__ char* H4toH5get_group_name(hid_t,int32,char*);
__DLL425__ char* H4toH5get_SDS_name(hid_t,int32,char*);
__DLL425__ char* H4toH5get_image_name(hid_t,int32,char*);
__DLL425__ char* H4toH5get_vdata_name(hid_t,int32,char*);
__DLL425__ char* H4toH5get_pal_name(hid_t,int32,char*);

__DLL425__ int H4toH5error_get(hid_t h4toh5id);
#ifdef HAVE_LIBHDFEOS
__DLL425__ int H4toH5eos_num_grid(const char *);
__DLL425__ int H4toH5eos_num_swath(const char *);
__DLL425__ int H4toH5eos_num_point(const char *);
__DLL425__ int H4toH5eos_test_emptydataset(const char *);

__DLL425__ int H4toH5eos_initialization(hid_t);
__DLL425__ int H4toH5eos_finalization(hid_t);

__DLL425__ int H4toH5eos_add_mangled_vgroupname(hid_t,const char *,const char *);
__DLL425__ int H4toH5eos_add_mangled_vdataname(hid_t,const char *,const char *);
__DLL425__ int H4toH5eos_add_mangled_sdsname(hid_t,const char *,const char *);

__DLL425__ int H4toH5sds_eosgrid_dimscale(hid_t,const char*,int32,hid_t,hobj_ref_t,int*);
__DLL425__ int H4toH5sds_eosswath_dimscale(hid_t,const char*,int32,hid_t,hobj_ref_t,int*);
__DLL425__ int H4toH5sds_noneos_fake_dimscale(hid_t,const char*,int32,hid_t,hobj_ref_t,int*);

__DLL425__ int H4toH5vdata_eosgrid_dimscale(hid_t,const char*,int32,int*);
__DLL425__ int H4toH5vdata_eosswath_dimscale(hid_t,const char*,int32,int*);
__DLL425__ int H4toH5vdata_eos_attribute(hid_t h4toh5id, const char *h5groupname, const char *vdataname, int32 vdataid, int *handled);
__DLL425__ int H4toH5vdata_noneos_fake_dimscale(hid_t,const char*,int32,int32,hid_t,hobj_ref_t,int*);
#endif
/* wrapper of H5Gcreate1 */
__DLL425__ hid_t H4toH5_H5Gcreate(hid_t h4toh5id, hid_t loc_id, const char *name, size_t size_hint);

__DLL425__ int H4toH5set_convert_flags(int eos2, int nc4hack, int nc4strict, int nc4fakedim, int verbose, int hdf5latest);
__DLL425__ int H4toH5config_use_eos2_conversion();
__DLL425__ int H4toH5config_use_netcdf4_hack();
__DLL425__ int H4toH5config_use_netcdf4_strict();
__DLL425__ int H4toH5config_use_netcdf4_fakedim();
__DLL425__ int H4toH5config_use_verbose_msg();
__DLL425__ int H4toH5config_use_debug_msg();
__DLL425__ int H4toH5config_use_latest_format();

__DLL425__ char * H4toH5correct_name_netcdf4(hid_t h4toh5id, const char *oldname);
#endif
