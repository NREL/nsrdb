#ifndef H4TOH5APICOMPITABLE_H
#define H4TOH5APICOMPITABLE_H

#if (H5_VERS_MAJOR == 1 && H5_VERS_MINOR == 6) || defined H5_USE_16_API

 	#define H5GOPEN(fd, name) H5Gopen(fd, name)
    #define H5DOPEN(fd, name) H5Dopen(fd, name)
    #define H5DCREATE(fd, name, tid, sid, did) H5Dcreate(fd, name, tid, sid, did)
    #define H5ACREATE(fd, name, tid, sid, aid) H5Acreate(fd, name, tid, sid, aid)
    #define H5GCREATE(fd, name, hint) H5Gcreate(fd, name, hint)
    #define H5TCOMMIT(fd, name, tid) H5Tcommit(fd, name, tid)
    #define H5TARRAY_CREATE(tid, rank, dims, perm) H5Tarray_create(tid, rank, dims, perm)

#else

	#define H5GOPEN(fd, name) H5Gopen2(fd, name, H5P_DEFAULT)
	#define H5DOPEN(fd, name) H5Dopen2(fd, name, H5P_DEFAULT)
	#define H5DCREATE(fd, name, tid, sid, did) H5Dcreate2(fd, name, tid, sid, H5P_DEFAULT, did, H5P_DEFAULT)
	#define H5ACREATE(fd, name, tid, sid, aid) H5Acreate2(fd, name, tid, sid, aid, H5P_DEFAULT)
	#define H5GCREATE(fd, name, hint) H5Gcreate2(fd, name, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)
	#define H5TCOMMIT(fd, name, tid) H5Tcommit2(fd, name, tid, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT)
	#define H5TARRAY_CREATE(tid, rank, dims, perm) H5Tarray_create2(tid, rank, dims)
#endif

#endif

