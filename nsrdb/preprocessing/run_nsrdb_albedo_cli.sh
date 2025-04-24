#!/bin/bash

# Process albedo for a year or fraction of a year. Existing albedo files are not
# re-run. There is one required parameter and two optional parameters.
#
#   $1 Year - the four digit year to process
#   $2 start day (optional) - the first day of year to process, in DDD format
#   $3 end day (optional) - the last day of year to process, in DDD format

if [ -z $1 ]; then
    echo Please set a year as the first argument
    exit
fi
YEAR=$1

if [ `expr $YEAR % 400` -eq 0 ]; then
    ISLEAP=true
elif [ `expr $YEAR % 100` -eq 0 ]; then
    ISLEAP=true
elif [ `expr $YEAR % 4` -eq 0 ]; then
    ISLEAP=true
else
    ISLEAP=false
fi


if [ -z $2 ]; then
    STARTDAY=001
else
    STARTDAY=$2
fi

if [ -z $3 ]; then
    if [ $ISLEAP ]; then
        ENDDAY=365
    else
        ENDDAY=364
    fi
else
    ENDDAY=$3
fi


# --- Setup paths, make dirs if needed
if [ $YEAR -gt 2021 ]; then
    echo Setting modis source to 2021
    MPATH=/kfs2/projects/pxs/ancillary/albedo/modis/v6.1/source_2021/
elif [ $YEAR -gt 2012 ]; then
    echo Setting modis source to v6.1 for ${YEAR}
    MPATH=/kfs2/projects/pxs/ancillary/albedo/modis/v6.1/source_${YEAR}/
elif [ $YEAR -lt 2001 ]; then
    echo Setting modis source to 2001
    MPATH=/kfs2/projects/pxs/ancillary/albedo/modis/v6/source_2001/
else
    echo Setting modis source to v6 for ${YEAR}
    MPATH=/kfs2/projects/pxs/ancillary/albedo/modis/v6/source_${YEAR}/
fi

IPATH=/kfs2/projects/pxs/ancillary/albedo/ims/${YEAR}/

if [ ! -d "${IPATH}" ]; then
    echo Making ${IPATH}
    mkdir ${IPATH}
fi

APATH=/kfs2/projects/pxs/ancillary/albedo/nsrdb_${YEAR}/
if [ ! -d "${APATH}" ]; then
    echo Making ${APATH}
    mkdir ${APATH}
fi

LOGDIR=/kfs2/projects/pxs/ancillary/albedo/nsrdb_${YEAR}/logs/
LOGFILE=${LOGDIR}/albedo.log
if [ ! -d "${LOGDIR}" ]; then
    echo Making ${LOGDIR}
    mkdir ${LOGDIR}
fi

MEPATH=/kfs2/projects/pxs/ancillary/merra/

# --- Check for existing albedo h5 files and process if missing
START=${YEAR}${STARTDAY}
END=${YEAR}${ENDDAY}
echo Processing albedo from $START to $END

for DAY in $(seq -f "%03g"  $STARTDAY $ENDDAY)
do
    #  set -xv
    AFILE=${APATH}/nsrdb_albedo_${YEAR}_${DAY}.h5
    DATE=${YEAR}$DAY

    #echo Processing ${DATE}...
    if [ ! -f "${AFILE}" ]; then
        python -m nsrdb.albedo.cli -m ${MPATH} -i ${IPATH} -a ${APATH} -me ${MEPATH} --log-file ${LOGFILE} multiday $DATE $DATE --alloc pxs -wt 1.0 -l "--qos=normal"
    else
        echo - $AFILE exists, skipping
    fi
done

