#!/bin/csh
#PBS -l nodes=1:ppn=12
#PBS -W x=NACCESSPOLICY:SINGLEJOB
#PBS -l walltime=20:00:00
#PBS -m ae
#PBS -M jkcm@uw.edu
#PBS -N regrid_MERRA_2015

cd /home/disk/p/jkcm/Code/classified_cset/logs
conda activate classified-cset
python ../process_gridded_to_dataframe.py SEP 2015 >& divergence_extract_2015_newmerra.log
exit 0
