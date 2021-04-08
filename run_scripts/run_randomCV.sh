#!/bin/csh
#PBS -l nodes=1:ppn=15
#PBS -W x=NACCESSPOLICY:SINGLEJOB
#PBS -l walltime=8:00:00
#PBS -m ae
#PBS -M jkcm@uw.edu
#PBS -N randomCV_10

cd /home/disk/p/jkcm/Code/classified_cset/logs
conda activate classified-cset
#which python >>& divergence_extract.log
python ../random_grid_search.py >& randomCV_10.log
exit 0

