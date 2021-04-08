#!/bin/csh
#PBS -W x=NACCESSPOLICY:SINGLEJOB
#PBS -l walltime=20:00:00
#PBS -m ae
#PBS -M jkcm@uw.edu
#PBS -N pdp_calculator_1

cd /home/disk/p/jkcm/Code/classified_cset/logs
conda activate classified-cset
#which python >>& divergence_extract.log
python ../random_forest_pdp.py >>& pdp_extract_1.log
exit 0

