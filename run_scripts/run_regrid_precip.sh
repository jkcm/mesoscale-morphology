#!/bin/csh
#PBS -W x=NACCESSPOLICY:SINGLEJOB
#PBS -l walltime=20:00:00
#PBS -m ae
#PBS -M jkcm@uw.edu
#PBS -N regrid_precip_2015

cd /home/disk/p/jkcm/Code/classified_cset/logs
conda activate classified-cset
#which python >>& divergence_extract.log
echo "restarting" >>& test.log
python ../regrid_rain_rate.py >>& regrid_rain_2015_reboot.log
exit 0

