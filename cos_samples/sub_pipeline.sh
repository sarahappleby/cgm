#!/bin/bash

line=$1
pipeline_path=/home/sapple/cgm/cos_samples/
model=m25n512
wind=s50
snap=137
survey=halos
background=uvb_fg20
output_file=/home/sapple/cgm/cos_samples/$model/cos_$survey/$wind/$background/output/


for ii in {0..43}
do
   echo Submitting job $ii
   job=$ii'_'$line'.txt'
   python $pipeline_path/pipeline.py $model $snap $wind $survey $background $ii $line > $output_file/cos_halo_$job
   echo Finished job $ii
done
