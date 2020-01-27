#!/bin/bash

line=$1
pipeline_path=/home/sapple/cgm/cos_samples/
model=m100n1024
wind=s50
snap=137
survey=halos
output_file=/home/sapple/cgm/cos_samples/cos_$survey/output/


for ii in {0..43}
do
   echo Submitting job $ii
   job=$ii'_'$line'.txt'
   python $pipeline_path/pipeline.py $model $snap $wind $survey $ii $line > $output_file/cos_halo_$job
   echo Finished job $ii
done
