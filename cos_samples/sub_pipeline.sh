#!/bin/bash

line=$1
pipeline_path=/home/sapple/cgm/cos_samples/
model=m50n512
wind=s50j7k
snap=151
survey=dwarfs
output_file=/home/sapple/cgm/cos_samples/$model/cos_$survey/$wind/output/


for ii in {0..37}
do
   echo Submitting job $ii
   job=$ii'_'$line'.txt'
   python $pipeline_path/pipeline.py $model $snap $wind $survey $ii $line > $output_file/cos_halo_$job
   echo Finished job $ii
done
