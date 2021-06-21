#!/bin/bash

line=$1
pipeline_path=/disk01/sapple/cgm/absorption/cos_comparison/cos_samples/
model=m100n1024
wind=s50
snap=151
survey=dwarfs
background=uvb_fg20
output_dir=/disk01/sapple/cgm/absorption/cos_comparison/cos_samples/output/


for ii in {0..38}
do
   echo Submitting job $ii
   job=$ii'_'$line'.txt'
   python $pipeline_path/pipeline.py $model $snap $wind $survey $background $ii $line > $output_dir/cos_halo_$job
   echo Finished job $ii
done
