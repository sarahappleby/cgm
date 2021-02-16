#!/bin/bash

line=$1
pipeline_path=/disk01/sapple/cgm/absorption/cos_comparison/cos_samples/
model=m50n512
wind=s50nofb
snap=151
survey=dwarfs
background=uvb_fg20
output_file=/disk01/sapple/cgm/absorption/cos_comparison/cos_samples/$model/cos_$survey/$wind/$background/output/


for ii in {0..38}
do
   echo Submitting job $ii
   job=$ii'_'$line'.txt'
   python $pipeline_path/pipeline.py $model $snap $wind $survey $background $ii $line > $output_file/cos_halo_$job
   echo Finished job $ii
done
