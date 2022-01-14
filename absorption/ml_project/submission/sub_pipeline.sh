#!/bin/bash

line=$1
pipeline_path=/disk04/sapple/cgm/absorption/ml_project/
model=m100n1024
wind=s50
snap=151
output_file=/disk04/sapple/cgm/absorption/ml_project/output/


for ii in {0..18}
do
   echo Submitting job $ii
   job=$ii'_'$line'.txt'
   python $pipeline_path/pipeline.py $model $snap $wind $ii $line > $output_file/$job
   echo Finished job $ii
done
