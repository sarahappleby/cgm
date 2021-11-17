#!/bin/bash

pipeline_path=/disk04/sapple/cgm/absorption/ml_project/
model=m100n1024
wind=s50
snap=151
output_file=/disk04/sapple/cgm/absorption/ml_project/output/


for ii in {0..10000}
do
   echo Submitting job $ii
   job=$ii'_'$line'.txt'
   python $pipeline_path/fit_profiles.py $ii > $output_file/$job
   echo Finished job $ii
done

