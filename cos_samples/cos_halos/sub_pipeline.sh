#!/bin/bash

line=NeVIII770
pipeline_path=/home/sapple/cgm/cos_samples/
model=m50n512
wind=s50j7k
snap=151
output_file=/home/sapple/cgm/cos_samples/samples/output/

python_use=/home/sapple/anaconda2/bin/python

for ii in {0..43}
do
   echo Submitting job $ii
   job=$ii'_'$line'.txt'
   $python_use $pipeline_path/pipeline.py $model $snap $wind $ii $line > $output_file/cos_halo_$job
   echo Finished job $ii
done
