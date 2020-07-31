#!/bin/bash

pipeline_path=/home/sapple/cgm/cos_samples/
survey=halos
model=m50n512
wind=s50j7k
output_file=/home/sapple/cgm/cos_samples/$model/cos_$survey/$wind/output/


for ii in {0..219}
do
   echo Submitting job $ii
   job='gal_'$ii'_.txt'
   python $pipeline_path/select_los_particles.py $model $wind $ii $survey > $output_file/cos_halo_$job
   echo Finished job $ii
done
