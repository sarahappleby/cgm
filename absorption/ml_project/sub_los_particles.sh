#!/bin/bash

pipeline_path=/disk01/sapple/cgm/absorption/ml_project/
model=m100n1024
wind=s50
snap=151
output_path=/disk01/sapple/cgm/absorption/ml_project/output/

for ii in {0..215}
do
   echo Submitting job $ii
   job='gal_'$ii'_.txt'
   python $pipeline_path/select_los_particles.py $model $wind $snap $ii > $output_path/$job
   echo Finished job $ii
done
