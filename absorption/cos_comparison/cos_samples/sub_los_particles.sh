#!/bin/bash

pipeline_path=/disk01/sapple/cgm/absorption/cos_comparison/cos_samples/
survey=dwarfs
model=m50n512
wind=s50nofb
output_file=/disk01/sapple/cgm/absorption/cos_comparison/cos_samples/$model/cos_$survey/$wind/output/


for ii in {0..189}
do
   echo Submitting job $ii
   job='gal_'$ii'_.txt'
   python $pipeline_path/select_los_particles.py $model $wind $ii $survey > $output_file/cos_halo_$job
   echo Finished job $ii
done
