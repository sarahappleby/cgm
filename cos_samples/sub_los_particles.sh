#!/bin/bash

pipeline_path=/home/sapple/cgm/cos_samples/
survey=dwarfs
output_file=/home/sapple/cgm/cos_samples/cos_$survey/output/


for ii in {2..190}
do
   echo Submitting job $ii
   job='gal_'$ii'_.txt'
   python $pipeline_path/select_los_particles.py $ii $survey > $output_file/cos_halo_$job
   echo Finished job $ii
done
