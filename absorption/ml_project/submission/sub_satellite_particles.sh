#!/bin/bash

pipeline_path=/disk04/sapple/cgm/absorption/ml_project/

for ii in {0..19766}
do
   echo Submitting job $ii
   python $pipeline_path/select_satellite_particles.py $ii
   echo Finished job $ii
done

