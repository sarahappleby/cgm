#!/bin/bash

pipeline_path=/disk04/sapple/cgm/absorption/ml_project/make_spectra/
model=m100n1024
wind=s50
snap=137

for ii in {0..216}
do
   echo Submitting job $ii
   python $pipeline_path/select_los_particles.py $model $wind $snap $ii
   echo Finished job $ii
done
