#!/bin/bash

pipeline_path=/disk04/sapple/cgm/absorption/ml_project/make_spectra/
model=m100n1024
wind=s50
snap=151

for ii in {0..10000}
do
   echo Submitting job $ii
   python $pipeline_path/fit_profiles.py $ii
   echo Finished job $ii
done

