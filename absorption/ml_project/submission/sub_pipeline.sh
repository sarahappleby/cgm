#!/bin/bash

line=$1
pipeline_path=/disk04/sapple/cgm/absorption/ml_project/make_spectra/
model=m100n1024
wind=s50
snap=151

for ii in {0..18}
do
   echo Submitting job $ii
   python $pipeline_path/pipeline.py $model $snap $wind $ii $line
   echo Finished job $ii
done
