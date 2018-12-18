#!/bin/bash

line=$1
pipeline_path=/home/sapple/cgm/cos_samples/
model=m50n512
wind=fh_qr
snap=127
output_file=/home/sapple/cgm/cos_samples/pygad/output/


for ii in {0..44}
do
   echo Submitting job $ii
   python $pipeline_path/pipeline.py $model $snap $wind $ii $line > $output_file/cos_halo_$ii_$line.txt
   echo Finished job $ii
done
