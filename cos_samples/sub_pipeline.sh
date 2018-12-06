#!/bin/bash

pipeline_path=/home/sapple/cgm/cos_samples/
model=m50n512
wind=fh_qr
snap=151
output_file=/home/sapple/cgm/cos_samples/output/


for ii in {0..1}
do
   echo Submitting job $ii
   python $pipeline_path/pipeline.py $model $snap $wind $ii > $output_file/sample_$ii.txt
   echo Finished job $ii
done
