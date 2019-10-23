#!/bin/bash

output_file=/home/sapple/cgm/cos_samples/pygad/output/

declare -a arr=("H1215" "SiII1260" "CII1335", "SiIII1206", "SiIV1393", "CIII977", "OVI1031")
for i in "${arr[@]}"
do
	echo Generating sample for "$i"
	bash sub_pipeline.sh "$i" > $output_file/bash_"$i".txt
done
