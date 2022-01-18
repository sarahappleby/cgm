#!/bin/bash

model=m100n1024
snap=151
wind=s50
pipeline_path=/disk04/sapple/cgm/absorption/ml_project/analyse_spectra/

declare -a lines=("H1215" "MgII2796" "SiIII1206" "CIV1548" "OVI1031" "NeVIII770")
declare -a fr200=("0.25" "0.5" "0.75" "1.0" "1.25")

for l in "${lines[@]}"
do
	for f in "${fr200[@]}"
	do
		python $pipeline_path/gather_results.py $model $snap $wind $f $l
	done
done
