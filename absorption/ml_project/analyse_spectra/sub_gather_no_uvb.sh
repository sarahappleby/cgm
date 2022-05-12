#!/bin/bash

declare -a fr200=("0.25" "0.5" "0.75" "1.0" "1.25")
declare -a lines=("MgII2796" "CII1334" "SiIII1206" "CIV1548" "OVI1031")

model=m100n1024
wind=s50
snap=151
for f in "${fr200[@]}"
do
	for l in "${lines[@]}"
	do

		python gather_line_results_no_uvb.py $model $wind $snap $f $l 
	done
done