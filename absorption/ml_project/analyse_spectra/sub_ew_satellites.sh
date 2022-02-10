#!/bin/bash

model=m100n1024
wind=s50
snap=151
declare -a log_frad=("0.0" "0.5" "1.0" "1.5" "2.0" "2.5" "3.0")

for fr in "${log_frad[@]}"
do
	echo Starting minT $mT
	python equivalent_width_satellites.py $model $wind $snap $fr
done
