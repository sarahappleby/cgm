#!/bin/bash

declare -a fr200=("0.25" "0.5" "0.75" "1.0" "1.25")

for f in "${fr200[@]}"
do
	python get_alignments.py $f
done
