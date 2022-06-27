#!/bin/bash

model=m100n1024
wind=s50
declare -a snaps=("125" "105")

for snap in "${snaps[@]}"
do
	echo Starting snap $snap
	python equivalent_width.py $model $wind $snap
done
