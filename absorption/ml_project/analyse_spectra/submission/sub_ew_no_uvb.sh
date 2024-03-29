#!/bin/bash

model=m100n1024
wind=s50
snap=151
declare -a uvb=("with_uvb" "no_uvb")
declare -a minT=("6.0")

for u in "${uvb[@]}"
do
	echo Starting uvb option $u
	for mT in "${minT[@]}"
	do
		echo Starting minT $mT
		python equivalent_width_no_uvb.py $model $wind $snap $u $mT
	done
done
