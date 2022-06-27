#!/bin/bash

declare -a lines=("MgII2796" "CII1334" "SiIII1206" "CIV1548" "OVI1031")

for l in "${lines[@]}"
do
	echo Starting line $l
	bash sub_pipeline.sh $l
done
