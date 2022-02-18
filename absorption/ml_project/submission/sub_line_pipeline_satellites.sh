#!/bin/bash

declare -a lines=("H1215" "MgII2796" "CII1334" "SiIII1206" "CIV1548" "OVI1031")

for l in "${lines[@]}"
do
	echo Starting line $l
	bash sub_pipeline_satellites.sh $l
done
