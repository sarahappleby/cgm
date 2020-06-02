#!/bin/bash

declare -a lines=("H1215" "MgII2796" "SiIII1206" "CIV1548" "OVI1031" "NeVIII770 ")

for l in "${lines[@]}"
do
	echo Starting line $l
	bash sub_pipeline.sh $l
done
