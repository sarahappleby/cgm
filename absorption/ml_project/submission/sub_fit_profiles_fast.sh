#!/bin/bash

pipeline_path=/disk04/sapple/cgm/absorption/ml_project/make_spectra/
model=m100n1024
wind=s50
snap=125

total_jobs=50000
max_jobs=20
cur_jobs=0
ii=0

for ((i=0; i<$total_jobs; i++)); 
do
	# If true, wait until the next background job finishes to continue.
  	((cur_jobs >= max_jobs)) && wait -n
  	# Increment the current number of jobs running.
	echo Submitting job $ii
	python $pipeline_path/fit_profiles.py $model $wind $snap $ii & ((++cur_jobs))
	((++ii))
done
wait
