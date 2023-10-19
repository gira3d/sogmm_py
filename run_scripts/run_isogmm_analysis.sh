#!/bin/bash
machine=$1

for d in {stonewall,copyroom}
do
  python isogmm_analysis.py --config ${d}.yaml --machine ${machine} --voxel_size 0.01
done
