#!/bin/bash
machine=$1

for d in {livingroom1,office1,stonewall,copyroom}
do
  python remote_isogmm.py --config ${d}.yaml --machine ${machine} --bandwidth 0.02
  python remote_isogmm.py --config ${d}.yaml --machine ${machine} --bandwidth 0.03
  python remote_isogmm.py --config ${d}.yaml --machine ${machine} --bandwidth 0.04
  python remote_isogmm.py --config ${d}.yaml --machine ${machine} --bandwidth 0.05
done
