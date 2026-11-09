#!/bin/bash
# Graded study host bootstrap: LeWM bootstrap, then the three AP datasets in parallel. Caches come via compute.py fetch.
source /workspace/LeWM_Experiment1/src/graded/host_env.sh
L=$LEWM_RUN_DIR/logs; mkdir -p $L
cd $SRC
$PY -m qd_adapters.bootstrap > $L/bootstrap.log 2>&1; echo "bootstrap exit $? $(date -u +%H:%M:%S)" > $L/setup.status
for t in tworoom pusht reacher; do
  nohup $PY -m qd_adapters.cli download $t > $L/download-$t.log 2>&1 &
done
wait
echo "downloads done $(date -u +%H:%M:%S)" >> $L/setup.status
df -h /workspace | tail -1 >> $L/setup.status
