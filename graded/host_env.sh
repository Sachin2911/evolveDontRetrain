# Source on the rented host before any graded-study command.
export WS=/workspace/LeWM_Experiment1
export SRC=$WS/src
export LEWM_CONFIG=$SRC/qd_adapters/config.json
export LEWM_PROTOCOL=$SRC/qd_adapters/protocol.md
export LEWM_RUN_DIR=$WS/runs/graded
export LEWM_SEEDS='[0]'
export PYTHONPATH=$SRC:$SRC/scripts:$SRC/computeConfigs/scripts${PYTHONPATH:+:$PYTHONPATH}
export PY=/venv/main/bin/python
export GQ=$WS/runs/graded/queue
mkdir -p $LEWM_RUN_DIR/metrics $LEWM_RUN_DIR/logs $LEWM_RUN_DIR/artifacts $LEWM_RUN_DIR/.control $GQ
