#!/bin/bash
# Graded study production launcher (on the host via graded/remote.sh <h|k> run "bash graded/run.sh ...").
#   run.sh ap <env> <cuda:0,cuda:1>      LeWM/AP workers on the production queue
#   run.sh td <0,1>                      TD-MPC2 basketball workers (GPU indices; may share a LeWM GPU)
#   run.sh natives <envs> <tag>          native runs (log_senses) on every panel case of those envs
#   run.sh search <steer|choose> <repeat> <tag> <deadline_epoch> [max_gens]
#   run.sh search2 <steer|choose> <repeat> <tag> <deadline_epoch> <max_gens> [extra]   (run 2, v2 adapters)
#   run.sh test <envs> <tag> [--only prefixes]
#   run.sh stop                          STOP every production queue
source /workspace/LeWM_Experiment1/src/graded/host_env.sh
cd $SRC
Q=$WS/runs/graded/prod; OUT=$WS/runs/graded/out; P=$SRC/graded/panels; L=$WS/runs/graded/logs
mkdir -p $Q $OUT $L
case "$1" in
  ap)
    env=$2; IFS=',' read -ra devs <<< "$3"; mkdir -p $Q/$env; rm -f $Q/$env/STOP
    for d in "${devs[@]}"; do
      tag=${d/:/}; WD=$WS/runs/graded/w/ap-$env-$tag-$(date +%s); mkdir -p $WD/metrics $WD/logs $WD/artifacts $WD/.control
      LEWM_RUN_DIR=$WD nohup $PY -m graded.ap_worker $env --device $d --queue $Q/$env > $L/prod-ap-$env-$tag.log 2>&1 &
      echo "ap worker $env $d pid $!"
    done ;;
  td)
    IFS=',' read -ra devs <<< "$2"; mkdir -p $Q/basketball; rm -f $Q/basketball/STOP
    for d in "${devs[@]}"; do
      WD=$WS/runs/graded/w/td-$d-$(date +%s%N); mkdir -p $WD
      LEWM_RUN_DIR=$WD nohup $PY -m graded.td_worker --device $d --queue $Q/basketball > $L/prod-td-$d-$(date +%s%N).log 2>&1 &
      echo "td worker gpu $d pid $!"
    done ;;
  natives)
    nohup $PY -m graded.search natives --queue $Q --panels $P --out $OUT --envs $2 --tag $3 > $L/natives-$3.log 2>&1 &
    echo "natives pid $!" ;;
  search)
    nohup $PY -m graded.search search --adapter $2 --repeat $3 --tag $4 --deadline $5 --max-gens ${6:-24} \
      --queue $Q --panels $P --out $OUT > $L/search-$2-r$3.log 2>&1 &
    echo "search $2 r$3 pid $!" ;;
  search2)  # run 2: search2 <steer|choose> <repeat> <tag> <deadline_epoch> <max_gens> [extra args]
    a=$2; r=$3; tag=$4; dl=$5; mx=$6; shift 6
    nohup $PY -m graded.search search --adapter $a --repeat $r --tag $tag --deadline $dl --max-gens $mx --min-gens 4 \
      --version 2 --variant=-v2 --fixed 1,2,1 --pop 8 "$@" \
      --queue $Q --panels $P --out $OUT > $L/search2-$a-r$r.log 2>&1 &
    echo "search2 $a r$r pid $!" ;;
  test)
    envs=$2; tag=$3; shift 3
    nohup $PY -m graded.search test --queue $Q --panels $P --out $OUT --envs $envs --tag $tag "$@" > $L/test-$tag.log 2>&1 &
    echo "test pid $!" ;;
  stop)
    for d in $Q/*/; do touch $d/STOP; done; echo stopped ;;
  *) echo "ap | td | natives | search | test | stop"; exit 1 ;;
esac
