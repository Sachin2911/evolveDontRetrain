#!/bin/bash
# TD-MPC2 + Meta-World basketball on the hazard-design host (the install that worked on 8 Nov, see residual_tdmpc2/README.md).
set -x
export PIP_DISABLE_PIP_VERSION_CHECK=1
PY=/venv/main/bin/python; PIP="$PY -m pip"
mkdir -p /workspace/tdmpc2/ckpt && cd /workspace/tdmpc2
[ -d tdmpc2 ] || git clone --depth 50 https://github.com/nicklashansen/tdmpc2.git tdmpc2
(cd tdmpc2 && git rev-parse HEAD > ../tdmpc2_commit.txt)
$PIP install hydra-core==1.3.8 omegaconf==2.3.1 termcolor cma imageio-ffmpeg matplotlib orjson pyvers 2>&1 | tail -1
$PIP install --no-deps "tensordict==0.6.2" 2>&1 | tail -1
$PIP install --no-deps gym==0.26.2 2>&1 | tail -1
$PIP install --force-reinstall --no-deps "git+https://github.com/Farama-Foundation/Metaworld.git@c822f28f582ba1ad49eb5dcf61016566f28003ba" 2>&1 | tail -1
$PY -c "import metaworld; from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE as E; print('metaworld ok', 'basketball-v2-goal-observable' in E)"
cd ckpt; [ -f mw-basketball-1.pt ] || curl -sL -o mw-basketball-1.pt https://huggingface.co/nicklashansen/tdmpc2/resolve/main/metaworld/mw-basketball-1.pt; ls -l; sha256sum *.pt
echo TD_SETUP_DONE
