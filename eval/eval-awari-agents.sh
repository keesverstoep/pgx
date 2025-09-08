#!/bin/bash

# This script assumes it is run from the pgx-main directory.
# For running under a cluster environment, add the suitable SBATCH directives,
# Python and GPU environment initialisation.

export PYTHONPATH=`pwd`:`pwd`/eval:`pwd`/examples/alphazero

CHECKPOINT_DIR=`pwd`/examples/alphazero/checkpoints

export ENV_ID=awari
export NUM_ACTIONS=7
export NUM_CHANNELS=128
export NUM_LAYERS=6
export NUM_SIMULATIONS=800
export RAND_SEED=$RANDOM

if [ "${ORACLE_MISTAKE_CHANCE}" = "" ]; then
    export ORACLE_MISTAKE_CHANCE=0.20
fi
ORACLE_MISTAKE_CHANCE_LABEL=`echo ${ORACLE_MISTAKE_CHANCE} | sed 's/\.//'`

if [ "${ORACLE_MISTAKE_MAX}" = "" ]; then
    export ORACLE_MISTAKE_MAX=2
fi
if [ "${MODEL_VERSION}" = "" ]; then
    MODEL_VERSION=000400
fi

MODEL_BASE=`(cd examples/alphazero/checkpoints; ls -t | grep $ENV_ID | head -1)`
if [ "X${MODEL_BASE}" = X ]; then
    echo "No checkpoints for ${ENV_ID}" >&2
    exit 1
fi

model=${MODEL_VERSION}
export CHECKPOINT=${CHECKPOINT_DIR}/${MODEL_BASE}/${model}.ckpt

echo Using checkpoint ${CHECKPOINT}

python3 -u eval/eval-agents.py env_id=$ENV_ID > eval-${MODEL_BASE}-v${model}-nsim${NUM_SIMULATIONS}-AOP${ORACLE_MISTAKE_CHANCE_LABEL}-${ORACLE_MISTAKE_MAX}.out
