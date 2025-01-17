#!/bin/bash

#############################################
# User defined parameters and env vars

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

export NEURON_CC_FLAGS="--model-type=transformer --distribution-strategy=llm-training --cache_dir=./neuron_compile_cache/"
export NEURON_FUSE_SOFTMAX=1

# Async Runtime
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3

# Limit Number of NEFFs
export NEURON_NUM_RECENT_MODELS_TO_KEEP=4

# HOST OOM
export MALLOC_ARENA_MAX=64

# TP degree
#TP_DEGREE=8
# 0: bf16; 1: mixed precision
#USE_MIX_PRECISION=0
# 0: use pure DP; 1: use ZeRO-1
#USE_ZERO_1=1
# global batch size
#GBS=32

#Tp degree
#TP_DEGREE=32
TP_DEGREE=8
# 0: bf16; 1: mixed precision
USE_MIX_PRECISION=0
# 0: use pure DP; 1: use ZeRO-1
#USE_ZERO_1=1
USE_ZERO_1=1
# global batch size
#: ${GBS:=1024}
GBS=32
# micro batch size
MBS=1
# number of steps to run
TOTAL_STEPS=100
# number of epochs to run
TOTAL_EPOCHS=5
# warmup steps
WARMUP_STEPS=20
# learning rate
LR=3.0e-4
# model path
MODEL_PATH=$SCRIPT_DIR
#MODEL_PATH=./Meta-Llama-3-8B/
# sequence length
#SEQ_LEN=2048
SEQ_LEN=8192
#SEQ_LEN=4096
#SEQ_LEN=1024
# Path to dataset
DATA_PATH="/shared/databricks/databricks-dolly-15k"
#MODEL_ID
#MODEL_ID="meta-llama/Meta-Llama-3-8B"
MODEL_ID="NousResearch/Meta-Llama-3.1-8B"
#Checkpoint Dir
CKPT_DIR="./LLAMA3_NEURON_CKPT"
#CKPT frequency
CKPT_FREQ=5
#Only fine tuning
#NEURON_EXTRACT_GRAPHS_ONLY=1
#############################################

export NUM_NEURONCORES=32
# NODE_ID=0
NUM_NODES=2
DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES"
# if [ ! -z "$SLURM_NTASKS" ]; then
#     NUM_NODES=$SLURM_NTASKS
#     NODE_ID=$SLURM_NODEID
#     MASTER_ADDRESS=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
#     DISTRIBUTED_ARGS="--nproc_per_node $NUM_NEURONCORES --nnodes $NUM_NODES --node_rank $NODE_ID --master_addr $MASTER_ADDRESS --master_port 44000"
#     if [ $NODE_ID -eq 0 ]; then
#         echo "NUM_NODES=$NUM_NODES"
#         echo "NODE_ID=$NODE_ID"
#         echo "MASTER_ADDRESS=$MASTER_ADDRESS"
#         echo "DISTRIBUTED_ARGS=$DISTRIBUTED_ARGS"
#     fi
#     export FI_EFA_USE_DEVICE_RDMA=1
#     export FI_PROVIDER=efa
# fi

echo "NUM_NODES=$NUM_NODES"
# echo "NODE_ID=$NODE_ID"
# echo "MASTER_ADDRESS=$MASTER_ADDRESS"

sudo sysctl -w net.ipv4.ip_local_reserved_ports=44000,48620

export NEURON_RT_NUM_CORES=32
export NUM_NEURONCORES=$NEURON_RT_NUM_CORES
export TPU_NUM_DEVICES=$NEURON_RT_NUM_CORES
export TPU_CHIPS_PER_HOST_BOUNDS=$NEURON_RT_NUM_CORES

#############################################

EXTRA_ARGS=" "
if [ $USE_MIX_PRECISION -gt 0 ]; then
    EXTRA_ARGS+=" --use_mix_precision"
fi
if [ $USE_ZERO_1 -gt 0 ]; then
    EXTRA_ARGS+=" --use_zero_1"
fi

DP=$(($NEURON_RT_NUM_CORES * $NUM_NODES / $TP_DEGREE))
#ACC_STEPS=$(($GBS / $MBS / $DP))
ACC_STEPS=16


if [ $NEURON_EXTRACT_GRAPHS_ONLY -gt 0 ]; then
    #STEPS_THIS_RUN=2
    STEPS_THIS_RUN=-1
    OUTPUT_LOG=log_compile-$NODE_ID.log
else
    STEPS_THIS_RUN=-1
    OUTPUT_LOG=log_exe-$NODE_ID.log
    EXTRA_ARGS+=" --do_eval"
fi

echo TP_DEGREE=$TP_DEGREE
echo USE_MIX_PRECISION=$USE_MIX_PRECISION
echo USE_ZERO_1=$USE_ZERO_1
echo GBS=$GBS
echo MBS=$MBS
echo TOTAL_STEPS=$TOTAL_STEPS
echo TOTAL_EPOCHS=$TOTAL_EPOCHS
echo WARMUP_STEPS=$WARMUP_STEPS
echo LR=$LR
echo MODEL_PATH=$MODEL_PATH
echo SEQ_LEN=$SEQ_LEN

echo EXTRA_ARGS=$EXTRA_ARGS
echo DP=$DP
echo ACC_STEPS=$ACC_STEPS
echo STEPS_THIS_RUN=$STEPS_THIS_RUN
echo OUTPUT_LOG=$OUTPUT_LOG

#torchrun $DISTRIBUTED_ARGS \
#    tp_zero1_llama2_7b_hf_finetune_ptl.py \
#    --model_path $MODEL_PATH \
#    --model_id $MODEL_ID \
#    --data_dir $DATA_PATH \
#    --task "open_qa" \
#    --tensor_parallel_size $TP_DEGREE \
#    --batch_size $MBS \
#    --steps_this_run $STEPS_THIS_RUN\
#    --max_steps $TOTAL_STEPS \
#    --num_train_epochs $TOTAL_EPOCHS \
#    --warmup_steps $WARMUP_STEPS \
#    --lr $LR \
#    --grad_accum_usteps $ACC_STEPS \
#    --seq_len $SEQ_LEN \
#    --selective_checkpoint_enabled \
#    --separate_qkv \
#     --save_checkpoint false\
#     --checkpoint_dir $CKPT_DIR \
#     --checkpoint_freq $CKPT_FREQ \
#    $EXTRA_ARGS |& tee $OUTPUT_LOG
# torchrun $DISTRIBUTED_ARGS \
#     tp_zero1_llama3_8b_hf_finetune_ptl.py \
#     --model_path $MODEL_PATH \
#     --model_id $MODEL_ID \
#     --data_dir $DATA_PATH \
#     --task "open_qa" \
#     --tensor_parallel_size $TP_DEGREE \
#     --batch_size $MBS \
#     --steps_this_run $STEPS_THIS_RUN\
#     --max_steps $TOTAL_STEPS \
#     --num_train_epochs $TOTAL_EPOCHS \
#     --warmup_steps $WARMUP_STEPS \
#     --lr $LR \
#     --grad_accum_usteps $ACC_STEPS \
#     --seq_len $SEQ_LEN \
#     --selective_checkpoint_enabled \
#     $EXTRA_ARGS |& tee $OUTPUT_LOG
python \
    ray_train_llama3.py \
    --model_path $MODEL_PATH \
    --model_id $MODEL_ID \
    --data_dir $DATA_PATH \
    --task "open_qa" \
    --tensor_parallel_size $TP_DEGREE \
    --batch_size $MBS \
    --steps_this_run $STEPS_THIS_RUN\
    --max_steps $TOTAL_STEPS \
    --num_train_epochs $TOTAL_EPOCHS \
    --warmup_steps $WARMUP_STEPS \
    --lr $LR \
    --grad_accum_usteps $ACC_STEPS \
    --seq_len $SEQ_LEN \
    --sequence_parallel_enabled \
    --selective_checkpoint_enabled \
    --num_nodes $NUM_NODES \
    $EXTRA_ARGS |& tee $OUTPUT_LOG
exit ${PIPESTATUS[0]}
