#!/bin/bash

echo "funcodec run"

. ./path.sh || exit 1;

# machines configuration
gpu_devices="0,1"
gpu_num=2
count=1

# general configuration
feats_dir="."
exp_dir="."
# dumpdir=dump/LibriTTS
dumpdir=/home/users/ntu/ccdshyzh/scratch/dump/LibriTTS
stage=3
stop_stage=3
# corpus_dir=corpus/LibriTTS
corpus_dir=/home/users/ntu/ccdshyzh/scratch/corpus/LibriTTS

# training related
tag=""
train_set=train
valid_set=dev
# train_config=conf/encodec_lstm_16k_n32_600k_step_rmseg.yaml
train_config=conf/soundstream_16k_n32_600k_step.yaml
init_param=
state_dir=LibriTTS_states

# inference related
inference_model=30epoch.pth
inference_tag="inference"
batch_size=1
test_sets="test-clean"
gpu_inference=true  # Whether to perform gpu decoding, set false for cpu decoding
need_indices=false
need_sub_quants=false
# for gpu decoding, inference_nj=ngpu*njob; for cpu decoding, inference_nj=njob
njob=5
docker_nj=32
infer_cmd=utils/run.pl
sample_frequency=16000
file_sampling_rate=16000
bit_width=4000
use_scale=false
use_ppg=false
model_dir=

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',

echo "model dir settings..."

if [ -z "${model_dir}" ]; then
  model_dir="$(basename "${train_config}" .yaml)${tag}"
fi

# you can set gpu num for decoding here
gpuid_list=$gpu_devices  # set gpus for decoding, the same as training stage by default
ngpu=$(echo $gpuid_list | awk -F "," '{print NF}')

echo "inference settings..."

inference_nj=$((ngpu * njob))
_ngpu=1

echo "start training..."
# Training Stage
world_size=$gpu_num  # run on one machine

echo "stage 3: Training"
mkdir -p ${exp_dir}/exp/${model_dir}
mkdir -p ${exp_dir}/exp/${model_dir}/log
INIT_FILE=${exp_dir}/exp/${model_dir}/ddp_init
if [ -f $INIT_FILE ];then
    rm -f $INIT_FILE
fi
ppg_opt=""
init_opt=""
# if [ ! -z "${init_param}" ]; then
#     init_opt="--init_param ${init_param}"
#     echo ${init_opt}
# fi

init_method=file://$(readlink -f $INIT_FILE)
echo "log can be found at ${exp_dir}/exp/${model_dir}/log/train.log.0"
for ((i = 0; i < $gpu_num; ++i)); do
    {
        rank=$i
        local_rank=$i
        gpu_id=$(echo $gpu_devices | cut -d',' -f$[$i+1])
        python -m funcodec.bin.codec_train \
            --gpu_id $gpu_id \
            --use_preprocessor true \
            --train_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${train_set}/wav.scp,speech,kaldi_ark \
            --train_shape_file ${feats_dir}/exp/${state_dir}/${train_set}/speech_shape \
            --valid_data_path_and_name_and_type ${feats_dir}/${dumpdir}/${valid_set}/wav.scp,speech,kaldi_ark \
            --valid_shape_file ${feats_dir}/exp/${state_dir}/${valid_set}/speech_shape \
            ${init_opt} --ignore_init_mismatch true \
            ${ppg_opt} --resume true \
            --output_dir ${exp_dir}/exp/${model_dir} \
            --config $train_config \
            --ngpu $gpu_num \
            --num_worker_count $count \
            --multiprocessing_distributed true \
            --dist_init_method $init_method \
            --dist_world_size $world_size \
            --dist_rank $rank \
            --local_rank $local_rank 1> ${exp_dir}/exp/${model_dir}/log/train.log.$i 2>&1
    } &
    done
    wait

EOF
