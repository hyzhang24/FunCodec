#!/usr/bin/env python3

import os
import torch
from funcodec.tasks.gan_speech_codec import GANSpeechCodecTask
import logging

logging.basicConfig(
    level=logging.DEBUG,
    format='%(message)s',  # 只输出日志信息
    filename='/home/users/ntu/ccdshyzh/FunCodec/egs/LibriTTS/codec/logs.log',  # 指定日志文件路径
    filemode='w'  # 文件模式，'w' 表示覆盖，'a' 表示追加
)

# for ASR Training
def parse_args():
    parser = GANSpeechCodecTask.get_parser()
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="local gpu id.",
    )
    args = parser.parse_args()
    return args


def main(args=None, cmd=None):
    # for codec training
    GANSpeechCodecTask.main(args=args, cmd=cmd)


if __name__ == '__main__':
    logging.info("Parsing args...")
    args = parse_args()

    if torch.cuda.is_available():
        # 获取可用的 CUDA 设备数量
        num_devices = torch.cuda.device_count()
        print(f"Number of CUDA devices: {num_devices}")

        # 遍历每个 CUDA 设备并打印其信息
        for i in range(num_devices):
            device_name = torch.cuda.get_device_name(i)
            device_capability = torch.cuda.get_device_capability(i)
            device_memory = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # 以 GB 为单位
            print(f"Device {i}: {device_name}")
            print(f"  Compute Capability: {device_capability[0]}.{device_capability[1]}")
            print(f"  Total Memory: {device_memory:.2f} GB")
    else:
        print("No CUDA devices available.")

    # setup local gpu_id
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    if torch.__version__ >= "1.7.2":
        torch.cuda.set_device(args.gpu_id)

    # DDP settings
    if args.ngpu > 1:
        args.distributed = True
    else:
        args.distributed = False
    assert args.num_worker_count == 1

    # re-compute batch size: when dataset type is small
    if args.dataset_type == "small":
        if args.batch_size is not None:
            args.batch_size = args.batch_size * args.ngpu
        if args.batch_bins is not None:
            args.batch_bins = args.batch_bins * args.ngpu

    logging.info("Codec Training Started...")
    main(args=args)
