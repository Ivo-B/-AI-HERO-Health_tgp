import json
import os
import os.path as osp
import pickle
import shutil
import tempfile
import time
from argparse import ArgumentParser

import mmcv
import numpy as np
import pandas as pd
import requests
import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import get_dist_info
from pytorch_lightning import Trainer

from src.datamodules.mvit_datamodule import MViTDataModuleTesting
from src.models.effinet_module import EFFILitModule
from src.models.mvit_module import MVITLitModule


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device="cuda"
    )
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device="cuda")
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device="cuda")
    part_send[: shape_tensor[0]] = part_tensor
    part_recv_list = [part_tensor.new_zeros(shape_max) for _ in range(world_size)]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(pickle.loads(recv[: shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results


def collect_results_cpu(result_part, size, tmpdir="/tmp/.dist_test"):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN,), 32, dtype=torch.uint8, device="cuda")
        if rank == 0:
            mmcv.mkdir_or_exist(".dist_test")
            tmpdir = tempfile.mkdtemp(dir=".dist_test")
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device="cuda"
            )
            dir_tensor[: len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f"part_{rank}.pkl"))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f"part_{i}.pkl")
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


if __name__ == "__main__":
    parser = ArgumentParser()
    # parser.add_argument("--weights_path", type=str, default='/hkfs/work/workspace/scratch/im9193-H5/logs/experiments/our_split/runs/2022-02-03/09-29-07/checkpoints/epoch_038.ckpt',
    # parser.add_argument("--weights_path", type=str, default='/hkfs/work/workspace/scratch/im9193-H5/logs/experiments/effinet/runs/2022-02-02/18-26-53/checkpoints/epoch_004.ckpt',
    parser.add_argument(
        "--weights_path",
        type=str,
        default="/hkfs/work/workspace/scratch/im9193-H5/logs/experiments/baseline_02_02/runs/2022-02-02/19-04-47/checkpoints/epoch_030.ckpt",
        help="Model weights path",
    )  # TODO: adapt to your model weights path in the bash script
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Directory where weights and results are saved",
        default="/hkfs/work/workspace/scratch/im9193-H5/submission_test",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        help="Directory containing the data you want to predict",
        default="/hkfs/work/workspace/scratch/im9193-health_challenge",
    )
    args = parser.parse_args()

    weights_path = args.weights_path
    save_dir = args.save_dir
    data_dir = args.data_dir

    os.makedirs(save_dir, exist_ok=True)

    check_script = "test_data" not in data_dir

    filename = weights_path.split("/")[-1]

    # dataloader
    data_split = "validation" if check_script else "test"
    print("Running inference on {} data".format(data_split))

    # # Init lightning datamodule
    # datamodule = MViTDataModuleTesting(
    #     test_path=os.path.join(data_dir, 'data/valid.csv'),
    #     img_dir=os.path.join(data_dir, 'data/imgs/' ),
    #     batch_size=2,
    #     num_workers=1
    # )

    datamodule = MViTDataModuleTesting(
        test_path=os.path.join(
            data_dir, "evaluation/{}.csv".format("valid" if check_script else "test")
        ),
        img_dir=os.path.join(data_dir, "data/imgs" if check_script else "data/test"),
        batch_size=128,
        num_workers=20,
        data_size=224,
    )

    # Init lightning model
    if "effinet" in weights_path:
        model = EFFILitModule.load_from_checkpoint(weights_path)
    else:
        model = MVITLitModule.load_from_checkpoint(weights_path)
    model.eval()
    model.freeze()

    trainer: Trainer = Trainer(gpus=4, strategy="ddp", callbacks=None, logger=None)

    predictions = trainer.predict(
        model=model, datamodule=datamodule, return_predictions=True
    )

    ordered_results = collect_results_cpu(predictions, len(datamodule.data_test))
    rank, world_size = get_dist_info()
    if rank == 0:
        flattened_preds = [np.concatenate(i) for i in list(zip(*ordered_results))]
        df = pd.DataFrame(
            data={
                "image": flattened_preds[0],
                "prediction": np.squeeze(flattened_preds[1]),
            }
        )
        df.to_csv(os.path.join(save_dir, "predictions.csv"), index=False)

        print("Done! The result is saved in {}".format(save_dir))
