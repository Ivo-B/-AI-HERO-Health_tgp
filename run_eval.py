import os
import requests
import json
from argparse import ArgumentParser
import pandas as pd
import torch
import torch.nn as nn
import numpy as np

from src.datamodules.mvit_datamodule import MViTDataModuleTesting
from src.models.mvit_module import MVITLitModule
from src.models.effinet_module import EFFILitModule
from pytorch_lightning import Trainer


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--weights_path", type=str, default='/hkfs/work/workspace/scratch/im9193-H5/logs/experiments/effinet/runs/2022-02-02/18-26-53/checkpoints/epoch_004.ckpt',
                        help="Model weights path")  # TODO: adapt to your model weights path in the bash script
    parser.add_argument("--save_dir", type=str, help='Directory where weights and results are saved',
                        default='/hkfs/work/workspace/scratch/im9193-H5/submission')
    parser.add_argument("--data_dir", type=str, help='Directory containing the data you want to predict',
                        default='/hkfs/work/workspace/scratch/im9193-H5')
    args = parser.parse_args()

    weights_path = args.weights_path
    save_dir = args.save_dir
    data_dir = args.data_dir

    os.makedirs(save_dir, exist_ok=True)

    check_script = 'test_data' not in data_dir

    filename = weights_path.split('/')[-1]

    # dataloader
    data_split = 'validation' if check_script else 'test'
    print('Running inference on {} data'.format(data_split))

    # # Init lightning datamodule
    # datamodule = MViTDataModuleTesting(
    #     test_path=os.path.join(data_dir, 'data/valid.csv'),
    #     img_dir=os.path.join(data_dir, 'data/imgs/' ),
    #     batch_size=2,
    #     num_workers=1
    # )

    datamodule = MViTDataModuleTesting(
        test_path=os.path.join(data_dir, 'evaluation/{}.csv'.format('valid' if check_script else 'test')),
        img_dir=os.path.join(data_dir, 'data/imgs' if check_script else 'data/test'),
        batch_size=64,
        num_workers=128
    )

    # Init lightning model
    if "effinet" in weights_path:
        model = EFFILitModule.load_from_checkpoint(weights_path)
    else:
        model = MVITLitModule.load_from_checkpoint(weights_path)
    model.eval()
    model.freeze()

    trainer: Trainer = Trainer(gpus=4, strategy="ddp", callbacks=None, logger=None)

    predictions = trainer.predict(model=model, datamodule=datamodule, return_predictions=True)

    flattened_preds = [np.concatenate(i) for i in list(zip(*predictions))]
    df = pd.DataFrame(data={'image': flattened_preds[0], 'prediction': np.squeeze(flattened_preds[1])})
    df.to_csv(os.path.join(save_dir, 'predictions.csv'), index=False)

    print('Done! The result is saved in {}'.format(save_dir))
