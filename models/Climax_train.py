import glob 
import multiprocessing as mp
import os
import time
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional
from pytorch_lightning import Trainer  
from data.seqrecord.module import MultiSourceDataModule
from data.monitor import monitor
from models.framework.pretrain import ForecastPretrain
from pytorch_lightning.cli import LightningArgumentParser, LightningCLI
from torch.utils.tensorboard import SummaryWriter
from modules.climnet import ClimNet
from configs.Climax_train_modelparam import * # hyperparameters
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

def monitoring_process(q: mp.Queue, root: Path, key_suffix: str) -> None:
    # Determine where to write the logs.
    root = root / "logs" / "monitor"
    root.mkdir(parents=True, exist_ok=True)

    # We need to maintain a counter for every key. Otherwise `SummaryWriter` behaves weirdly.
    counter: Dict[str, int] = defaultdict(lambda: 0)

    # Perform logging loop.
    sw = SummaryWriter(str(root))
    receive, join = monitor()
    while True:
        if not q.empty() and q.get() == "kill":
            join()
            break
        for k, v in receive():
            k += key_suffix
            sw.add_scalar(k, v, counter[k])
            counter[k] += 1
        time.sleep(0.1)


class Monitor:
    def __init__(self, root: Path) -> None:
        c = mp.get_context("fork")
        self.root = root
        self.q = c.Manager().Queue()
        self.p: Optional[mp.Process] = None

    def __enter__(self):
        if "NODE_RANK" in os.environ:
            if int(os.environ["LOCAL_RANK"]) != 0:
                # Only monitor in rank zero processes.
                return
            else:
                # Produce separate reports for all nodes.
                key_suffix = "/" + os.environ["NODE_RANK"]
        else:
            key_suffix = ""
        self.p = mp.Process(target=monitoring_process, args=(self.q, self.root, key_suffix))
        self.p.start()

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.p:
            self.q.put("kill")
            self.p.join()


class CustomLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        parser.add_argument("--disable-monitor", default=False)


def main():
    net = ClimNet(
        const_vars=tuple(const_vars),
        single_vars=tuple(single_vars),
        atmos_vars=tuple(atmos_vars),
        atmos_levels=tuple(atmos_levels),
        patch_size=16,
        embed_dim=768,
        depth=12,
        decoder_depth=2,
        num_heads=12,
        mlp_ratio=4,
        drop_path=0.1,
        drop_rate=0.1,
        use_flash_attn=True,
    )
    torch.save(net.state_dict(), "/blob/weather-blob/DEL-Weather/checkpoints/model_weights_init_2.pth") 
    model_class = ForecastPretrain(net, restart_path, lr, beta_1, beta_2, weight_decay, warmup_steps, 
                                   warmup_start_lr=warmup_start_lr, eta_min=eta_min, opt_name=opt_name,)
    datamodule_class = MultiSourceDataModule(dict_root_dirs, dict_data_spatial_shapes, 
                                             dict_single_vars, dict_atmos_vars, dict_hrs_each_step, 
                                             dict_max_predict_range, batch_size, dict_metadata_dirs, 
                                             shuffle_buffer_size=shuffle_buffer_size, 
                                             val_shuffle_buffer_size=val_shuffle_buffer_size, 
                                             num_workers=num_workers,
                                             pin_memory=pin_memory,
                                             use_old_loader=use_old_loader)
    # cli = CustomLightningCLI(
    #     model_class=model_class,
    #     datamodule_class=datamodule_class,
    #     seed_everything_default=42,
    #     run=False,
    #     # parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    #     # parser_kwargs={"parser_mode": "yaml", "error_handler": None},
    # )
    # default logger used by trainer (if tensorboard is installed)
    os.makedirs(default_root_dir, exist_ok=True)
    prev_ckpts = glob.glob(os.path.join(default_root_dir, "checkpoints", "*.ckpt"))
    if len(prev_ckpts) > 0:
        resume_from_checkpoint = os.path.join(
            default_root_dir, "checkpoints", "last.ckpt"
        )
    else:
        resume_from_checkpoint = None
    checkpoint_callback = ModelCheckpoint(  
        dirpath=dirpath,  # saving checkpoint dir  
        filename=filename,  # checkpoints file name
        save_top_k=save_top_k,  # save top k models
        verbose=verbose,  # Only print news when save models
        monitor=monitor_param,  # use which loss to judge model 
        mode=mode,  # val_loss min is better  
        save_last=mode,  # save the last model too
    )  
    logger = TensorBoardLogger(save_dir=default_checkpoints_dir, version=1, name="lightning_logs")
    trainer = Trainer(accelerator=accelerator, devices=devices, max_epochs=max_epochs, 
                      enable_checkpointing=enable_checkpointing, strategy=strategy, 
                      logger=logger, precision=precision, num_nodes=num_nodes, callbacks=[checkpoint_callback])   
    print(type(model_class))
    
    # local debug mode will automatically disable the monitor.
    # if not cli.config.disable_monitor:
    #     with Monitor(Path(default_root_dir)):
    #         cli.trainer.fit(
    #             model=cli.model,
    #             datamodule=cli.datamodule,
    #             ckpt_path=resume_from_checkpoint,
    #         )
    # else:
    #     cli.trainer.fit(
    #         model=cli.model,
    #         datamodule=cli.datamodule,
    #         ckpt_path=resume_from_checkpoint,
    #     )
    trainer.fit(model_class, datamodule=datamodule_class, ckpt_path=resume_from_checkpoint,) 
    

if __name__ == "__main__":
    main()
