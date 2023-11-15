import torch.multiprocessing as mp
from data.study_dataset import get_splitted_dataloaders

from configs import configs
from models.bottleneck import ViTBottleneckForImageClassification
import torch
import torch.nn as nn
import numpy as np
import cv2
from transformers import get_cosine_schedule_with_warmup
from src.utils import save_img

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor

from sklearn.metrics import average_precision_score as AUPRC
from sklearn.metrics import roc_auc_score as AUROC
from sklearn.metrics import f1_score

import os.path
pjoin = os.path.join
import warnings
warnings.filterwarnings('ignore')

from torch.utils.tensorboard import SummaryWriter

class LightningSupervisedModel(pl.LightningModule):
    def __init__(self, from_checkpoint=None, class_means=None, CFG=None):
        super().__init__()
        assert CFG is not None, "You must specify a CFG to initialize the model"

        self.CFG = CFG
        self.name = CFG.name
        self.config = configs[CFG.config]
        self.model = ViTBottleneckForImageClassification(
            configs[CFG.config], 
            class_means=class_means, 
            from_checkpoint=from_checkpoint
            )
        self.validation_losses = []
        self.validation_preds = []
        self.validation_targets = []
        self.labels_name = []
        
    @torch.no_grad()
    def get_patient0_info(self, id_patient_0):
        self.ref = dict()

        # load study data related to patient_0 used for visualisation
        patient_0 = self.trainer.val_dataloaders.dataset.get_by_pid(id_patient_0, device=self.device)
        self.ref['images'] = patient_0['images']
        self.ref['lengths'] = patient_0['lengths']
        
        # create reference noise to reproduce same masking at each step of the validation
        batch_size = sum(map(len, self.ref['images']))
        seq_length = (self.config.image_size // self.config.patch_size) ** 2
        
        #seeded noise to be able to compare different pretraining runs fairly
        rng = np.random.default_rng(seed=69)
        random_array = rng.random((batch_size, seq_length))
        self.ref['noises'] = torch.from_numpy(random_array).to(self.device)

        #dummy call to the model to generate a reference mask
        self.ref['masks'] = self.model.vit(
            pixel_values=self.ref['images'], 
            noise=self.ref['noises'],
            lengths=self.ref['lengths'], 
        ).mask.unsqueeze(2)

    @torch.no_grad()
    def log_patient0_attention_map(self):
        backbone = self.model.vit
        images = self.ref['images']
        lengths = self.ref['lengths']

        #extract attention scores from encoder
        features = backbone.embeddings.patch_embeddings(images)
        features = torch.cat([backbone.embeddings.cls_token.repeat(features.size(0), 1, 1), features], dim=1)
        features += backbone.embeddings.position_embeddings
        features = backbone.encoder(features, lengths=lengths, output_attentions=True).attentions

        #create attention map for patch tokens wrt cls token
        attentions = torch.stack(features, dim=1).squeeze()
        attentions = attentions.mean(dim=2) #average attention across heads
        studywise_attention = torch.split(attentions, tuple(lengths))
        patient_result = []
        for study_attention in studywise_attention:
            study_attention = study_attention[:, :, 0, 1:] #attention wrt cls token towards patch tokens
            sums = study_attention.sum(dim=-1, keepdims=True)
            study_attention /= sums

            study_attention = torch.stack([torch.prod(image_attention , dim=0) for image_attention in study_attention])
            mean, var = study_attention.mean(dim=-1, keepdims=True), study_attention.var(dim=-1, keepdims=True)
            study_attention = (study_attention - mean) / var ** 0.5
            mini, maxi = study_attention.min(dim=-1, keepdims=True).values, study_attention.max(dim=-1, keepdims=True).values

            study_attention = (study_attention - mini) / (maxi - mini)

            study_attention = study_attention.reshape(-1, 8, 8)
            study_result = torch.stack([torch.tensor(cv2.resize(image_attention.cpu().numpy(), (self.config.image_size, self.config.image_size))) for image_attention in study_attention])

            patient_result.append(study_result)
        patient_result = torch.cat(patient_result).unsqueeze(1).to(self.device)

        #create attention-augmented image
        green_channel = (1 + images) / 2
        red_channel = patient_result.clip(0, 2/3) * (3 / 2)
        blue_channel = patient_result.clip(1/3, 1) * (3 / 2) - (1 / 2)
        attended_images = torch.stack([red_channel, green_channel, blue_channel], dim=-1).squeeze()

        epoch = 1 + self.current_epoch
        attended_images = torch.split(attended_images, tuple(lengths))
        for study_idx, study in enumerate(attended_images):
            for idx, image in enumerate(study):
                path = pjoin("checkpoints", self.name)
                if not os.path.exists(path):
                    os.mkdir(path)
                path = pjoin(path, f"amap_s{study_idx + 1}i{idx + 1}")
                if not os.path.exists(path):
                    os.mkdir(path)
                save_img(image, pjoin(path, f"epoch_{epoch}.jpg"))

    def forward(self, inputs, lengths, labels):
        return self.model(pixel_values=inputs, lengths=lengths, labels=labels)

    def training_step(self, batch, batch_idx):
        pixel_values, lengths = batch['images'], batch['lengths']
        y = batch['labels']
        res = self(pixel_values, lengths=lengths, labels=y)
        loss = res['loss']
        return loss
    
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def on_test_epoch_end(self):
        is_distributed = self.trainer.world_size > 1
        device = self.device

        self.validation_losses = torch.tensor(self.validation_losses).to(device)
        self.validation_targets = torch.cat(self.validation_targets).to(device)
        self.validation_preds = torch.cat(self.validation_preds).to(device)

        if self.validation_targets.shape[0] > 10:
            mean_auroc, mean_auprc, mean_f1, mean_precision, mean_recall = [], [], [], [], []
            for idx in range(self.validation_preds.size(1)):
                auroc = AUROC(self.validation_targets[:, idx].cpu(), self.validation_preds[:, idx].cpu())
                auprc = AUPRC(self.validation_targets[:, idx].cpu(), self.validation_preds[:, idx].cpu())

                # Computing predictions from probabilities (assuming threshold of 0.5)
                binary_preds = (self.validation_preds[:, idx] > 0.5).long()
                
                f1 = f1_score(self.validation_targets[:, idx].cpu(), binary_preds.cpu())
                # precision = precision_score(self.validation_targets[:, idx].cpu(), binary_preds.cpu())
                # recall = recall_score(self.validation_targets[:, idx].cpu(), binary_preds.cpu())

                mean_auroc.append(auroc)
                mean_auprc.append(auprc)
                mean_f1.append(f1)
                # mean_precision.append(precision)
                # mean_recall.append(recall)

                self.log(f'test_auroc/class_{idx}', auroc, on_step=False, on_epoch=True, sync_dist=is_distributed)
                self.log(f'test_auprc/class_{idx}', auprc, on_step=False, on_epoch=True, sync_dist=is_distributed)
                self.log(f'test_f1/class_{idx}', f1, on_step=False, on_epoch=True, sync_dist=is_distributed)
                # self.log(f'test_precision/class_{idx}', precision, on_step=False, on_epoch=True, sync_dist=is_distributed)
                # self.log(f'test_recall/class_{idx}', recall, on_step=False, on_epoch=True, sync_dist=is_distributed)

            self.log('test_auroc/mean', sum(mean_auroc) / len(mean_auroc), on_step=False, on_epoch=True, sync_dist=is_distributed)
            self.log('test_auprc/mean', sum(mean_auprc) / len(mean_auprc), on_step=False, on_epoch=True, sync_dist=is_distributed)
            self.log('test_f1/mean', sum(mean_f1) / len(mean_f1), on_step=False, on_epoch=True, sync_dist=is_distributed)
            # self.log('test_precision/mean', sum(mean_precision) / len(mean_precision), on_step=False, on_epoch=True, sync_dist=is_distributed)
            # self.log('test_recall/mean', sum(mean_recall) / len(mean_recall), on_step=False, on_epoch=True, sync_dist=is_distributed)
            
        self.log('mean_test_loss', self.validation_losses.mean(), on_step=False, on_epoch=True, sync_dist=is_distributed)

        self.validation_losses = []
        self.validation_targets = []
        self.validation_preds = []

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pixel_values, lengths = batch['images'], batch['lengths']
        y = batch['labels']

        res = self(pixel_values, lengths=lengths, labels=y)
        loss = res['loss']

        self.validation_losses.append(loss.cpu())
        self.validation_targets.append(y.cpu())
        preds = torch.sigmoid(res['logits'])
        self.validation_preds.append(preds.cpu())

    def on_validation_epoch_end(self):
        is_distributed = self.trainer.world_size > 1
        device = self.device

        self.validation_losses = torch.tensor(self.validation_losses).to(device)
        self.validation_targets = torch.cat(self.validation_targets).to(device)
        self.validation_preds = torch.cat(self.validation_preds).to(device)

        if self.validation_targets.shape[0] > 10:
            aurocs, auprcs, f1s = [], [], []
            for idx in range(self.validation_preds.size(1)):
                binary_preds = (self.validation_preds[:, idx] > 0.5).long()                

                aurocs.append(AUROC(self.validation_targets[:, idx].cpu(), self.validation_preds[:, idx].cpu()))
                auprcs.append(AUPRC(self.validation_targets[:, idx].cpu(), self.validation_preds[:, idx].cpu()))
                f1s.append(f1_score(self.validation_targets[:, idx].cpu(), binary_preds.cpu()))

            # Add mean metrics to "Time series"
            self.log('mean_auroc', sum(aurocs) / len(aurocs), on_step=False, on_epoch=True, sync_dist=is_distributed)
            self.log('mean_auprc', sum(auprcs) / len(auprcs), on_step=False, on_epoch=True, sync_dist=is_distributed)
            self.log('mean_f1', sum(f1s) / len(f1s), on_step=False, on_epoch=True, sync_dist=is_distributed)

            auroc_dict = {name: aurocs[idx] for idx, name in enumerate(self.labels_name)}
            auprc_dict = {name: auprcs[idx] for idx, name in enumerate(self.labels_name)}
            f1_dict = {name: f1s[idx] for idx, name in enumerate(self.labels_name)}
            
            # Add class-wise metrics to "Scalars"
            log_dict = {"auroc":auroc_dict, "auprc":auprc_dict, "f1":f1_dict,}
            writer = SummaryWriter()
            for metric_name, values_dict in log_dict.items():
                writer.add_scalars(metric_name, values_dict, self.global_step)
            
        self.log('supervised_val_loss', self.validation_losses.mean(), on_step=False, on_epoch=True, sync_dist=is_distributed)

        self.validation_losses = []
        self.validation_targets = []
        self.validation_preds = []

        # Save patient0's attention maps
        self.log_patient0_attention_map()

    def on_train_start(self):
        # a reference valid set patient having two studies:
        # - the 1st one labellized as "pneumonia" 
        # - the 2nd as "no pneumonia"
        self.get_patient0_info(18026823)

        # Save patient0's attention maps
        self.log_patient0_attention_map()

        optimizer = self.trainer.optimizers[0]
        scheduler_config = self.trainer.lr_scheduler_configs[0]
        scheduler_config.interval = "step"

        self.labels_name = self.trainer.train_dataloader.dataset.labels
        
        num_training_steps = len(self.trainer.train_dataloader) * self.trainer.max_epochs
        num_warmup_steps = int(0.05 * num_training_steps)  # 5% warmup steps

        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
            
        scheduler_config.scheduler = scheduler
        self.trainer.lr_scheduler_configs[0] = scheduler_config

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)

        for mn, m in self.model.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
                elif not p.requires_grad:
                    no_decay.add(fpn)
                elif fpn in ['vit.embeddings.cls_token']:
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.model.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": self.CFG.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]

        optimizer = torch.optim.AdamW(optim_groups, lr=self.CFG.base_lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=10, 
            num_training_steps=100
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

def supervised(CFG, args, dataloaders):
    train_dl, valid_dl, test_dl = dataloaders
    for dl in [train_dl, valid_dl, test_dl]:
        dl.dataset.include_labels = True

    checkpoint_path = pjoin("checkpoints", CFG.name)
    if not os.path.exists(checkpoint_path):
        print(f'Did not find reference pretrain run at path {checkpoint_path}. Running pretrain.')
    
    lr_callback = LearningRateMonitor(logging_interval='step')

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        save_top_k=1, 
        verbose=True, 
        monitor='supervised_val_loss', 
        mode='min'
    )

    early_stop_callback = EarlyStopping(
        monitor='supervised_val_loss',
        patience=CFG.patience,
        verbose=True,
        mode='min'
    )

    trainer = pl.Trainer(
        strategy="ddp", 
        accelerator="gpu", devices=args.device,
        callbacks=[
            lr_callback,
            checkpoint_callback, 
            early_stop_callback,
            ],
        max_epochs=CFG.epochs,
        num_sanity_val_steps=0
        )

    model = LightningSupervisedModel(
        from_checkpoint=pjoin(checkpoint_path, "backbone_weights.pth"),
        class_means=train_dl.dataset.label_means,
        CFG=CFG,
        )
    
    trainer.fit(
        model=model, 
        train_dataloaders=train_dl,
        val_dataloaders=valid_dl,
        )
    
    trainer.test(
        model=model, 
        dataloaders=test_dl
        )