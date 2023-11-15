import torch.multiprocessing as mp
import os.path
from os.path import join as pjoin

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score as AUPRC
from sklearn.metrics import roc_auc_score as AUROC
from sklearn.metrics import f1_score

import lightning.pytorch as pl
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
import torch
from transformers import get_cosine_schedule_with_warmup
import numpy as np

from configs import configs
from models.bottleneck import ViTMAEBottleneckForPreTraining, DiffMAEBottleneckForPreTraining, ViTMAEBottleneckModel

from src.utils import get_nth_boxed_visualisation, log_results


class VanillaPretrainingModel(pl.LightningModule):
    def __init__(self, CFG=None):
        super().__init__()
        assert CFG is not None, "You must specify a CFG to initialize the model"

        self.CFG = CFG
        self.name = CFG.name
        self.config = configs[CFG.config] #Size of the ViT encoder (like 'xs' or 'l')

        self.model = ViTMAEBottleneckForPreTraining(self.config)

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
        self.ref['masks'] = self.model(
            pixel_values=self.ref['images'], 
            noise=self.ref['noises'],
            lengths=self.ref['lengths'], 
        ).mask.unsqueeze(2)

    def forward(self, pixel_values, lengths, noise=None):
        return self.model(pixel_values=pixel_values, lengths=lengths, noise=noise)

    def training_step(self, batch, batch_idx):
        pixel_values = batch['images']
        lengths = batch['lengths']
        res = self(pixel_values, lengths)
        loss = res['loss']
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pixel_values, lengths = batch['images'], batch['lengths']
        res = self(pixel_values, lengths)
        loss = res['loss']

        self.log('valid_loss', loss, on_step=False, on_epoch=True)

    def get_mixed_image(self, image, logits, mask):
        true_patches = self.model.patchify(image) * (1 - mask)
        pred_patches = logits * mask
        mixed_image = self.model.unpatchify(true_patches + pred_patches)
        return mixed_image
    
    @torch.no_grad()
    def extract_features(self, images, lengths):
        '''
        Extract cls features from the backbone ViT encoder
        '''

        images, lengths = images.to(self.device), lengths.to(self.device)

        encoder = self.model.vit
        features = encoder.embeddings.patch_embeddings(images)
        features = torch.cat([encoder.embeddings.cls_token.repeat(features.size(0), 1, 1), features], dim=1)
        features += encoder.embeddings.position_embeddings
        features = encoder.encoder(features, lengths=lengths).last_hidden_state

        #pool to study-wise and cls-indexed representation
        index = torch.cumsum(lengths, 0) - lengths[0]
        features = features[index, 0, :]

        return features
    
    @torch.no_grad()
    def extract_dataloader(self, dataloader):
        '''
        Extract study representations from a whole dataloader
        '''

        dataloader_features = torch.empty((0, self.config.hidden_size), device=self.device)
        dataloader_labels = torch.empty((0, self.config.num_labels), device=self.device)
        for batch in dataloader:
            features = self.extract_features(images=batch['images'], lengths=batch['lengths'])
            dataloader_features = torch.cat([dataloader_features, features])
            
            if "labels" in batch.keys():
                labels = batch['labels'].to(self.device)
                dataloader_labels = torch.cat([dataloader_labels, labels])

        return dataloader_features.cpu().numpy(), dataloader_labels.cpu().numpy()
    
    @torch.no_grad()
    def linear_probe(self):
        clf = LogisticRegression(solver='saga', class_weight="balanced", penalty=None)
        
        model = self.model
        model.eval()
        X_train, y_train = self.extract_dataloader(self.trainer.train_dataloader)
        X_test, y_test = self.extract_dataloader(self.trainer.val_dataloaders)
        model.train()
        
        auroc_list, auprc_list, f1_list = [], [], []
        for idx in range(y_train.shape[1]):
            y_train_i, y_test_i = y_train[:, idx], y_test[:, idx]

            clf.fit(X_train, y_train_i)
            y_pred_proba = clf.predict_proba(X_test)[:, 1]

            auroc_list.append(AUROC(y_test_i, y_pred_proba))
            auprc_list.append(AUPRC(y_test_i, y_pred_proba))

            y_pred_binary = (y_pred_proba > 0.5).astype(float)
            f1_list.append(f1_score(y_test_i, y_pred_binary))

        auroc_list.append(np.array(auroc_list).mean())
        auprc_list.append(np.array(auprc_list).mean())
        f1_list.append(np.array(f1_list).mean())
        
        log_results(pjoin("checkpoints", f"linear_probing_{self.current_epoch}.txt"), self.trainer.train_dataloader.dataset.labels, auroc_list, auprc_list, f1_list)

    @torch.no_grad()
    def on_validation_epoch_end(self):
        if not hasattr(self, "ref"):
            return
        
        logits = self.model(self.ref['images'], noise=self.ref['noises'], lengths=self.ref['lengths']).logits
        images = self.ref['images']
        if self.config.norm_pix_loss:
            images, patch_means, patch_vars = self.norm_patches(images)

        mixed_image = self.get_mixed_image(images, logits, self.ref['masks'])
        self.log_reconstructed_images(mixed_image, lengths=self.ref['lengths'])

        #perform linear classification to evaluate learned study representations
        if self.current_epoch % 50 == 0 or self.current_epoch == self.trainer.max_epochs - 1:
            self.linear_probe()

    @torch.no_grad()
    def norm_patches(self, pixel_values):
        '''Normalize images patch-wise and return the patches mean and variance.'''

        sequence = self.model.patchify(pixel_values)
        
        patch_means = sequence.mean(dim=-1, keepdim=True)
        patch_vars = sequence.var(dim=-1, keepdim=True)
        sequence = (sequence - patch_means) / (patch_vars + 1.0e-6) ** 0.5
        
        pixel_values = self.model.unpatchify(sequence)
        return pixel_values, patch_means, patch_vars

    @torch.no_grad()
    def log_reconstructed_images(self, mixed_image, lengths, original_images=False):
        mixed_image = torch.split(mixed_image, tuple(lengths))
        if original_images:
            mask = torch.split(torch.zeros_like(self.ref['masks']), tuple(lengths))
        else:
            mask = torch.split(self.ref['masks'], tuple(lengths)) 

        epoch = 1 + self.current_epoch if not original_images else self.current_epoch
        for study_idx, study in enumerate(mixed_image):
            for idx, image in enumerate(study):
                path = pjoin("checkpoints", self.name, f"s{study_idx + 1}i{idx + 1}"); os.makedirs(path, exist_ok=True)
                # path = pjoin("checkpoints", self.name); os.mkdir(path, exist_ok=True)
                # path = pjoin(path, f"s{study_idx + 1}i{idx + 1}"); os.mkdir(path, exist_ok=True)
                get_nth_boxed_visualisation(self.config, image, mask=mask[study_idx][idx], save=pjoin(path, f"epoch_{epoch}.jpg"))
        
    def on_train_start(self):
        if self.trainer.val_dataloaders is None:
            self.trainer.val_dataloaders = self.val_dataloader()

        # a reference valid set patient having two studies:
        # - the 1st one labellized as "pneumonia" 
        # - the 2nd as "no pneumonia"
        self.get_patient0_info(18026823)

        # Save patient0's reconstructed images
        if self.config.norm_pix_loss:
            images, patch_means, patch_vars = self.norm_patches(self.ref['images'])
        self.log_reconstructed_images(images, lengths=self.ref['lengths'], original_images=True)

        #define optimizer & scheduler with right number of steps
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
                elif fpn in ['vit.embeddings.cls_token', 'decoder.mask_token']:
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
        scheduler = get_cosine_schedule_with_warmup( #TODO pê possible de pas mettre de scheduler ici puisqu'on le modifie après
            optimizer, 
            num_warmup_steps=10, 
            num_training_steps=100
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }

class DiffusionPretrainingModel(VanillaPretrainingModel):
    def __init__(self, CFG=None):
        assert CFG is not None, "You must specify a CFG to initialize the model"
        super().__init__(CFG)

        self.CFG = CFG
        self.name = CFG.name
        self.config = configs[CFG.config] #String size of the ViT encoder (e.g 'b', 's'...)

        self.model = DiffMAEBottleneckForPreTraining(self.config)

        self.num_diffusion_steps = 100
        self.num_sparse_diffusion_steps = 5
        self.probe_every = 5

        self.beta_min = torch.tensor(1e-4)
        self.beta_max = torch.tensor(2e-1)
        rho = torch.tensor(1., device=self.device) # Hyperparameter to tune the schedule of noise
        beta_schedule = torch.linspace(self.beta_min, self.beta_max, self.num_diffusion_steps, device=self.device) ** rho
        alpha_schedule = 1 - beta_schedule
        self.alpha_hat_schedule = torch.cumprod(alpha_schedule, dim=0)

    def sample_multiple_timesteps(self, batch_size, n_timesteps):
        """
        Samples batch_size integer subsets of [0, T[ of size n_timesteps in increasing order
        """

        sampling = torch.randn(batch_size, self.num_diffusion_steps)
        sampled_index = torch.argsort(sampling, dim=1)[:, :n_timesteps]
        sampled_index = sampled_index.sort(dim=1, descending=True).values
        return sampled_index
    
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
        self.ref['masks'] = self.model(
            pixel_values=self.ref['images'], 
            noised_pixel_values=self.ref['images'], 
            noise=self.ref['noises'],
            lengths=self.ref['lengths'], 
        ).mask.unsqueeze(2)

    @torch.no_grad()
    def add_noise(self, pixel_values, mask, t):
        """
        Add noise to the pixel_values patches corresponding to mask == 1, according to noise
        corresponding to (batched) timesteps 't'
        """

        mask = mask.unsqueeze(-1)
        
        alpha_hat = self.alpha_hat_schedule.to(self.device)
        w1, w2 = torch.sqrt(alpha_hat), torch.sqrt(1 - alpha_hat)
        w1, w2 = w1[t], w2[t]

        pixel_values_noised = w1[..., None, None, None] * pixel_values + w2[..., None, None, None] * torch.randn_like(pixel_values, device=self.device)

        pixel_values_partially_noised = mask * self.model.patchify(pixel_values_noised) + (1 - mask) * self.model.patchify(pixel_values)
        pixel_values_partially_noised = self.model.unpatchify(pixel_values_partially_noised)
        pixel_values_partially_noised = pixel_values_partially_noised.clip(-1, 1)
        return pixel_values_partially_noised
    
    def denoise(self, pixel_values, lengths, noise=None, t_seq=None):
        """
        From the encoder outputs, performs reverse diffusion process according to (batched) 't_seq'
        noise sequence, or with the full noise schedule if 'timesteps' is not specified.
        """

        #encode the visible patches
        encoder_outputs = self.model.vit(pixel_values, noise=noise, lengths=lengths)
        hidden_states = encoder_outputs["last_hidden_state"]
        mask = encoder_outputs["mask"]
        ids_restore = encoder_outputs["ids_restore"]

        #get patch-normalized target and patch moments
        target, patch_means, patch_vars = self.norm_patches(pixel_values)

        #when the time schedule is not specified, fallback to the full diffusion time schedule
        if t_seq is None:
            # t_seq = torch.arange(0, self.num_diffusion_steps, device=self.device).tile([hidden_states.shape[0], 1])
            t_seq = torch.arange(self.num_diffusion_steps - 1, -1, step=-1, device=self.device).tile([hidden_states.shape[0], 1])

        loss = []
        for diff_idx in range(t_seq.size(1)):
            t = t_seq[:, diff_idx]
            
            #forward diffusion
            pixel_values = self.add_noise(pixel_values, mask, t)

            #backward diffusion
            decoder_outputs = self.model.decoder(
                hidden_states=hidden_states, 
                lengths=lengths,
                ids_restore=ids_restore, 
                noised_pixel_values=pixel_values
                )
            logits = decoder_outputs["logits"] #logits are pach-normalized

            #add contribution of the diff_idx step to the loss
            loss.append(self.model.forward_loss(target, logits, mask))

            #merge visible and reconstructed patches as input for the next iteration
            if self.config.norm_pix_loss: #rescale the predictions if needed
                logits = patch_means + logits * (patch_vars + 1.0e-6) ** 0.5
            pixel_values = self.get_mixed_image(pixel_values, logits, mask.unsqueeze(-1))
            # pixel_values = mixed_image.detach()

        full_loss = torch.stack(loss).mean()

        return pixel_values, full_loss

    def forward(self, pixel_values, lengths, noise=None):
        batch_size = pixel_values.size(0)
        t = self.sample_multiple_timesteps(batch_size, self.num_sparse_diffusion_steps)
        
        logits, loss = self.denoise(pixel_values, lengths, noise=noise, t_seq=t)
        return {
            "logits": logits,
            "loss": loss
        }

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        pixel_values, lengths = batch['images'], batch['lengths']
        
        #add reconstruction loss corresponding to the full diffusion process
        _, loss = self.denoise(pixel_values, lengths, t_seq=None)
        self.log('valid_loss', loss, on_step=False, on_epoch=True)
    
    @torch.no_grad()
    def on_validation_epoch_end(self):
        if not hasattr(self, "ref"):
            return

        #perform forward-reverse diffusion on the whole diffusion schedule
        images = self.ref['images']
        logits, valid_loss = self.denoise(images, lengths=self.ref['lengths'], noise=self.ref['noises'])

        if self.config.norm_pix_loss: #comment this to save rescaled reconstructions 
            images, patch_means, patch_vars = self.norm_patches(images)
            logits, logits_means, logits_vars = self.norm_patches(logits)

        mixed_image = self.get_mixed_image(images, self.model.patchify(logits), self.ref['masks'])
        self.log_reconstructed_images(mixed_image, lengths=self.ref['lengths'])

        #perform linear classification to evaluate learned study representations
        if self.probe_every % 10 == 0 or self.current_epoch == self.trainer.max_epochs - 1:
            self.linear_probe()
            self.probe_every = 0
        else:
            self.probe_every +=1

def pretrain(CFG, args, dataloaders):
    
    available_schemes = [
        "vanilla", 
        "diffusion"
        ]
    if CFG.pretraining_scheme not in available_schemes:
        raise NotImplementedError(
    f"The method '{CFG.pretraining_scheme}' is not implemented.\n Available methods are: [{', '.join(available_schemes)}]"
    )
    
    checkpoint_path = pjoin("checkpoints", CFG.name)
    print(f'Did not find reference pretrain encoder weights at path {checkpoint_path}. Running pretrain.')

    train_dl, valid_dl, test_dl = dataloaders

    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_path,
        save_top_k=1,
        verbose=True, 
        monitor='valid_loss', 
        mode='min'
    )

    trainer = pl.Trainer(
        strategy="ddp", 
        accelerator="gpu", devices=args.device,
        callbacks=[
            LearningRateMonitor(logging_interval='step'),
            checkpoint_callback, 
            ],
        max_epochs=CFG.epochs,
        num_sanity_val_steps=-1,
        check_val_every_n_epoch=10,
        profiler="simple"
        )

    if CFG.pretraining_scheme == "vanilla":
        model = VanillaPretrainingModel(CFG=CFG)
    elif CFG.pretraining_scheme == "diffusion":
        model = DiffusionPretrainingModel(CFG=CFG)
    
    trainer.fit(
        model=model, 
        train_dataloaders=train_dl,
        val_dataloaders=valid_dl,
        )
    
    torch.save(model.model.state_dict(), f"checkpoints/{CFG.name}/weights.pth")
    torch.save(model.model.vit.state_dict(), f"checkpoints/{CFG.name}/backbone_weights.pth")