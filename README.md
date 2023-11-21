# RadDiffusion: high-resolution representation learning on MIMIC-CXR radiology images
*Master Thesis, University of Geneva - Digital Health department.*

\
[MIMIC-CXR](https://physionet.org/content/mimic-cxr/2.0.0/) is a radiology studies dataset, each study being composed of 2 to 5 different views of highly structured chest images. \
Using a single *bottleneck-cls* token [1] for inter-image intra-study information flow, we train a ViT encoder to produce representations at the study level. We explore two Masked Image Modeling pretraining schemes:
- using the vanilla Masked AutoEncoder [2], which converges prematurely to blurry reconstructions,
- using a diffusion variant of Masked AutoEncoder [3], which converges slow and steady to detailed reconstructions.
### Reconstructions

Qualitatively, we can observe the reconstructions with a masking ratio of 0.5 for the two pretraining schemes:

- Vanilla MAE scheme: smooth yet undetailed reconstructions
<p align="center">
  <img src="https://github.com/AugustinCombes/RadDiffusion/blob/main/images/vanilla_study_reconstruction.jpg" alt="Study vanilla reconstruction" width="828" height="256"/>
  <br align="center"> 
    <span style="font-size: larger;">
      Study vanilla reconstruction
    </span>
  </br>
</p>

- Diffusion MAE scheme: detailed yet noisy reconstructions
<p align="center">
  <img src="https://github.com/AugustinCombes/RadDiffusion/blob/main/images/backward_diffusion_xs_study.gif" alt="Study backward diffusion" width="828" height="256"/>
  <br align="center"> 
    <span style="font-size: larger;">
      Study-wise backward diffusion
    </span>
  </br>
</p>


### Evaluation
We systematically evaluate the quality of the learned representation for both pretraining schemes via linear probing and supervised fine-tuning, using multi-label binary classification for the 14 classes defined by the CheXpert article [4].

#### Linear probing
<table>
  <tr>
    <td>
      <p>
        We evaluate the predictive capacity of the bottleneck-cls pooled study representations during pretraining, using an 'xs' ViT encoder. 
      </p>
      <p>
        We observe extracting study features from the model pretrained using the diffusion-MAE scheme induce greater predictive performances. 
      </p>
      <p>
        Overall, it can be argued the vanilla-MAE scheme tends to converges faster to lower performances as it gets stucked in trivial image reconstructions, while the diffusion-MAE scheme steadily improve the patch reconstructions' quality, allowing for final study representations to capture finer-grained details, and eventually reaching higher predictive performances.
      </p>
    </td>
    <td>
      <img src="https://github.com/AugustinCombes/RadDiffusion/blob/main/images/linear_probings.png" alt="Linear probing vs. pretraining epochs)" alt="Your Image Description" width="5500"/>
    </td>
  </tr>
</table>

#### Fine-tuning

[TBD]


### Run the code

Download the MIMIC-CXR database and preprocess it:
```python
python data/image_preprocessing.py
```

Run pretraining and supervised inference:
```python
python main.py --ref_name "vits" --num_workers 4 --encoder_size xs --device [0, 1, 2] --scheme diffusion
```

#### Bibliography
[1] [**Attention Bottlenecks for Multimodal Fusion**](https://ar5iv.org/abs/2107.00135) (2021), *Arsha Nagrani, Shan Yang, Anurag Arnab, Aren Jansen, Cordelia Schmid, Chen Sun*

[2] [**Masked Autoencoders Are Scalable Vision Learners**](https://arxiv.org/abs/2111.06377) (2021), *Kaiming He, Xinlei Chen, Saining Xie, Yanghao Li, Piotr Dollár, Ross Girshick*

[3] [**Diffusion Models as Masked Autoencoders**](https://ar5iv.org/abs/2304.03283) (2023), *Chen Wei, Karttikeya Mangalam, Po-Yao Huang, Yanghao Li, Haoqi Fan, Hu Xu, Huiyu Wang, Cihang Xie, Alan Yuille, Christoph Feichtenhofer*

[4] [**CheXpert: A Large Chest Radiograph Dataset with Uncertainty Labels and Expert Comparison**](https://arxiv.org/abs/2107.00135) (2019), *Jeremy Irvin, Pranav Rajpurkar, Michael Ko, Yifan Yu, Silviana Ciurea-Ilcus, Chris Chute, Henrik Marklund, Behzad Haghgoo, Robyn Ball, Katie Shpanskaya, et al.*
