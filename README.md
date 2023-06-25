# Know Your Orientation: A Viewpoint-Aware Framework for Polyp Segmentation
The official implementation of [PPFormer](https://link.springer.com/chapter/10.1007/978-3-031-16440-8_60) (MICCAI 2022) and VANet (Pending).  
## 1. Introduction
## 1.1 Background
Automatic polyp segmentation is a challenging task due to two reasons:  
(i) The viewpoint variations presented in colonoscopy images, leading to diverse visual appearance of polyps (as shown in Figure 1).  
(ii) The camouflage property of polyps poses difficulties in polyp boundary determination.  
<p align="center">
    <img src="./figures/Orientation.png"/ width="400"> <br />
    <em> 
    Figure 1. Illustration of various viewpoints in colonoscopy images caused by different orientations of the colonoscope tip.
    </em>
</p>

To overcome these issues, we present a novel framework, named viewpoint-aware network (VANet), that improves polyp segmentation performance by effectively using the viewpoint variations in colonoscopy images. Our motivation stems from the observation that during a colonoscopy, clinicians steer the colonoscope according to the position of the central lumen; when possible lesions are encountered, they reorient the colonoscope for detailed information. Therefore, we argue that the central lumen and polyps are two key characteristics for distinguishing viewpoints in colonoscopy images. Thus, a viewpoint classifier ([ResNet](https://openaccess.thecvf.com/content_cvpr_2016/html/He_Deep_Residual_Learning_CVPR_2016_paper.html) used here) can accurately capture the polyps via Grad-CAM++.

### 1.2. Framework
The overview of our work is shown in Figure 3. Given an input image, VANet first employs the cue collector to capture potential polyp locations. The location cue is then fed into the VAFormer layers of the encoder to improve attention calculation. VANet's encoder comprises three encoder blocks, where each block is formed by alternating VAFormer and Transformer layers. In the decoding stage, VANet generates prediction maps at different levels and uses them to guide self-attention in BAFormer layers for refinement. We use Convolution Vision Transformer ([CvT](https://openaccess.thecvf.com/content/ICCV2021/html/Wu_CvT_Introducing_Convolutions_to_Vision_Transformers_ICCV_2021_paper.html) as a backbone to construct the encoder-decoder, which integrates convolution operations in Vision Transformer, enabling our model to extract both global and local features. In addition, the removal of position encoding in CvT supports VANet with flexible inputs.
<p align="center">
    <img src="./figures/VANet.png"/ width="800"> <br />
    <em> 
    Figure 2. Overview of VANet.
    </em>
</p>

### 1.3 Performance
VANet achieves the state-of-the-art performance on five polyp segmentation datasets ([benchmark](https://github.com/DengPingFan/PraNet/blob/master)), especially on the unseen datasets. Additionally, VANet is capable of calibrating misaligned predictions and precisely determining polyp boundary.
<p align="center">
    <img src="./figures/Qualitative.png"/ width="800"> <br />
    <em> 
    Figure 3. Qualitative comparison of the state-of-the-art polyp segmentation algorithms.
    </em>
</p>

<p align="center">
    <img src="./figures/Table3.png"/ width="800"> <br />
    <em> 
    Table 1. Quantative comparison of the state-of-the-art polyp segmentation algorithms.
    </em>
</p>

## 2. Quick start
