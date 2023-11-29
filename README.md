# TIRDet
Source Code for '**TIRDet: Mono-Modality Thermal InfraRed Object Detection Based on Prior Thermal-To-Visible Translation**' 

Accepted by **ACM MM'23**

![image](https://github.com/zeyuwang-zju/TIRDet/assets/112078495/78f6c706-8b9b-4be2-909b-87b778d7a074)

![image](https://github.com/zeyuwang-zju/TIRDet/assets/112078495/8ff7acff-10ad-4e64-b7d4-303ef13c8aef)


This repo highly inherits the **mmdetection** framework.

# Abstract
Cross-modality images that combine visible-infrared spectra can provide complementary information for object detection. In particular, they are well-suited for autonomous vehicle applications in dark environments with limited illumination. However, it is time-consuming to acquire a large number of pixel-aligned visible-thermal image pairs, and real-time alignment is challenging in practical driving systems. Furthermore, the quality of visible-spectrum images can be adversely affected by complex environmental conditions. In this paper, we propose a novel neural network called TIRDet, which only utilizes Thermal InfraRed (TIR) images for mono-modality object detection. To compensate for the lacked visible-band information, we adopt a prior Thermal-To-Visible (T2V) translation model to obtain the translated visible images and the latent T2V codes. In addition, we introduce a novel attention-based Cross-Modality Aggregation (CMA) module, which can augment the modality-translation awareness of TIRDet by preserving the T2V semantic information. Extensive experiments on FLIR and LLVIP datasets demonstrate that our TIRDet significantly outperforms all mono-modality detection methods based on thermal images, and it even surpasses most State-Of-The-Art (SOTA) multispectral methods using visible-thermal image pairs. Code is available at https://github.com/zeyuwang-zju/TIRDet.

# Requirements
- torch=1.9.1 
- torchvision=0.9.1 
- cuda=11.1
- mmdet=2.28.2

# Usage

1. Download the FLIR and LLVIP datasets.
    FLIR: https://www.flir.eu/oem/adas/adas-dataset-form/
  LLVIP: https://bupt-ai-cz.github.io/LLVIP/
2. Prepare the FLIR and LLVIP datasets into Microsoft COCO version.
3. Replace the **''mmdet'' in the environment** with the **''mmdet'' in our repository**
4. Replace the **''configs'' in the environment** with the **''configs'' in our repository**
5. Download the **Pearl-GAN** pretrained weights from https://github.com/FuyaLuo/PearlGAN/. Place them into **configs/tirdet/pearlgan_ckpt/FLIR_NTIR2DC/**.
6. Follow the implementations of **mmdetection** to train and test our model.

# Pretrained Model Weights
[TIRDet-S-FLIR](https://pan.baidu.com/s/1jWAOQuaTo_KI67Cywjj6Yg?pwd=2fgt). Code:2fgt

# Experiments

![image](https://github.com/zeyuwang-zju/TIRDet/assets/112078495/02a03c1b-e473-49e5-97b1-3dd2d312bc8f)

# Citation

If you are interested this repo for your research, welcome to cite our paper:

```
@inproceedings{wang2023tirdet,
  title={TIRDet: Mono-Modality Thermal InfraRed Object Detection Based on Prior Thermal-To-Visible Translation},
  author={Wang, Zeyu and Colonnier, Fabien and Zheng, Jinghong and Acharya, Jyotibdha and Jiang, Wenyu and Huang, Kejie},
  booktitle={Proceedings of the 31st ACM International Conference on Multimedia},
  pages={2663--2672},
  year={2023}
}
```

