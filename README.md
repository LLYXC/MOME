<!-- ## Towards Non-invasive and Personalized Management of Breast Cancer Patients from Multiparametric MRI via A Large Mixture-of-Modality-Experts Model-->
![GitHub last commit](https://img.shields.io/github/last-commit/birkhoffkiki/GPFM?style=flat-square)
[![X (formerly Twitter) Follow](https://img.shields.io/twitter/follow/SMARTLab_HKUST%20)](https://x.com/SMARTLab_HKUST)
--- 

# Towards Non-invasive and Personalized Management of Breast Cancer Patients from Multiparametric MRI via A Large Mixture-of-Modality-Experts Model

![Alt text](figures/Framework.jpg "Overall Framework of MOME")

# Model Card for MOME
MOME conducts multimodel fusion and classification based on multi-sequence 3D medical data, e.g., multiparametric breast MRI.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->


- **Developed by:** Luyang Luo
- **Model type:** Transformer (based on BEiT3)
- **License:** MIT
- **Finetuned from model :** BEiT-3
- **Repository:** https://github.com/LLYXC/MOME
- **Paper:** Towards Non-invasive and Personalized Management of Breast Cancer Patients from Multiparametric MRI via A Large Mixture-of-Modality-Experts Model

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->
- **Requirement/dependencies:** Please see the requirement of [BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3).
- **Installation:** The installation will take a few seconds to minutes.
```
git clone https://github.com/LLYXC/MOME.git
mkdir log
```
- **Training and Testing:** The training and testing commands are provided in [scripts](scripts).

## Citation
### If you found our work useful, please consider cite the following:
```
@article{luo2024towards,
title={Towards Non-invasive and Personalized Management of Breast Cancer Patients from Multiparametric MRIvia A Large Mixture-of-Modality-Experts Model},
author={Luo, Luyang and Wu, Mingxiang and Li, Mei and Xin, Yi and Wang, Qiong and Vardhanabhuti, Varut andChu, Winnie CW and Li, Zhenhui and Zhou, Juan and Rajpurkar, Pranav and Chen, Hao},
year={2024}
}
```

### Our work is standing on the sholders of [BEiT-3](https://github.com/microsoft/unilm/tree/master/beit3) and [soft MOE](https://github.com/bwconrad/soft-moe), please also consider cite the following works:
```
@inproceedings{wang2023image,
title={Image as a foreign language: Beit pretraining for vision and vision-language tasks},
author={Wang, Wenhui and Bao, Hangbo and Dong, Li and Bjorck, Johan and Peng, Zhiliang and Liu, Qiang and Aggarwal, Kriti and Mohammed, Owais Khan and Singhal, Saksham and Som, Subhojit and others},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
pages={19175--19186},
year={2023}
}
```
```
@inproceedings{puigcerversparse,
  title={From Sparse to Soft Mixtures of Experts},
  author={Puigcerver, Joan and Ruiz, Carlos Riquelme and Mustafa, Basil and Houlsby, Neil},
  booktitle={The Twelfth International Conference on Learning Representations}
}
```
