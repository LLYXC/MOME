# Code repo for paper: Towards Non-invasive and Personalized Management of Breast Cancer Patients from Multiparametric MRI via A Large Mixture-of-Modality-Experts Model

![Alt text](figures/Framework.jpg "Overall Framework of MOME")

# Model Card for MOME

<!-- Provide a quick summary of what the model is/does. -->

MOME conducts multimodel fusion and classification based on multi-sequence 3D medical data, e.g., multiparametric breast MRI.

## Model Details

### Model Description

<!-- Provide a longer summary of what this model is. -->


- **Developed by:** Luyang Luo
- **Model type:** Transformer (based on BEiT3)
- **License:** MIT
- **Finetuned from model :** BEiT-3
- **Repository:** https://github.com/LLYXC/MOME
- **Paper [optional]:** Towards Non-invasive and Personalized Management of Breast Cancer Patients from Multiparametric MRI via A Large Mixture-of-Modality-Experts Model

## Uses

<!-- Address questions around how the model is intended to be used, including the foreseeable users of the model and those affected by the model. -->
The training and testing commands are provided in ./scripts

## Citation
### If you found our paper useful, please consider cite the following:
```
@article{luo2024towards,
title={Towards Non-invasive and Personalized Management of Breast Cancer Patients from Multiparametric MRIvia A Large Mixture-of-Modality-Experts Model},
author={Luo, Luyang and Wu, Mingxiang and Li, Mei and Xin, Yi and Wang, Qiong and Vardhanabhuti, Varut andChu, Winnie CW and Li, Zhenhui and Zhou, Juan and Rajpurkar, Pranav and Chen, Hao},
year={2024}
}
```

### As our code is based on BEiT-3 and soft MOE, we also recommend cite the following works:
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
