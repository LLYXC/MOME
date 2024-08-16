---
# For reference on model card metadata, see the spec: https://github.com/huggingface/hub-docs/blob/main/modelcard.md?plain=1
# Doc / guide: https://huggingface.co/docs/hub/model-cards
{{ card_data }}
---

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

## Citation: If you found our paper useful, please consider cite the following:

**BibTeX:**

 @article{luo2024towards,
 title={Towards Non-invasive and Personalized Management of Breast Cancer Patients from Multiparametric MRI via A Large Mixture-of-Modality-Experts Model},
 author={Luo, Luyang and Wu, Mingxiang and Li, Mei and Xin, Yi and Wang, Qiong and Vardhanabhuti, Varut and Chu, Winnie CW and Li, Zhenhui and Zhou, Juan and Rajpurkar, Pranav and Chen, Hao},
 year={2024}
 }