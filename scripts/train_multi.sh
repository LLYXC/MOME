# # sparse-soft moe on diagnosis
python train_beit_multimodal.py --modal Multi --optimizer_tag Adam\
       --vis_embed_norm IN --model beit3_multimodal_adapter_base_patch16_224 --drop_path 0.15 --finetune path/to/beit3/beit3_large_patch16_224.pth \
       --bs 1 --lr 1e-4 --lowest_lr 1e-6 --num_epochs 100 --seed 1 --device 0 --task diagnosis --exp_name Multimodal-syn_aug-smoe
