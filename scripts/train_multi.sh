## multimodal mri 
#python train_beit_multimodal.py --modal Multi --bs 1 --optimizer_tag Adam\
#       --model beit3_multimodal_adapter_base_patch16_224 --drop_path 0.15 --finetune /home/cseluyang/BC/MRI/2324/beit3-weight/beit3_base_patch16_224.pth \
#       --lr 1e-4 --lowest_lr 1e-6 --num_epochs 200 --seed 1 --device 0 --exp_name adapter-multi-longer

# # sparse-soft moe on diagnosis
python train_beit_multimodal.py --modal Multi --optimizer_tag Adam\
       --vis_embed_norm IN --model beit3_multimodal_adapter_large_patch16_224 --drop_path 0.15 --finetune /home/cseluyang/BC/MRI/2324/beit3-weight/beit3_large_patch16_224.pth \
       --bs 1 --lr 1e-4 --lowest_lr 1e-6 --num_epochs 100 --seed 1 --device 0 --task diagnosis --exp_name Multimodal-syn_aug-smoe-large-2  #Multimodal-syn_aug-smoe-rpt4-rpt

# # sparse-soft moe on treatment response prediction
#python train_beit_multimodal.py --modal Multi --optimizer_tag Adam\
#        --model beit3_multimodal_adapter_base_patch16_224 --drop_path 0.15 --finetune /home/cseluyang/BC/MRI/2324/beit3-weight/beit3_base_patch16_224.pth \
#         --bs 1 --lr 1e-4 --lowest_lr 1e-6 --num_epochs 200 --seed 1 --device 3  --task treatment --exp_name Multimodal-syn_aug-smoe-wce

# python train_beit_multimodal.py --modal Multi --optimizer_tag Adam\
#        --model beit3_multimodal_adapter_base_patch16_224 --drop_path 0.15 \
#        --finetune home/cseluyang/BC/MRI/2324/code/MultiSequence_Beit/log/result/model.th --model_key 'state_dict'\
#        --bs 1 --lr 1e-4 --lowest_lr 1e-6 --num_epochs 100 --seed 1 --device 2 --task treatment --exp_name Multimodal-syn_aug-smoe-dianosis_weight-init_lasterthreelayer-w_ce


# # ---------------- Ablation ----------------- #
# python train_beit_multimodal.py --modal Multi --optimizer_tag Adam\
#        --model beit3_multimodal_adapter_base_patch16_224 --drop_path 0.15 --finetune /home/cseluyang/BC/MRI/2324/beit3-weight/beit3_base_patch16_224.pth \
#        --bs 1 --lr 1e-4 --lowest_lr 1e-6 --num_epochs 100 --seed 1 --device 2 --task diagnosis --exp_name beit3-without_adapter-without_split_attn