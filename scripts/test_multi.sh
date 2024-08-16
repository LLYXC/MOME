# #  DS2
# python test_beit_multimodal.py --modal Multi --model beit3_multimodal_adapter_base_patch16_224 --drop_path 0.15 \
#        --finetune /home/cseluyang/BC/MRI/2324/code/MultiSequence_Beit/log/result/model.th \
#        --model_key 'state_dict' --bs 1 --seed 1 \
#        --device 0 --exp_name Multimodal-syn_aug-smoe-rpt4 --data_root /project/medimgfmod/Breast_MRI/DS2 --split SZRM_batch_2 #--save_csv 

# additional_test, test
# python test_beit_multimodal.py --modal Multi --model beit3_multimodal_adapter_base_patch16_224 --drop_path 0.15 \
#        --finetune /home/cseluyang/BC/MRI/2324/code/MultiSequence_Beit/log/result/Multimodal-syn_aug-smoe-rpt4/model.th \
#        --model_key 'state_dict' --bs 1 --seed 1 \
#        --device 0 --exp_name Multimodal-syn_aug-smoe-rpt4 --data_root /project/medimgfmod/Breast_MRI/DS1 --split additional_test #--save_csv 

#python test_beit_multimodal.py --modal Multi --model beit3_multimodal_adapter_base_patch16_224 --drop_path 0.15 \
#       --finetune /home/cseluyang/BC/MRI/2324/code/MultiSequence_Beit/log/result/Multimodal-syn_aug-smoe-rpt4/model.th \
#       --model_key 'state_dict' --bs 1 --seed 1 \
#       --device 0 --exp_name Multimodal-syn_aug-smoe-rpt4 --data_root /project/medimgfmod/Breast_MRI/DS1 --split test #--save_csv 

# ACRIN
python test_beit_multimodal.py --modal Multi --model beit3_multimodal_adapter_base_patch16_224 --drop_path 0.15\
       --finetune /home/cseluyang/BC/MRI/2324/code/MultiSequence_Beit/log/result/Multimodal-syn_aug-smoe-rpt4-rpt/model.th --model_key 'state_dict' \
       --model_key 'state_dict' --bs 1 --seed 1 --exp_name Multimodal-syn_aug-smoe-rpt4 \
       --device 0 --task treatment --data_root /scratch/medimgfmod/Breast_MRI/TreatmentResponse --split ACRIN_T3 #--save_csv
