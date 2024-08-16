# test
python test_beit_multimodal.py --modal Multi --model beit3_multimodal_adapter_base_patch16_224 --drop_path 0.15 \
       --finetune path/to/finetuned/model/model.th \
       --model_key 'state_dict' --bs 1 --seed 1 \
       --device 0 --exp_name Multimodal-syn_aug-smoe-rpt4 --data_root data/root --split test
