from data.util import get_dataset
import argparse
import random
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score,\
    recall_score, f1_score, average_precision_score, roc_curve
import os
import shutil
from tqdm import tqdm
import pandas as pd

import torch
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, WeightedRandomSampler
import torch.optim.lr_scheduler as lr_scheduler
import utils
import modeling_finetune
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='diagnosis', help="diagnosis or treatment")
    parser.add_argument('--exp_name', type=str, default='exp', help="name of this experiment")
    parser.add_argument('--seed', type=int, default=1, help="1, 2, 3, or any other seeds")
    parser.add_argument('--data_root', type=str, default='/project/medimgfmod/Breast_MRI/DS1', help="root of data")
    parser.add_argument('--optimizer_tag', type=str, default='Adam', help="Choose a optimizer.")
    parser.add_argument('--modal', type=str, default='DCE', help="Choose modal for experiment.")
    parser.add_argument('--bs', type=int, default=32, help="Batch size.")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate.")
    parser.add_argument('--lowest_lr', type=float, default=1e-6, help="Lowest learning rate.")
    parser.add_argument('--wd', type=float, default=5e-4, help="Weight decay.")
    parser.add_argument('--num_epochs', type=int, default=30, help="Number of epochs.")
    parser.add_argument('--device', type=int, default=0, help="Choose gpu device.")
    parser.add_argument('--log_dir', type=str, default='./log', help="Address to store the log files.")
    parser.add_argument('--debug', action='store_true', help='False for saving result or Ture for not saving result.')
    parser.add_argument('--vis_embed_norm', default='IN', type=str, help='Norm used in visual embedding.')
    
    # Model parameters
    parser.add_argument('--model', default='beit_base_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--checkpoint_activations', action='store_true', default=None, 
                        help='Enable checkpointing to save your memory.')
    parser.add_argument('--vocab_size', type=int, default=64010)
    # Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--model_key', default='model|module', type=str)
    parser.add_argument('--model_prefix', default='', type=str)
    args = parser.parse_args()

    # ------------------------------------ seed, device, log ------------------------------------- #
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    device = torch.device(args.device)

    exp_name = args.exp_name
    log_dir = args.log_dir
    
    if not args.debug:
        print('saving the result.')
        if args.task == 'diagnosis':
            writer = SummaryWriter(os.path.join(log_dir, "summary", exp_name))
        elif args.task == 'treatment':
            writer = SummaryWriter(os.path.join(log_dir, "summary_treatment", exp_name))
        elif args.task == 'subtyping':
            writer = SummaryWriter(os.path.join(log_dir, "summary_subtyping", exp_name))
        os.makedirs(os.path.join(log_dir, "bk", exp_name), exist_ok=True)
        shutil.copyfile('./scripts/train_multi.sh', os.path.join(log_dir, "bk", exp_name, 'train_multi.sh'))
        #shutil.copyfile('./*.py', os.path.join(log_dir, "bk", exp_name, 'train_multi.sh'))

    print('Device: {}'.format(args.device))
    print('Experiment: {}'.format(exp_name))
    print('Seed: {}'.format(seed))

    # ------------------------------------ dataloader ------------------------------------- #
    train_dataset = get_dataset(
        args.modal,
        dataset_split="train",
        transform_split="train",
        task=args.task,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bs,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        #persistent_workers=True,    # set this so that the workers won't be kiiled every epoch
    )
    
    valid_dataset = get_dataset(
        args.modal,
        dataset_split="val",
        transform_split="val",
        task=args.task,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        #persistent_workers=True,    # set this so that the workers won't be kiiled every epoch
    )
    
    test_dataset = get_dataset(
        args.modal,
        dataset_split="test",
        transform_split="test",
        task=args.task,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        #persistent_workers=True,    # set this so that the workers won't be kiiled every epoch
    )
    
    if args.task == 'diagnosis':
        atest_dataset = get_dataset(
            args.modal,
            dataset_split="additional_test",
            transform_split="test",
            task=args.task,
        )
        atest_loader = DataLoader(
            atest_dataset,
            batch_size=args.bs,
            shuffle=False,
            num_workers=4,
            pin_memory=False,
            #persistent_workers=True,    # set this so that the workers won't be kiiled every epoch
        )

    # ------------------------------------ model and optimizer ------------------------------------- #
    num_classes = 2
    if args.task == 'subtyping':
        num_classes = 4
    img_size_dce=(384, 256, 128)
    in_chans_dce=6
    img_size_dwi=(256, 128, 32)
    in_chans_dwi=1
    img_size_t2=(384, 256, 48)
    in_chans_t2=1
    
    from timm.models import create_model
    model_config = "%s_imageclassification" % args.model    # args.model = 'beit3_multimodal_adapter_base_patch16_224'
    model = create_model(
        model_config,
        pretrained=False,
        drop_path_rate=args.drop_path,
        vocab_size=args.vocab_size,
        checkpoint_activations=args.checkpoint_activations,
        num_classes=num_classes,
        img_size_dce=img_size_dce,
        in_chans_dce=in_chans_dce,
        img_size_dwi=img_size_dwi,
        in_chans_dwi=in_chans_dwi,
        img_size_t2=img_size_t2,
        in_chans_t2=in_chans_t2,
        vis_embed_norm=args.vis_embed_norm,
    )
    utils.load_model_and_may_interpolate(args.finetune, model, args.model_key, args.model_prefix)
    model = model.to(device)

    for p in model.parameters():    
        p.requires_grad = False
    for p in model.beit3.dce_embed.parameters():
        p.requires_grad = True
    for p in model.beit3.dwi_embed.parameters():
        p.requires_grad = True
    for p in model.beit3.t2_embed.parameters():
        p.requires_grad = True
    for p in model.beit3.encoder.embed_positions.parameters():
        p.requires_grad = True
    for p in model.fc_norm.parameters():
        p.requires_grad = True
    for p in model.head.parameters():
        p.requires_grad = True
        
    #for n, m in model.beit3.named_modules():
    #    if 'S_Adapter' in n:
    #        for p in m.parameters():
    #            p.requires_grad = True
            
    for n, m in model.beit3.named_modules():
        if 'MLP_Adapter' in n:
            for p in m.parameters():
                p.requires_grad = True
                
    # for n, m in model.beit3.named_modules():
    #     if ('9.MLP_Adapter' in n) or ('10.MLP_Adapter' in n) or ('11.MLP_Adapter' in n):
    #         for p in m.parameters():
    #             p.requires_grad = True
    #     if getattr(m, 'is_smoe_adapter', False):
    #         print(n)
    #         m.reset_parameters()    
    
    #with open('model_architecture.txt','w') as f:
    #    print(model, file=f)
    #    f.close()
            
    def get_trainable_parameters(model):    
        trainable_params = 0
        all_param = 0
        for _, param in model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        print(
            f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
        )
    get_trainable_parameters(model)
                
    def get_optimizer(optimizer_tag, model, lr, weight_decay):
        if optimizer_tag == 'Adam':
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_tag == 'AdamW':
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=(0.9, 0.999)
            )
        elif optimizer_tag == 'SGD':
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9
            )
        return optimizer
    
    optimizer = get_optimizer(args.optimizer_tag, model, args.lr, args.wd)
    
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=args.lowest_lr)
    
    # -------------------------------------- evalaution --------------------------------------- #
    def eval(model, data_loader):
        model.eval()
        gts = torch.LongTensor().to(device)
        probs = torch.FloatTensor().to(device)
        preds = torch.FloatTensor().to(device)
        indices = torch.LongTensor().to(device)
        for index, t2, dwi, dce, label in tqdm(data_loader, leave=False):
            t2 = t2.to(device)
            dwi = dwi.to(device)
            dce = dce.to(device)
            label = label.to(device)
            index = index.to(device)
            with torch.no_grad():
                logit = model(dce, dwi, t2)
                prob = torch.softmax(logit, dim=1)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                
            gts = torch.cat((gts, label), 0)
            probs = torch.cat((probs, prob[:, 1]), 0)  # prob[1] for malignancy
            preds = torch.cat((preds, pred), 0)
            indices = torch.cat((indices, index), 0)

        gts_numpy = gts.cpu().detach().numpy()
        probs_numpy = probs.cpu().detach().numpy()
        preds_numpy = preds.cpu().detach().numpy()
        indices_numpy = indices.cpu().detach().numpy()
        tp = np.sum((preds_numpy == 1) & (gts_numpy == 1))
        fp = np.sum((preds_numpy == 1) & (gts_numpy == 0))
        tn = np.sum((preds_numpy == 0) & (gts_numpy == 0))
        fn = np.sum((preds_numpy == 0) & (gts_numpy == 1))
        
        accs = (tp+tn)*1./len(gts_numpy)
        tpr = tp*1. / (gts_numpy == 1).sum()   # recall, sensitivity
        tnr = tn*1. / (gts_numpy == 0).sum()   # specificity    
        ppv = tp*1. / (tp+fp)  #precision
        npv = tn*1. / (tn+fn)
        f1 = tp*2./(tp*2.+fp+fn)

        aps = average_precision_score(gts_numpy, probs_numpy)
        aucs = roc_auc_score(gts_numpy, probs_numpy)
        model.train()
        return accs, aps, aucs, tp, fp, tn, fn, tpr, tnr, ppv, npv, f1, probs_numpy, preds_numpy, gts_numpy, indices_numpy

    # ----------------------------------- loss function -----------------------------------  #
    # P = (train_dataset.labels==1).sum()
    # N = (train_dataset.labels==0).sum()
    # weight = torch.stack([P/(P+N), N/(P+N)])*num_classes
    # weight = weight.to(device)
    # criterion = torch.nn.CrossEntropyLoss(weight=weight,reduction='none')
    # print('Using weighted cross entropy')
    
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    # ----------------------------------- start training ------------------------------------------- #
    if not args.debug:
        if args.task == 'diagnosis':
            result_path = os.path.join(log_dir, "result", exp_name)
        elif args.task == 'treatment':
            result_path = os.path.join(log_dir, "result_treament", exp_name)
        elif args.task == 'subtyping':
            result_path = os.path.join(log_dir, "result_subtyping", exp_name)
        os.makedirs(result_path, exist_ok=True)
        model_path = os.path.join(result_path, "model.th")
        val_csv_path = os.path.join(result_path, "val_result.csv")
        test_csv_path = os.path.join(result_path, "test_result.csv")
        atest_csv_path = os.path.join(result_path, "additional_test_result.csv")
    best_auc = 0
    
    
    def train_one_epoch():
        for iter_num, (_, t2, dwi, dce, label) in tqdm(enumerate(train_loader)):
            step = epoch*len(train_loader) + iter_num

            t2 = t2.to(device)
            dwi = dwi.to(device)
            dce = dce.to(device)
            label = label.to(device)
            
            logit = model(dce, dwi, t2)
            
            loss = criterion(logit, label).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            main_log_freq = 10
            if step % main_log_freq == 0:
                if not args.debug:
                    writer.add_scalar('loss', loss.detach().cpu(), step)
                    
    def train_one_epoch_w_accumulation():
        num_accumulations = 32
        num_ = 0
        num__ = 0
        
        for iter_num, (_, t2, dwi, dce, label) in tqdm(enumerate(train_loader)):
            #num_ += 1
            #num__ += 1
            num_accu = num_accumulations
            #if num__ > len(train_loader)//num_accumulations*num_accumulations:
            #    num_accu = len(train_loader) - len(train_loader)//num_accumulations*num_accumulations
                
            step = epoch*len(train_loader) + iter_num

            t2 = t2.to(device)
            dwi = dwi.to(device)
            dce = dce.to(device)
            label = label.to(device)
            
            logit = model(dce, dwi, t2)
            loss = criterion(logit, label).mean() / num_accu
            
            loss.backward()
            
            if (iter_num+1) % num_accumulations == 0:
                optimizer.step()
                for param in model. parameters():
                    param. grad = None
                #num_=0
                
            main_log_freq = 10
            if step % main_log_freq == 0:
                if not args.debug:
                    writer.add_scalar('loss', loss.detach().cpu()*num_accu, step)
    
    for epoch in range(args.num_epochs):
        
        train_one_epoch_w_accumulation()
             
            #if epoch == 15:        
            #    for param_group in optimizer.param_groups:
            #        param_group['lr'] = args.lr / 10
            #    print('Change lr to', args.lr / 10)

        if not args.debug:
            # ------------------------------------- validation ----------------------------------------#
            val_acc, val_ap, val_auc, val_tp, val_fp, val_tn, val_fn, val_tpr, val_tnr, val_ppv, val_npv, val_f1, val_probs, val_preds, val_gts, val_idxs = eval(model, valid_loader)
            writer.add_scalar("val/1_auc", val_auc, epoch)
            writer.add_scalar("val/2_acc", val_acc, epoch)
            writer.add_scalar("val/3_ap", val_ap, epoch)
            writer.add_scalar("val/4_f1", val_f1, epoch)
            writer.add_scalar("val/5_tpr", val_tpr, epoch)
            writer.add_scalar("val/6_tnr", val_tnr, epoch)
            writer.add_scalar("val/7_ppv", val_ppv, epoch)
            writer.add_scalar("val/8_npv", val_npv, epoch)
            writer.add_scalar("val/9_tp", val_tp, epoch)
            writer.add_scalar("val/10_fp", val_fp, epoch)
            writer.add_scalar("val/11_tn", val_tn, epoch)
            writer.add_scalar("val/12_fn", val_fn, epoch)
            
            # ------------------------------------- test ----------------------------------------#
            te_acc, te_ap, te_auc, te_tp, te_fp, te_tn, te_fn, te_tpr, te_tnr, te_ppv, te_npv, te_f1, te_probs, te_preds, te_gts, te_idxs = eval(model, test_loader)
            writer.add_scalar("test/1_auc", te_auc, epoch)
            writer.add_scalar("test/2_acc", te_acc, epoch)
            writer.add_scalar("test/3_ap", te_ap, epoch)
            writer.add_scalar("test/4_f1", te_f1, epoch)
            writer.add_scalar("test/5_tpr", te_tpr, epoch)
            writer.add_scalar("test/6_tnr", te_tnr, epoch)
            writer.add_scalar("test/7_ppv", te_ppv, epoch)
            writer.add_scalar("test/8_npv", te_npv, epoch)
            writer.add_scalar("test/9_tp", te_tp, epoch)
            writer.add_scalar("test/10_fp", te_fp, epoch)
            writer.add_scalar("test/11_tn", te_tn, epoch)
            writer.add_scalar("test/12_fn", te_fn, epoch)
            
            if (val_auc >= best_auc):
                if args.task == 'diagnosis':
                
                    ate_acc, ate_ap, ate_auc, ate_tp, ate_fp, ate_tn, ate_fn, ate_tpr, ate_tnr, ate_ppv, ate_npv, ate_f1, ate_probs, ate_preds, ate_gts, ate_idxs = eval(model, atest_loader)
                    writer.add_scalar("atest/1_auc", ate_auc, epoch)
                    writer.add_scalar("atest/2_acc", ate_acc, epoch)
                    writer.add_scalar("atest/3_ap", ate_ap, epoch)
                    writer.add_scalar("atest/4_f1", ate_f1, epoch)
                    writer.add_scalar("atest/5_tpr", ate_tpr, epoch)
                    writer.add_scalar("atest/6_tnr", ate_tnr, epoch)
                    writer.add_scalar("atest/7_ppv", ate_ppv, epoch)
                    writer.add_scalar("atest/8_npv", ate_npv, epoch)
                    writer.add_scalar("atest/9_tp", ate_tp, epoch)
                    writer.add_scalar("atest/10_fp", ate_fp, epoch)
                    writer.add_scalar("atest/11_tn", ate_tn, epoch)
                    writer.add_scalar("atest/12_fn", ate_fn, epoch)
            
                #----------------------------- save model and results, do test ------------------------------#
                best_auc = val_auc
                state_dict = {
                    'epoch': epoch, 
                    'state_dict': model.state_dict(), 
                    'optimizer': optimizer.state_dict(), 
                 }
                with open(model_path, "wb") as f:
                    torch.save(state_dict, f)  
                
                df_val = valid_dataset.df
                df_val['Probability'] = pd.Series(val_probs, index=val_idxs)
                df_val['Prediction'] = pd.Series(val_preds, index=val_idxs)
                df_val['GT'] = pd.Series(val_gts, index=val_idxs)
                df_val.to_csv(val_csv_path, index=False, encoding='utf-8_sig')
                
                df_test = test_dataset.df
                df_test['Probability'] = pd.Series(te_probs, index=te_idxs)
                df_test['Prediction'] = pd.Series(te_preds, index=te_idxs)
                df_test['GT'] = pd.Series(te_gts, index=te_idxs)
                df_test.to_csv(test_csv_path, index=False, encoding='utf-8_sig')
                
                df_atest = atest_dataset.df
                df_atest['Probability'] = pd.Series(ate_probs, index=ate_idxs)
                df_atest['Prediction'] = pd.Series(ate_preds, index=ate_idxs)
                df_atest['GT'] = pd.Series(ate_gts, index=ate_idxs)
                df_atest.to_csv(atest_csv_path, index=False, encoding='utf-8_sig')

        scheduler.step()


if __name__ == '__main__':
    train()