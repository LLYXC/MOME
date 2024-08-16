from data.util import get_dataset
import argparse
import random
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score,\
    recall_score, f1_score, average_precision_score, roc_curve
import os
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

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', type=str, default='exp', help="name of this experiment")
    parser.add_argument('--seed', type=int, default=1, help="1, 2, 3, or any other seeds")
    parser.add_argument('--data_root', type=str, default='/project/medimgfmod/Breast_MRI/DS1', help="root of data")
    parser.add_argument('--modal', type=str, default='DCE', help="Choose modal for experiment.")
    parser.add_argument('--bs', type=int, default=32, help="Batch size.")
    parser.add_argument('--device', type=int, default=0, help="Choose gpu device.")
    parser.add_argument('--log_dir', type=str, default='./log', help="Address to store the log files.")
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
    
    parser.add_argument('--task', type=str, default='diagnosis', help="diagnosis or treatment")
    parser.add_argument('--split', default='test', type=str)
    parser.add_argument('--save_csv', action='store_true', help='True for saving a csv file.')
    args = parser.parse_args()

    # ------------------------------------ seed, device, log ------------------------------------- #
    seed = args.seed
    seed = 1
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
    
    print('Device: {}'.format(args.device))
    print('Experiment: {}'.format(exp_name))
    print('Seed: {}'.format(seed))

    # ------------------------------------ dataloader ------------------------------------- #
    
    test_dataset = get_dataset(
        args.modal,
        dataset_split=args.split,
        transform_split="test",
        task=args.task,
        root=args.data_root
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.bs,
        shuffle=False,
        num_workers=1,
        pin_memory=False,
        #persistent_workers=True,    # set this so that the workers won't be kiiled every epoch
    )

    # ------------------------------------ model and optimizer ------------------------------------- #
    num_classes = 2
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
    #utils.load_model_and_may_interpolate(args.finetune, model, args.model_key, args.model_prefix)
    utils.load_model_and_may_interpolate_from_trained_model(args.finetune, model, args.model_key, args.model_prefix)
    
    model = model.to(device)
    
    model.eval()
                
    
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
    
    # ------------------------------------- test ----------------------------------------#
    te_acc, te_ap, te_auc, te_tp, te_fp, te_tn, te_fn, te_tpr, te_tnr, te_ppv, te_npv, te_f1, te_probs, te_preds, te_gts, te_idxs = eval(model, test_loader)

    print('Accuracy:', te_acc)
    print('Average Precision:', te_ap)
    print('AUC:', te_auc)
    print('True Positive Rate:', te_tpr)
    print('True Negative Rate:', te_tnr)
    print('Positive Predictive Value:', te_ppv)
    print('Negative Predictive Value:', te_npv)
    print('F1 Score:', te_f1)
    print('True Positives:', te_tp)
    print('False Positives:', te_fp)
    print('True Negatives:', te_tn)
    print('False Negatives:', te_fn)
    #----------------------------- save model and results, do test ------------------------------#
    result_path = os.path.join(log_dir, "result", exp_name)
    te_csv_path = os.path.join(result_path, "{}_result.csv".format(args.split))
    df_te = test_dataset.df
    df_te['Probability'] = pd.Series(te_probs, index=te_idxs)
    df_te['Prediction'] = pd.Series(te_preds, index=te_idxs)
    df_te['GT'] = pd.Series(te_gts, index=te_idxs)
    
    if args.save_csv:
        df_te.to_csv(te_csv_path, index=False, encoding='utf-8_sig')


if __name__ == '__main__':
    train()