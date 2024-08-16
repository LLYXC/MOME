import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer

def get_csv_file(split, task):
    if split == 'train':
        csv_file = 'train.csv'
        print('Training with: ', csv_file)
    elif split == 'val':
        csv_file = 'val.csv'
        print('Validation: ', csv_file)
    elif split == 'test':
        csv_file = 'test.csv'
        print('Testing: ', csv_file)
    elif split == 'additional_test':
        csv_file = 'additional_test.csv'
        print('Additional testing: ', csv_file)
    elif split == 'SZRM_batch_1':
        csv_file = 'SZRM_batch_1.csv'
        print('External Testing:', csv_file)
    elif split == 'SZRM_batch_2':
        csv_file = 'SZRM_batch_2.csv'
        print('External Testing:', csv_file)
    elif split == 'SZRM_batch_2':
        csv_file = 'SZRM_batch_2.csv'
        print('External Testing:', csv_file)
    elif split == 'YN':
        csv_file = 'YN_diagnosis.csv'
        print('External Testing:', csv_file)
    elif split == 'ACRIN_T3':
        csv_file = 'ACRIN_T3.csv'
        print('External Testing:', csv_file)
    
    if task == 'treatment':
        csv_file = csv_file.replace('.csv', '_response.csv')
    elif task == 'subtyping' or task == 'tnbc' or task == 'tnbc_her2p' or task=='her2':
        csv_file = csv_file.replace('.csv', '_subtype.csv')
    
    return csv_file

class MultiModalDataset(Dataset):
    def __init__(self, split, root, transform=None, task='diagnosis', fold=0):
        """
        Args:
            csv_file: path to the file containing images
                      with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(MultiModalDataset, self).__init__()
        if fold!=0:
            fold_list = [1,2,3,4,5]
            assert fold in [1,2,3,4,5], "Not valid fold number"
            t = 'response' if task == 'treatment' else 'subtype'
            if split == 'train':
                fold_list.remove(fold)
                #df1 = pd.read_csv(os.path.join(root, f'ACRIN_DS1_fold_{fold_list[0]}_{t}.csv'))
                #df2 = pd.read_csv(os.path.join(root, f'ACRIN_DS1_fold_{fold_list[1]}_{t}.csv'))
                #df3 = pd.read_csv(os.path.join(root, f'ACRIN_DS1_fold_{fold_list[2]}_{t}.csv'))
                #df4 = pd.read_csv(os.path.join(root, f'ACRIN_DS1_fold_{fold_list[3]}_{t}.csv'))
                df1 = pd.read_csv(os.path.join(root, f'fold_{fold_list[0]}_{t}.csv'))
                df2 = pd.read_csv(os.path.join(root, f'fold_{fold_list[1]}_{t}.csv'))
                df3 = pd.read_csv(os.path.join(root, f'fold_{fold_list[2]}_{t}.csv'))
                df4 = pd.read_csv(os.path.join(root, f'fold_{fold_list[3]}_{t}.csv'))
                self.df = pd.concat([df1, df2, df3, df4]).reset_index(drop=True)

            elif split == 'test':
                #self.df = pd.read_csv(os.path.join(root, f'ACRIN_DS1_fold_{fold}_{t}.csv'))
                self.df = pd.read_csv(os.path.join(root, f'fold_{fold}_{t}.csv'))
        else:
            csv_file = get_csv_file(split, task)
            self.df = pd.read_csv(os.path.join(root, csv_file))
        
        if split=='YN':
            self.t2_path = [i.replace('/ssd3/data/YN_BC_MRI', '/project/medimgfmod/Breast_MRI/DS3') for i in self.df['T2']]
            self.dwi_path = [i.replace('/ssd3/data/YN_BC_MRI', '/project/medimgfmod/Breast_MRI/DS3') for i in self.df['DWI']]
            self.sub_path = [i.replace('/ssd3/data/YN_BC_MRI', '/project/medimgfmod/Breast_MRI/DS3') for i in self.df['SUB_concate']]
        elif split != 'ACRIN_T3':
            #self.subject = root+'/'+self.df['Subject']
            self.t2_path = root+'/'+self.df['T2']
            self.dwi_path = root+'/'+self.df['DWI']
            self.sub_path = root+'/'+self.df['SUB_concate']
        else:
            #self.subject = root+'/'+self.df['Subject']
            self.t2_path = root+'/'+self.df['T2']
            self.dwi_path = root+'/'+self.df['DWI']
            self.sub_path = root+'/'+self.df['SUB_concate']
        #self.birads = self.df['BIRADS']
        #self.RadiologyReport = self.df['RadiologyReport']
        #self.Diagnosis = self.df['Diagnosis']
        #self.PathologyResult = self.df['PathologyResult']
        if task == 'diagnosis':
            self.labels = self.df['malignant']
            self.labels = torch.LongTensor(self.labels)
        elif task == 'treatment':
            self.labels = self.df['Miller/Payne']
            self.labels = self.labels.fillna(0)     # In DS1, these are no pCR cases. So fill with 0 is fine
            self.labels = torch.LongTensor(self.labels==5)
            #self.labels = torch.LongTensor(self.df['pCR']) # for ACRIN and ACRIN+DS1
        elif task == 'subtyping':
            self.labels = self.df['subtype_label']
            self.labels = torch.LongTensor(self.labels)
        elif task == 'tnbc':
            self.labels = self.df['subtype_label']
            self.labels = torch.LongTensor(self.labels==3) # BLBC/triple-negative vs. others    
        elif task == 'tnbc_her2p':
            self.labels = self.df['subtype_label']
            self.labels = torch.LongTensor((self.labels==2)|(self.labels==3)) # BLBC/triple-negative+HER2 positive vs. others   
        elif task == 'her2':
            self.labels = self.df['subtype_label']
            self.labels = torch.LongTensor((self.labels==2)) # HER2 positive vs. others   
        #self.birads = torch.LongTensor(self.birads)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        t2_pth = self.t2_path[index]
        t2 = np.load(t2_pth)[np.newaxis, :]
        dwi_pth = self.dwi_path[index]
        dwi = np.load(dwi_pth)[np.newaxis, :]
        # try:
        #     adc = np.load(self.dwi_path[index].replace('dwi/dwi_b1000.npy', 'adc/adc.npy'))[np.newaxis, :]
        # except:
        #     adc = np.load(self.dwi_path[index+1].replace('dwi/dwi_b1000.npy', 'adc/adc.npy'))[np.newaxis, :]
        #     
        # dwi=adc
        sub_pth = self.sub_path[index]
        sub = np.load(sub_pth)
        label = self.labels[index]
        #print(sub.shape, t2.shape, dwi.shape)
        #input()
        if self.transform is not None:
            data = self.transform({'dce':sub, 'dwi':dwi, 't2':t2})
            sub, dwi, t2 = data['dce'], data['dwi'], data['t2']
        
        return index, t2, dwi, sub, label
        #return index, t2, adc, sub, label

    def __len__(self):
        return len(self.sub_path)

class TextMRIDataset(Dataset):
    def __init__(self, split, root, modal='DCE', transform=None):
        super(TextMRIDataset, self).__init__()
        csv_file = get_csv_file(split)
        self.root = root
        self.modal = modal
        self.transform = transform
        
        self.df = pd.read_csv(os.path.join(root, csv_file))
        self.subject = root+'/'+self.df['Subject']
        self.malignant = self.df['malignant']
        self.RadiologyReport = self.df['RadiologyReport']
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.Diagnosis = self.df['Diagnosis']
        self.malignant = torch.LongTensor(self.malignant)
        self.path = self.get_path()
        self.load = self.get_npy_func()

    def get_path(self):
        if self.modal == 'DCE':
            return self.root+'/'+self.df['SUB_concate']
        elif self.modal == 'DWI':
            return self.root+'/'+self.df['SUB_concate']
        elif self.modal == 'ADC':
            return self.root+'/'+self.df['SUB_concate']
        elif self.modal == 'T2':
            return self.root+'/'+self.df['SUB_concate']
        
    def get_npy_func(self):
        if self.modal == 'DCE':
            return lambda pth: np.load(pth)
        else:
            return lambda pth: np.load(pth)[np.newaxis, :]

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        mri_pth = self.path[index]
        report = self.RadiologyReport[index]
        encoding = self.tokenizer(report, padding='max_length', truncation=True, max_length=500, return_tensors='pt')
        
        mri = self.load(mri_pth)
        
        label = self.malignant[index]

        if self.transform is not None:
            mri = self.transform(mri)

        return index, mri, {'input_ids': encoding['input_ids'].squeeze(), 
                'attention_mask': encoding['attention_mask'].squeeze(), 
                'token_type_ids': encoding['token_type_ids'].squeeze()}, label

    def __len__(self):
        return len(self.path)

class MRIDataset(Dataset):
    def __init__(self, split, root, modal='DCE', transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with corresponding labels.
            transform: optional transform to be applied on a sample.
        """

        super(MRIDataset, self).__init__()
        csv_file = get_csv_file(split)
        self.root = root
        self.transform = transform
        self.modal = modal
        
        self.df = pd.read_csv(os.path.join(root, csv_file))
        self.subject = root+'/'+self.df['Subject']
        self.malignant = self.df['malignant']
        self.malignant = torch.LongTensor(self.malignant)
        self.path = self.get_path()
        self.load = self.get_npy_func()
        #self.birads = self.df['BIRADS']
        #self.birads = torch.LongTensor(self.birads)
        #self.RadiologyReport = self.df['RadiologyReport']
        #self.Diagnosis = self.df['Diagnosis']
        #self.PathologyResult = self.df['PathologyResult']

    def get_path(self):
        if self.modal == 'DCE':
            return self.root+'/'+self.df['SUB_concate']
        elif self.modal == 'DWI':
            return self.root+'/'+self.df['SUB_concate']
        elif self.modal == 'ADC':
            return self.root+'/'+self.df['SUB_concate']
        elif self.modal == 'T2':
            return self.root+'/'+self.df['SUB_concate']
        
    def get_npy_func(self):
        if self.modal == 'DCE':
            return lambda pth: np.load(pth)
        else:
            return lambda pth: np.load(pth)[np.newaxis, :]

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        mri_pth = self.path[index]
        
        data = self.load(mri_pth)
        
        label = self.malignant[index]

        if self.transform is not None:
            data = self.transform(data)

        return index, data, label

    def __len__(self):
        return len(self.path)

    
class TextDataset(Dataset):
    def __init__(self, split, root, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with corresponding labels.
            transform: optional transform to be applied on a sample.
        """

        super(TextDataset, self).__init__()
        csv_file = get_csv_file(split)
        self.df = pd.read_csv(os.path.join(root, csv_file))

        self.subject = root+'/'+self.df['Subject']
        self.t2_path = root+'/'+self.df['T2']
        self.text_path = [i.replace('t2', 'text') for i in self.t2_path]
        self.malignant = self.df['malignant']
        self.RadiologyReport = self.df['RadiologyReport']
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.Diagnosis = self.df['Diagnosis']
        #self.PathologyResult = self.df['PathologyResult']
        
        self.malignant = torch.LongTensor(self.malignant)
        #self.birads = torch.LongTensor(self.birads)
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        #text_pth = self.text_path[index]
        #data = np.squeeze(np.load(text_pth))
        report = self.RadiologyReport[index]
        path_diag = self.Diagnosis[index]
        label = self.malignant[index]

        #return index, data, label
        encoding = self.tokenizer(report, padding='max_length', truncation=True, max_length=500, return_tensors='pt')
        #encoding = self.tokenizer(path_diag, padding='max_length', truncation=True, max_length=500, return_tensors='pt')
        return index, {'input_ids': encoding['input_ids'].squeeze(), 
                'attention_mask': encoding['attention_mask'].squeeze(), 
                'token_type_ids': encoding['token_type_ids'].squeeze()}, label

    def __len__(self):
        return len(self.RadiologyReport)