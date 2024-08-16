import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from transformers import AutoTokenizer

def get_csv_file(split):
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
    return csv_file

class MultiModalDataset(Dataset):
    def __init__(self, split, root, transform_t2=None, transform_dwi=None, transform_adc=None, transform_dce=None):
        """
        Args:
            csv_file: path to the file containing images
                      with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(MultiModalDataset, self).__init__()
        csv_file = get_csv_file(split)
        self.df = pd.read_csv(os.path.join(root, csv_file))

        self.subject = root+'/'+self.df['Subject']
        self.t2_path = root+'/'+self.df['T2']
        self.dwi_path = root+'/'+self.df['DWI']
        self.sub_path = root+'/'+self.df['SUB_concate']
        self.malignant = self.df['malignant']
        #self.birads = self.df['BIRADS']
        #self.RadiologyReport = self.df['RadiologyReport']
        #self.Diagnosis = self.df['Diagnosis']
        #self.PathologyResult = self.df['PathologyResult']
        
        self.malignant = torch.LongTensor(self.malignant)
        #self.birads = torch.LongTensor(self.birads)
        self.transform_t2 = transform_t2
        self.transform_dwi = transform_dwi
        self.transform_adc = transform_adc
        self.transform_dce = transform_dce

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
        #adc_pth = self.adc_pth[index]
        #adc = np.load(adc_pth)[np.newaxis, :]
        sub_pth = self.sub_path[index]
        sub = np.load(sub_pth)
        
        label = self.malignant[index]

        if self.transform_t2 is not None:
            t2 = self.transform_t2(t2)
        if self.transform_dwi is not None:
            dwi = self.transform_dwi(dwi)
        #if self.transform_dwi is not None:
        #    adc = self.transform_adc(adc)
        if self.transform_dce is not None:
            sub = self.transform_dce(sub)

        return index, t2, dwi, sub, label
        #return index, t2, adc, sub, label

    def __len__(self):
        return len(self.sub_path)

class MultiModalTextDataset(MultiModalDataset):
    def __init__(self, split, root, transform_t2=None, transform_dwi=None, transform_dce=None):
        """
        Args:
            csv_file: path to the file containing images
                      with corresponding labels.
            transform: optional transform to be applied on a sample.
        """
        super(MultiModalDataset, self).__init__(split, root, transform_t2, transform_dwi, transform_dce)
        self.text_path = [i.replace('t2', 'text') for i in self.t2_path]

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
        sub_pth = self.sub_path[index]
        sub = np.load(sub_pth)
        text_pth = self.text_path[index]
        text = np.load(text_pth)
        
        label = self.malignant[index]

        if self.transform_t2 is not None:
            t2 = self.transform_t2(t2)
        if self.transform_dwi is not None:
            dwi = self.transform_dwi(dwi)
        if self.transform_dce is not None:
            sub = self.transform_dce(sub)

        return index, t2, dwi, sub, text, label

    def __len__(self):
        return len(self.sub_path)


class DCEDataset(Dataset):
    def __init__(self, split, root, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with corresponding labels.
            transform: optional transform to be applied on a sample.
        """

        super(DCEDataset, self).__init__()
        csv_file = get_csv_file(split)
        self.df = pd.read_csv(os.path.join(root, csv_file))

        self.subject = root+'/'+self.df['Subject']
        self.sub_path = root+'/'+self.df['SUB_concate']
        self.malignant = self.df['malignant']
        self.malignant = torch.LongTensor(self.malignant)
        #self.birads = self.df['BIRADS']
        #self.birads = torch.LongTensor(self.birads)
        #self.RadiologyReport = self.df['RadiologyReport']
        #self.Diagnosis = self.df['Diagnosis']
        #self.PathologyResult = self.df['PathologyResult']
        
        self.transform = transform

    def __getitem__(self, index):
        """
        Args:
            index: the index of item
        Returns:
            image and its labels
        """
        sub_pth = self.sub_path[index]
        
        data = np.load(sub_pth)
        
        label = self.malignant[index]

        if self.transform is not None:
            data = self.transform(data)

        return index, data, label

    def __len__(self):
        return len(self.sub_path)


class DWIDataset(Dataset):
    def __init__(self, split, root, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with corresponding labels.
            transform: optional transform to be applied on a sample.
        """

        super(DWIDataset, self).__init__()
        csv_file = get_csv_file(split)
        self.df = pd.read_csv(os.path.join(root, csv_file))

        self.subject = root+'/'+self.df['Subject']
        self.dwi_path = root+'/'+self.df['DWI']
        self.malignant = self.df['malignant']
        #self.birads = self.df['BIRADS']
        #self.RadiologyReport = self.df['RadiologyReport']
        #self.Diagnosis = self.df['Diagnosis']
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
        dwi_pth = self.dwi_path[index]

        data = np.load(dwi_pth)[np.newaxis, :]
        label = self.malignant[index]

        if self.transform is not None:
            data = self.transform(data)

        return index, data, label

    def __len__(self):
        return len(self.dwi_path)
    
class ADCDataset(Dataset):
    def __init__(self, split, root, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with corresponding labels.
            transform: optional transform to be applied on a sample.
        """

        super(ADCDataset, self).__init__()
        csv_file = get_csv_file(split)
        self.df = pd.read_csv(os.path.join(root, csv_file))

        self.subject = root+'/'+self.df['Subject']
        self.adc_path = root+'/'+self.df['ADC']
        self.malignant = self.df['malignant']
        #self.birads = self.df['BIRADS']
        #self.RadiologyReport = self.df['RadiologyReport']
        #self.Diagnosis = self.df['Diagnosis']
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
        adc_pth = self.adc_path[index]

        data = np.load(adc_pth)[np.newaxis, :]
        label = self.malignant[index]

        if self.transform is not None:
            data = self.transform(data)

        return index, data, label

    def __len__(self):
        return len(self.adc_path)


class T2Dataset(Dataset):
    def __init__(self, split, root, transform=None):
        """
        Args:
            csv_file: path to the file containing images
                      with corresponding labels.
            transform: optional transform to be applied on a sample.
        """

        super(T2Dataset, self).__init__()
        csv_file = get_csv_file(split)
        self.df = pd.read_csv(os.path.join(root, csv_file))

        self.subject = root+'/'+self.df['Subject']
        self.t2_path = root+'/'+self.df['T2']
        self.malignant = self.df['malignant']
        #self.birads = self.df['BIRADS']
        #self.RadiologyReport = self.df['RadiologyReport']
        #self.Diagnosis = self.df['Diagnosis']
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
        t2_pth = self.t2_path[index]
        data = np.load(t2_pth)[np.newaxis, :]
        label = self.malignant[index]

        if self.transform is not None:
            data = self.transform(data)

        return index, data, label

    def __len__(self):
        return len(self.t2_path)
    
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

        if self.transform is not None:
            data = self.transform(data)

        #return index, data, label
        encoding = self.tokenizer(report, padding='max_length', truncation=True, max_length=500, return_tensors='pt')
        #encoding = self.tokenizer(path_diag, padding='max_length', truncation=True, max_length=500, return_tensors='pt')
        return index, {'input_ids': encoding['input_ids'].squeeze(), 
                'attention_mask': encoding['attention_mask'].squeeze(), 
                'token_type_ids': encoding['token_type_ids'].squeeze()}, label

    def __len__(self):
        return len(self.RadiologyReport)