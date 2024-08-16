import torch
from torchvision import transforms as T
#from data.mri import MultiModalDataset, DCEDataset, DWIDataset, ADCDataset, T2Dataset, TextDataset
from data.mri import MultiModalDataset, MRIDataset, TextDataset, TextMRIDataset
from data.augmentation import Inserter, Flipper, MultimodalInserter, MultimodalFlipper, MultimodalRotater, MultimodalResizer,\
                              MultimodalSixFlipper, MultimodalNineInserter
from torch.utils.data import Dataset


# transforms = {
#         'DCE':{
#             "train": T.Compose([Inserter(size=(448, 256, 88)), Flipper()]),
#             "val": T.Compose([Inserter(size=(448, 256, 88))]),
#             "test": T.Compose([Inserter(size=(448, 256, 88))])
#             },
#         'T2':{
#             "train": T.Compose([Inserter(size=(384, 256, 48)), Flipper()]),
#             "val": T.Compose([Inserter(size=(384, 256, 48))]),
#             "test": T.Compose([Inserter(size=(384, 256, 48))])
#             },
#         'DWI':{
#             "train": T.Compose([Inserter(size=(256, 128, 32)), Flipper()]),
#             "val": T.Compose([Inserter(size=(256, 128, 32))]),
#             "test": T.Compose([Inserter(size=(256, 128, 32))])
#             },
#         'Multi':{
#             "train": T.Compose([MultimodalInserter(dce_size=(448, 256, 88),
#                                                 dwi_size=(256, 128, 32),
#                                                 t2_size=(384, 256, 48)), MultimodalFlipper()]),
#             "val": T.Compose([MultimodalInserter(dce_size=(448, 256, 88),
#                                                 dwi_size=(256, 128, 32),
#                                                 t2_size=(384, 256, 48))]),
#             "test": T.Compose([MultimodalInserter(dce_size=(448, 256, 88),
#                                                 dwi_size=(256, 128, 32),
#                                                 t2_size=(384, 256, 48))])
#         },}

transforms = {
    'DCE':{
        "train": T.Compose([Inserter(size=(384, 256, 128)), Flipper()]),
        "val": T.Compose([Inserter(size=(384, 256, 128), rand=False)]),
        "test": T.Compose([Inserter(size=(384, 256, 128), rand=False)])
        },
    'T2':{
        "train": T.Compose([Inserter(size=(384, 256, 48)), Flipper()]),
        "val": T.Compose([Inserter(size=(384, 256, 48), rand=False)]),
        "test": T.Compose([Inserter(size=(384, 256, 48), rand=False)])
        },
    'DWI':{
        "train": T.Compose([Inserter(size=(256, 128, 32)), Flipper()]),
        "val": T.Compose([Inserter(size=(256, 128, 32), rand=False)]),
        "test": T.Compose([Inserter(size=(256, 128, 32), rand=False)])
        },
    'ADC':{
        "train": T.Compose([Inserter(size=(256, 128, 32)), Flipper()]),
        "val": T.Compose([Inserter(size=(256, 128, 32), rand=False)]),
        "test": T.Compose([Inserter(size=(256, 128, 32), rand=False)])
        },
    'Multi':{
        "train": T.Compose([MultimodalResizer(dce_size=(384, 256, 128),
                                             dwi_size=(256, 128, 32),
                                             t2_size=(384, 256, 48)),
                            MultimodalInserter(dce_size=(384, 256, 128),
                                               dwi_size=(256, 128, 32),
                                               t2_size=(384, 256, 48)), MultimodalFlipper()]),#, MultimodalRotater()]),
        "val": T.Compose([MultimodalResizer(dce_size=(384, 256, 128),
                                             dwi_size=(256, 128, 32),
                                             t2_size=(384, 256, 48)),
                            MultimodalInserter(dce_size=(384, 256, 128),
                                               dwi_size=(256, 128, 32),
                                               t2_size=(384, 256, 48), rand=False)]),
        "test": T.Compose([MultimodalResizer(dce_size=(384, 256, 128),
                                             dwi_size=(256, 128, 32),
                                             t2_size=(384, 256, 48)),
                           MultimodalInserter(dce_size=(384, 256, 128),
                                               dwi_size=(256, 128, 32),
                                               t2_size=(384, 256, 48), rand=False)]),# 
        **{"test_flip%d"%i: T.Compose([MultimodalResizer(dce_size=(384, 256, 128),
                                             dwi_size=(256, 128, 32),
                                             t2_size=(384, 256, 48)),
                                      MultimodalInserter(dce_size=(384, 256, 128),
                                                     dwi_size=(256, 128, 32),
                                                     t2_size=(384, 256, 48), rand=False), 
                                      MultimodalSixFlipper(op=i)]) for i in range(6)},
        
        **{"test_pad%d"%j: T.Compose([MultimodalResizer(dce_size=(384, 256, 128),
                                             dwi_size=(256, 128, 32),
                                             t2_size=(384, 256, 48)),
                                     MultimodalNineInserter(dce_size=(384, 256, 128),
                                               dwi_size=(256, 128, 32),
                                               t2_size=(384, 256, 48), op=j)]) for j in range(9)},
        **{"test_pad%dflip%d"%(n,m): T.Compose([MultimodalResizer(dce_size=(384, 256, 128),
                                               dwi_size=(256, 128, 32),
                                               t2_size=(384, 256, 48)),
                                               MultimodalNineInserter(dce_size=(384, 256, 128),
                                               dwi_size=(256, 128, 32),
                                               t2_size=(384, 256, 48), op=n),
                                               MultimodalSixFlipper(op=m)]) for n in range(9) for m in range(6)},
        },
}


def get_dataset(modal, dataset_split, transform_split, root='/project/medimgfmod/Breast_MRI/DS1', textmri=False, task='diagnosis', fold=0):
    root = root
    #root = '/project/medimgfmod/Breast_MRI/DS1'
    #root = '/project/medimgfmod/Breast_MRI/DS2'
    #root = '/jhcnas3/BreastMRI/npy/DS1'
    #transform_dce = transforms['DCE'][transform_split]
    #transform_t2 = transforms['T2'][transform_split]
    #transform_dwi = transforms['DWI'][transform_split]
    #transform_adc = transforms['ADC'][transform_split]
    transform = None
    
    if modal in ['DCE', 'T2', 'DWI', 'ADC']:
        dataset = MRIDataset(split=dataset_split,
                             root=root,
                             transform=transforms[modal][transform_split],
                             modal=modal,
                             task=task)
    #if modal == 'DCE':
    #    dataset = DCEDataset(
    #              split=dataset_split,
    #              root=root,
    #              transform=transform_dce) 
    #elif modal == 'T2':
    #    dataset = T2Dataset(
    #              split=dataset_split,
    #              root=root,
    #              transform=transform_t2)
    #elif modal == 'DWI':
    #    dataset = DWIDataset(
    #              split=dataset_split,
    #              root=root,
    #              transform=transform_dwi)
    #elif modal == 'ADC':
    #    dataset = ADCDataset(
    #              split=dataset_split,
    #              root=root,
    #              transform=transform_adc)
    elif modal == 'Multi':
        dataset = MultiModalDataset(
                  split=dataset_split,
                  root=root,
                  transform=transforms[modal][transform_split],
                  task=task,
                  fold=fold)     
    else:
        raise KeyError(f"This dataload function is not yet implemented.")
    
    return dataset

