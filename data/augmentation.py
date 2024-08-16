import numpy as np
import torch
import monai
from scipy.ndimage.interpolation import zoom        

class Flipper(object):
    '''
        Flipping data via three axis
        data is of dim: channel x width x height x depth
        axis: set to 1, 2, 3 for flipping on width, height, or depth, respectively
    '''
    def __call__(self, data):
        
        num_ops = 6
        odds = np.ones(num_ops) / num_ops
        op = np.random.choice(np.arange(num_ops), p=odds)
        if op == 0:
            data = self.flip(data, 1)
            data = np.ascontiguousarray(data)   # copy to avoid negative strides of numpy arrays
        elif op == 1:
            data = self.flip(data, 2)
            data = np.ascontiguousarray(data) 
        elif op == 2:
            data = self.flip(data, 3)
            data = np.ascontiguousarray(data) 
        elif op == 3:
            data = self.flip(data, 1)
            data = self.flip(data, 2)
            data = np.ascontiguousarray(data) 
        elif op == 4:
            data = self.flip(data, 1)
            data = self.flip(data, 3)
            data = np.ascontiguousarray(data) 
        elif op == 5:
            data = self.flip(data, 2)
            data = self.flip(data, 3)
            data = np.ascontiguousarray(data) 
        return  data

    def flip(self, data, axis):
        output = np.asarray(data).swapaxes(axis, 0)
        output = output[::-1, ...]
        output = output.swapaxes(0, axis)
        return output
    
class Inserter(object):
    '''
            Insert the data into a fixed size
            data is of dim: channel x width x height x depth
            return torch tensor
            size: set to be the largest of all data
    '''
    def __init__(self, size, rand=True):
        super(Inserter, self).__init__()
        self.size = size
        self.rand = rand
    def __call__(self, data):
        
        c, w, h, d = data.shape
        ww, hh, dd = self.size
        output = np.zeros((c, ww, hh, dd)).astype(np.float32)
        if self.rand:
            x = np.random.randint(0, ww - w + 1)
            y = np.random.randint(0, hh - h + 1)
            z = np.random.randint(0, dd - d + 1)
        else:
            x = (ww - w)//2
            y = (hh - h)//2 
            z = (dd - d)//2
        output[:, x:x+w, y:y+h, z:z+d] = data
            
        return torch.from_numpy(output)


class MultimodalFlipper(object):
    '''
        Flipping data via three axis
        data is of dim: channel x width x height x depth
        axis: set to 1, 2, 3 for flipping on width, height, or depth, respectively
    '''
    def __call__(self, data):
        dce, dwi, t2 = data['dce'], data['dwi'], data['t2']
        
        num_ops = 6
        odds = np.ones(num_ops) / num_ops
        op = np.random.choice(np.arange(num_ops), p=odds)
        if op == 0:
            dce = self.flip(dce, 1)
            dce = np.ascontiguousarray(dce)   # copy to avoid negative strides of numpy arrays
            dwi = self.flip(dwi, 1)
            dwi = np.ascontiguousarray(dwi)
            t2 = self.flip(t2, 1)
            t2 = np.ascontiguousarray(t2)
        elif op == 1:
            dce = self.flip(dce, 2)
            dce = np.ascontiguousarray(dce)
            dwi = self.flip(dwi, 2)
            dwi = np.ascontiguousarray(dwi)
            t2 = self.flip(t2, 2)
            t2 = np.ascontiguousarray(t2)
        elif op == 2:
            dce = self.flip(dce, 3)
            dce = np.ascontiguousarray(dce)
            dwi = self.flip(dwi, 3)
            dwi = np.ascontiguousarray(dwi)
            t2 = self.flip(t2, 3)
            t2 = np.ascontiguousarray(t2)
        elif op == 3:
            dce = self.flip(dce, 1)
            dce = self.flip(dce, 2)
            dce = np.ascontiguousarray(dce)
            dwi = self.flip(dwi, 1)
            dwi = self.flip(dwi, 2)
            dwi = np.ascontiguousarray(dwi)
            t2 = self.flip(t2, 1)
            t2 = self.flip(t2, 2)
            t2 = np.ascontiguousarray(t2)
        elif op == 4:
            dce = self.flip(dce, 1)
            dce = self.flip(dce, 3)
            dce = np.ascontiguousarray(dce)
            dwi = self.flip(dwi, 1)
            dwi = self.flip(dwi, 3)
            dwi = np.ascontiguousarray(dwi)
            t2 = self.flip(t2, 1)
            t2 = self.flip(t2, 3)
            t2 = np.ascontiguousarray(t2)
        elif op == 5:
            dce = self.flip(dce, 2)
            dce = self.flip(dce, 3)
            dce = np.ascontiguousarray(dce)
            dwi = self.flip(dwi, 2)
            dwi = self.flip(dwi, 3)
            dwi = np.ascontiguousarray(dwi)
            t2 = self.flip(t2, 2)
            t2 = self.flip(t2, 3)
            t2 = np.ascontiguousarray(t2)
        return  {'dce':dce, 'dwi':dwi,'t2':t2}

    def flip(self, data, axis):
        output = np.asarray(data).swapaxes(axis, 0)
        output = output[::-1, ...]
        output = output.swapaxes(0, axis)
        return output

class MultimodalSixFlipper(object):
    '''
        Flipping data via three axis
        data is of dim: channel x width x height x depth
        axis: set to 1, 2, 3 for flipping on width, height, or depth, respectively
    '''
    def __init__(self, op):
        self.op = op
    def __call__(self, data):
        dce, dwi, t2 = data['dce'], data['dwi'], data['t2']
        
        if self.op == 0:
            dce = self.flip(dce, 1)
            dce = np.ascontiguousarray(dce)   # copy to avoid negative strides of numpy arrays
            dwi = self.flip(dwi, 1)
            dwi = np.ascontiguousarray(dwi)
            t2 = self.flip(t2, 1)
            t2 = np.ascontiguousarray(t2)
        elif self.op == 1:
            dce = self.flip(dce, 2)
            dce = np.ascontiguousarray(dce)
            dwi = self.flip(dwi, 2)
            dwi = np.ascontiguousarray(dwi)
            t2 = self.flip(t2, 2)
            t2 = np.ascontiguousarray(t2)
        elif self.op == 2:
            dce = self.flip(dce, 3)
            dce = np.ascontiguousarray(dce)
            dwi = self.flip(dwi, 3)
            dwi = np.ascontiguousarray(dwi)
            t2 = self.flip(t2, 3)
            t2 = np.ascontiguousarray(t2)
        elif self.op == 3:
            dce = self.flip(dce, 1)
            dce = self.flip(dce, 2)
            dce = np.ascontiguousarray(dce)
            dwi = self.flip(dwi, 1)
            dwi = self.flip(dwi, 2)
            dwi = np.ascontiguousarray(dwi)
            t2 = self.flip(t2, 1)
            t2 = self.flip(t2, 2)
            t2 = np.ascontiguousarray(t2)
        elif self.op == 4:
            dce = self.flip(dce, 1)
            dce = self.flip(dce, 3)
            dce = np.ascontiguousarray(dce)
            dwi = self.flip(dwi, 1)
            dwi = self.flip(dwi, 3)
            dwi = np.ascontiguousarray(dwi)
            t2 = self.flip(t2, 1)
            t2 = self.flip(t2, 3)
            t2 = np.ascontiguousarray(t2)
        elif self.op == 5:
            dce = self.flip(dce, 2)
            dce = self.flip(dce, 3)
            dce = np.ascontiguousarray(dce)
            dwi = self.flip(dwi, 2)
            dwi = self.flip(dwi, 3)
            dwi = np.ascontiguousarray(dwi)
            t2 = self.flip(t2, 2)
            t2 = self.flip(t2, 3)
            t2 = np.ascontiguousarray(t2)
        return  {'dce':dce, 'dwi':dwi,'t2':t2}

    def flip(self, data, axis):
        output = np.asarray(data).swapaxes(axis, 0)
        output = output[::-1, ...]
        output = output.swapaxes(0, axis)
        return output
    
class MultimodalInserter(object):
    '''
            Insert the data into a fixed size
            data is of dim: channel x width x height x depth
            return torch tensor
            size: set to be the largest of all data
    '''
    def __init__(self, dce_size, dwi_size, t2_size, rand=True):
        super(MultimodalInserter, self).__init__()
        self.dce_size = dce_size
        self.dwi_size = dwi_size
        self.t2_size = t2_size
        self.rand = rand
        
    def __call__(self, data):
        
        dce, dwi, t2 = data['dce'], data['dwi'], data['t2']
        
        dce_output = self.insert(dce, self.dce_size)
        dce_output = torch.from_numpy(dce_output)
        dwi_output = self.insert(dwi, self.dwi_size)
        dwi_output = torch.from_numpy(dwi_output)
        t2_output = self.insert(t2, self.t2_size)
        t2_output = torch.from_numpy(t2_output)
        return {'dce':dce_output, 'dwi':dwi_output,'t2':t2_output}
        
    def insert(self, data, size):
        c, w, h, d = data.shape
        ww, hh, dd = size
        output = np.zeros((c, ww, hh, dd)).astype(np.float32)
        if self.rand:
            x = np.random.randint(0, ww - w + 1)
            y = np.random.randint(0, hh - h + 1)
            z = np.random.randint(0, dd - d + 1)
        else:
            x = (ww - w)//2
            y = (hh - h)//2 
            z = (dd - d)//2
        output[:, x:x+w, y:y+h, z:z+d] = data
        return output
    
class MultimodalNineInserter(object):
    '''
            Insert the data into a fixed size
            data is of dim: channel x width x height x depth
            return torch tensor
            size: set to be the largest of all data
    '''
    def __init__(self, dce_size, dwi_size, t2_size, op):
        super(MultimodalNineInserter, self).__init__()
        self.dce_size = dce_size
        self.dwi_size = dwi_size
        self.t2_size = t2_size
        self.op = op
        
    def __call__(self, data):
        
        dce, dwi, t2 = data['dce'], data['dwi'], data['t2']
        
        dce_output = self.insert(dce, self.dce_size)
        dce_output = torch.from_numpy(dce_output)
        dwi_output = self.insert(dwi, self.dwi_size)
        dwi_output = torch.from_numpy(dwi_output)
        t2_output = self.insert(t2, self.t2_size)
        t2_output = torch.from_numpy(t2_output)
        return {'dce':dce_output, 'dwi':dwi_output,'t2':t2_output}
        
    def insert(self, data, size, op=0):
        c, w, h, d = data.shape
        ww, hh, dd = size
        output = np.zeros((c, ww, hh, dd)).astype(np.float32)
        if self.op == 0:
            x = (ww - w)//2
            y = (hh - h)//2 
            z = (dd - d)//2
        elif self.op == 1:
            x = 0
            y = 0 
            z = 0
        elif self.op == 2:
            x = 0
            y = 0 
            z = (dd - d)
        elif self.op == 3:
            x = 0
            y = (hh - h) 
            z = 0
        elif self.op == 4:
            x = 0
            y = (hh - h) 
            z = (dd - d)
        elif self.op == 5:
            x = (ww - w)
            y = 0 
            z = 0
        elif self.op == 6:
            x = (ww - w)
            y = 0
            z = (dd - d)
        elif self.op == 7:
            x = (ww - w)
            y = (hh - h) 
            z = 0
        elif self.op == 8:
            x = (ww - w)
            y = (hh - h)
            z = (dd - d)
        output[:, x:x+w, y:y+h, z:z+d] = data
        return output
    
class MultimodalResizer(object):
    '''
            Insert the data into a fixed size
            data is of dim: channel x width x height x depth
            return torch tensor
            size: set to be the largest of all data
    '''
    def __init__(self, dce_size, dwi_size, t2_size):
        super(MultimodalResizer, self).__init__()
        self.dce_size = dce_size
        self.dwi_size = dwi_size
        self.t2_size = t2_size
        
    def __call__(self, data):
        
        dce, dwi, t2 = data['dce'], data['dwi'], data['t2']
        
        dce_output = self.resize(dce, self.dce_size)
        dce_output = torch.from_numpy(dce_output)
        dwi_output = self.resize(dwi, self.dwi_size)
        dwi_output = torch.from_numpy(dwi_output)
        t2_output = self.resize(t2, self.t2_size)
        t2_output = torch.from_numpy(t2_output)
        return {'dce':dce_output, 'dwi':dwi_output,'t2':t2_output}
        
    def resize(self, data, size):
        c, w, h, d = data.shape
        ww, hh, dd = size
        if (w<=ww) and (h<=hh) and (d<=dd):
            return data
        else:
            rat_1 = ww/w
            rat_2 = hh/h
            rat_3 = dd/d
            ratio = min(rat_1, rat_2, rat_3)
            return zoom(data, (1, ratio, ratio, ratio), order=1)
    
class MultimodalRotater(object):
    '''
        Flipping data via three axis
        data is of dim: channel x width x height x depth
        axis: set to 1, 2, 3 for flipping on width, height, or depth, respectively
    '''
    def __call__(self, data, radian=0.175):
        dce, dwi, t2 = data['dce'], data['dwi'], data['t2']
        
        num_ops = 7
        odds = np.ones(num_ops) / num_ops
        op = np.random.choice(np.arange(num_ops), p=odds)
        if op == 0:
            transform= monai.transforms.RandRotate(range_x=radian,prob=1)
        elif op == 1:
            transform = monai.transforms.RandRotate(range_y=radian, prob=1)
        elif op == 2:
            transform = monai.transforms.RandRotate(range_z=radian, prob=1)
        elif op == 3:
            transform = monai.transforms.RandRotate(range_x=radian,range_y=radian, prob=1)
        elif op == 4:
            transform = monai.transforms.RandRotate(range_x=radian,range_z=radian,prob=1)
        elif op == 5:
            transform = monai.transforms.RandRotate(range_y=radian,range_z=radian,prob=1)
        elif op==6:
            transform = monai.transforms.RandRotate(range_x=radian,range_y=radian,range_z=radian,prob=1)
        
        dce = transform(dce)
        dwi = transform(dwi)
        t2 = transform(t2)
        
        return  {'dce':dce, 'dwi':dwi,'t2':t2}
