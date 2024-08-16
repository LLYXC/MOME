import torch
import torch.nn as nn
import torch.nn.functional as F


class AdMSoftmaxLoss(nn.Module):

    #def __init__(self, embedding_dim, no_classes, scale = 30.0, margin=0.4):
    def __init__(self, no_classes=2, reduction='none', scale = 30.0, margin=0.4):
        '''
        Additive Margin Softmax Loss


        Attributes
        ----------
        embedding_dim : int 
            Dimension of the embedding vector
        no_classes : int
            Number of classes to be embedded
        scale : float
            Global scale factor
        margin : float
            Size of additive margin        
        '''
        super(AdMSoftmaxLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        #self.embedding_dim = embedding_dim
        self.no_classes = no_classes
        self.embedding = nn.Embedding(no_classes, embedding_dim, max_norm=1)
        self.loss = nn.CrossEntropyLoss(reduction=reduction)

    def forward(self, x, labels):
        '''
        Input shape (N, embedding_dim)
        '''
        n, m = x.shape        
        #assert n == len(labels)
        #assert m == self.embedding_dim
        #assert torch.min(labels) >= 0
        #assert torch.max(labels) < self.no_classes

        x = F.normalize(x, dim=1)
        w = self.embedding.weight        
        cos_theta = torch.matmul(w, x.T).T
        psi = cos_theta - self.margin
        
        onehot = F.one_hot(labels, self.no_classes)
        logits = self.scale * torch.where(onehot == 1, psi, cos_theta)        
        err = self.loss(logits, labels)
        
        return err, logits