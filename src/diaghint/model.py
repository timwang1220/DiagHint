"""One classifier and one shared ordered-tree utility model; no variants."""
import torch
from torch import nn


class CardinalityClassifier(nn.Module):
    def __init__(self,dim):
        super().__init__()
        self.shared=nn.Sequential(nn.Linear(dim,128),nn.ReLU(),nn.Dropout(.15),nn.Linear(128,32),nn.ReLU())
        self.classes=nn.Linear(32,5);self.ratio=nn.Linear(32,1)
    def forward(self,x):
        hidden=self.shared(x)
        return dict(logits=self.classes(hidden),log_rho=self.ratio(hidden).squeeze(-1))


class ExperienceUtility(nn.Module):
    """Demo increment, native-relative reuse, and offline NoDemo value."""
    def __init__(self,dim):
        super().__init__();self.project=nn.Sequential(nn.Linear(dim+6,64),nn.ReLU())
        self.input_gates=nn.Linear(66,320)
        self.left_gates=nn.Linear(64,320,bias=False);self.right_gates=nn.Linear(64,320,bias=False)
        def head(size):return nn.Sequential(nn.Linear(size,128),nn.ReLU(),nn.Dropout(.15),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
        self.reuse=head(260);self.demo=head(324);self.nodemo=head(64)

    def encode(self,x,children,diagnostics):
        x=self.project(torch.cat([x,diagnostics],-1));states={};cells={};zero=x.new_zeros(64)
        def visit(i):
            cs=children[i]
            if len(cs)>2:raise ValueError('Binary Tree-LSTM rejects more than two children')
            for child in cs:visit(child)
            hl,cl=(states[cs[0]],cells[cs[0]]) if cs else (zero,zero)
            hr,cr=(states[cs[1]],cells[cs[1]]) if len(cs)==2 else (zero,zero)
            masks=x.new_tensor([float(bool(cs)),float(len(cs)==2)])
            gates=self.input_gates(torch.cat([x[i],masks]))+self.left_gates(hl)+self.right_gates(hr)
            ingate,outgate,candidate,fl,fr=gates.chunk(5)
            cells[i]=ingate.sigmoid()*candidate.tanh()+fl.sigmoid()*cl+fr.sigmoid()*cr
            states[i]=outgate.sigmoid()*cells[i].tanh()
        visit(0)
        return states[0]

    def score(self,mode,embeddings,bits):
        a=embeddings[0]
        if mode=='nodemo':return self.nodemo(a).squeeze(-1)
        if mode=='reuse':
            b=embeddings[1];value=torch.cat([a,b,b-a,(b-a).abs(),bits])
        elif mode=='demo':
            b,c=embeddings[1:];value=torch.cat([a,b,c-b,(b-a).abs(),a*b,bits])
        else:raise ValueError('Unknown utility task')
        return getattr(self,mode)(value).squeeze(-1)
