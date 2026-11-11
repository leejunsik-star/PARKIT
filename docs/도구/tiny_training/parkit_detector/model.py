"""PARKIT Tiny: handwritten single-class anchor-free detector."""
import math
import torch
from torch import nn
import torch.nn.functional as F

class Conv(nn.Sequential):
    def __init__(self, a, b, stride=1):
        super().__init__(nn.Conv2d(a,b,3,stride,1,bias=False),nn.BatchNorm2d(b),nn.SiLU())

class Block(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.body=nn.Sequential(nn.Conv2d(c,c,3,padding=1,groups=c,bias=False),nn.BatchNorm2d(c),nn.SiLU(),nn.Conv2d(c,c,1,bias=False),nn.BatchNorm2d(c),nn.SiLU())
    def forward(self,x): return x+self.body(x)

class Head(nn.Module):
    def __init__(self,c):
        super().__init__()
        self.obj=nn.Sequential(Conv(c,c),nn.Conv2d(c,1,1))
        self.box=nn.Sequential(Conv(c,c),nn.Conv2d(c,4,1))
        nn.init.constant_(self.obj[-1].bias,math.log(.01/.99))
    def forward(self,x): return torch.cat((self.obj(x),self.box(x)),1)

class ParkitTiny(nn.Module):
    strides=(8,16)
    def __init__(self,width=16):
        super().__init__(); self.width=width
        self.stem=Conv(3,width,2)
        self.s4=nn.Sequential(Conv(width,width*2,2),Block(width*2))
        self.s8=nn.Sequential(Conv(width*2,width*4,2),Block(width*4))
        self.s16=nn.Sequential(Conv(width*4,width*8,2),Block(width*8),Block(width*8))
        self.lateral=nn.Conv2d(width*8,width*4,1)
        self.fuse=nn.Sequential(Conv(width*8,width*4),Block(width*4))
        self.heads=nn.ModuleList([Head(width*4),Head(width*8)])
    def forward(self,x):
        x=self.s4(self.stem(x)); p8=self.s8(x); p16=self.s16(p8)
        p8=self.fuse(torch.cat((p8,F.interpolate(self.lateral(p16),size=p8.shape[-2:],mode='nearest')),1))
        return [self.heads[0](p8),self.heads[1](p16)]

def decode(outputs):
    boxes=[]; logits=[]; points=[]; strides=[]
    for out,s in zip(outputs,ParkitTiny.strides):
        b,_,h,w=out.shape
        yy,xx=torch.meshgrid(torch.arange(h,device=out.device),torch.arange(w,device=out.device),indexing='ij')
        p=torch.stack((xx+.5,yy+.5),-1).reshape(-1,2)*s
        raw=out.flatten(2).transpose(1,2)
        dist=F.softplus(raw[...,1:])*s
        boxes.append(torch.cat((p-dist[...,:2],p+dist[...,2:]),-1))
        logits.append(raw[...,0]); points.append(p);strides.append(torch.full((h*w,),s,device=out.device))
    return torch.cat(boxes,1),torch.cat(logits,1),torch.cat(points),torch.cat(strides)

def iou(a,b):
    lt=torch.maximum(a[...,:2],b[...,:2]); rb=torch.minimum(a[...,2:],b[...,2:])
    inter=(rb-lt).clamp(min=0).prod(-1)
    aa=(a[...,2:]-a[...,:2]).clamp(min=0).prod(-1);bb=(b[...,2:]-b[...,:2]).clamp(min=0).prod(-1)
    return inter/(aa+bb-inter+1e-7)

def giou(a,b):
    lt=torch.maximum(a[:,:2],b[:,:2]); rb=torch.minimum(a[:,2:],b[:,2:]); inter=(rb-lt).clamp(min=0).prod(-1)
    union=(a[:,2:]-a[:,:2]).prod(-1)+(b[:,2:]-b[:,:2]).prod(-1)-inter
    enclosing=(torch.maximum(a[:,2:],b[:,2:])-torch.minimum(a[:,:2],b[:,:2])).prod(-1)
    return inter/(union+1e-7)-(enclosing-union)/(enclosing+1e-7)

def detection_loss(outputs,targets):
    boxes,logits,points,strides=decode(outputs)
    obj=torch.zeros_like(logits); positives=[]
    # Center sampling: inside GT, within 1.5 strides of center, scale split at 96 px.
    for b,gt in enumerate(targets):
        if not len(gt): continue
        center=(gt[:,:2]+gt[:,2:])/2; size=(gt[:,2:]-gt[:,:2]).amax(-1)
        inside=(points[:,None,:]>gt[None,:,:2]).all(-1)&(points[:,None,:]<gt[None,:,2:]).all(-1)
        near=((points[:,None,:]-center).abs()<=1.5*strides[:,None,None]).all(-1)
        level=torch.where(size[None,:]<=96,strides[:,None]==8,strides[:,None]==16)
        eligible=inside&near&level
        # Tiny boxes may have no point on their preferred level; permit the
        # nearest point inside the box on either level, when one exists.
        for j in range(len(gt)):
            if not eligible[:,j].any() and inside[:,j].any():
                distance=((points-center[j])**2).sum(-1)
                distance[~inside[:,j]]=float('inf')
                eligible[distance.argmin(),j]=True
        area=(gt[:,2:]-gt[:,:2]).prod(-1).expand(len(points),-1).clone();area[~eligible]=float('inf')
        values,idx=area.min(1); mask=torch.isfinite(values)
        obj[b,mask]=1
        if mask.any(): positives.append((boxes[b,mask],gt[idx[mask]]))
    prob=logits.sigmoid();bce=F.binary_cross_entropy_with_logits(logits,obj,reduction='none')
    pt=torch.where(obj.bool(),prob,1-prob);alpha=torch.where(obj.bool(),.25,.75)
    cls=(alpha*(1-pt).pow(2)*bce).sum()/obj.sum().clamp(min=1)
    reg=torch.cat([1-giou(a,g) for a,g in positives]).mean() if positives else boxes.sum()*0
    return cls+5*reg, {'objectness':float(cls.detach()),'box':float(reg.detach()),'positives':int(obj.sum())}

def nms(boxes,scores,threshold=.5):
    order=scores.argsort(descending=True); keep=[]
    while order.numel():
        idx=order[0];keep.append(idx)
        if len(order)==1:break
        order=order[1:][iou(boxes[idx],boxes[order[1:]])<=threshold]
    return torch.stack(keep) if keep else torch.empty(0,dtype=torch.long,device=boxes.device)

@torch.no_grad()
def predict(model,x,conf=.25,nms_iou=.5):
    boxes,logits,_,_=decode(model(x));results=[]
    for bb,ss in zip(boxes,logits.sigmoid()):
        mask=ss>=conf;bb=bb[mask].clamp(0,x.shape[-1]);ss=ss[mask]
        order=ss.argsort(descending=True)[:1000];bb=bb[order];ss=ss[order]
        k=nms(bb,ss,nms_iou)[:100];results.append((bb[k],ss[k]))
    return results
