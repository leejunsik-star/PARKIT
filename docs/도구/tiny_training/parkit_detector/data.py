from pathlib import Path
import random
import numpy as np
import torch
from PIL import Image,ImageEnhance,ImageOps,ImageDraw
from torch.utils.data import Dataset

def letterbox(image,size=640):
    w,h=image.size; scale=min(size/w,size/h);nw,nh=round(w*scale),round(h*scale)
    pad=((size-nw)//2,(size-nh)//2);canvas=Image.new('RGB',(size,size),(114,114,114))
    canvas.paste(image.resize((nw,nh)),pad)
    return canvas,(nw/w,nh/h),pad

def tensor(image): return torch.from_numpy(np.array(image,copy=True)).permute(2,0,1).float()/255

class Cars(Dataset):
    def __init__(self,root='.',split='train',size=640,augment=False,limit=None):
        self.root=Path(root);self.split=split;self.size=size;self.augment=augment
        self.files=sorted((self.root/'images'/split).glob('*.jpg'))
        if limit:self.files=self.files[:limit]
        if not self.files:raise ValueError(f'No images in {self.root}/images/{split}')
    def __len__(self):return len(self.files)
    def base(self,i):
        p=self.files[i];im=Image.open(p).convert('RGB');w,h=im.size
        rows=[list(map(float,r.split())) for r in (self.root/'labels'/self.split/(p.stem+'.txt')).read_text().splitlines() if r.strip()]
        boxes=[]
        for c,x,y,bw,bh in rows:
            if c!=0 or not all(0<=v<=1 for v in (x,y,bw,bh)) or bw<=0 or bh<=0:raise ValueError(f'Invalid label {p}')
            boxes.append([(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h])
        boxes=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4)
        im,scale,pad=letterbox(im,self.size)
        boxes=boxes*torch.tensor([*scale,*scale])+torch.tensor([*pad,*pad])
        return im,boxes,p.name
    def __getitem__(self,i):
        im,boxes,name=self.base(i)
        if self.augment:
            if random.random()<.25:
                # Four half-size letterboxed training images in a 2x2 canvas.
                canvas=Image.new('RGB',(self.size,self.size),(114,114,114));all_boxes=[]
                half=self.size//2
                for j,idx in enumerate([i]+[random.randrange(len(self)) for _ in range(3)]):
                    part,b,_=self.base(idx);offset=torch.tensor([j%2*half,j//2*half]*2)
                    canvas.paste(part.resize((half,half)),(j%2*half,j//2*half))
                    all_boxes.append(b*(half/self.size)+offset)
                im=canvas;boxes=torch.cat(all_boxes)
            # Isotropic scale and translation, PIL inverse affine for image pixels.
            factor=random.uniform(.85,1.15);dx=random.uniform(-.06,.06)*self.size;dy=random.uniform(-.06,.06)*self.size
            im=im.transform(im.size,Image.Transform.AFFINE,(1/factor,0,-dx/factor,0,1/factor,-dy/factor),resample=Image.Resampling.BILINEAR,fillcolor=(114,114,114))
            transformed=boxes*factor+torch.tensor([dx,dy,dx,dy]);clipped=transformed.clamp(0,self.size)
            before=(transformed[:,2:]-transformed[:,:2]).prod(-1).clamp(min=1e-6)
            dims=clipped[:,2:]-clipped[:,:2];keep=(dims.min(-1).values>=2)&(dims.prod(-1)/before>=.3)
            boxes=clipped[keep]
            if random.random()<.5:
                im=ImageOps.mirror(im);old=boxes.clone();boxes[:,0]=self.size-old[:,2];boxes[:,2]=self.size-old[:,0]
            im=ImageEnhance.Brightness(im).enhance(random.uniform(.65,1.35))
            im=ImageEnhance.Color(im).enhance(random.uniform(.65,1.35))
            im=ImageEnhance.Contrast(im).enhance(random.uniform(.8,1.2))
            hsv=np.array(im.convert('HSV'),copy=True)
            hsv[:,:,0]=((hsv[:,:,0].astype(np.int16)+random.randint(-8,8))%256).astype(np.uint8)
            im=Image.fromarray(hsv,mode='HSV').convert('RGB')
        return tensor(im),boxes,name

def collate(batch):
    x,y,n=zip(*batch);return torch.stack(x),list(y),n

def draw(image,boxes,path,scores=None):
    if torch.is_tensor(image):image=Image.fromarray((image.permute(1,2,0).cpu().numpy()*255).clip(0,255).astype('uint8'))
    else:image=image.copy()
    d=ImageDraw.Draw(image)
    for i,b in enumerate(boxes):
        d.rectangle(tuple(map(float,b)),outline='red',width=2)
        if scores is not None:d.text((float(b[0]),float(b[1])),f'{float(scores[i]):.2f}',fill='red')
    image.save(path)
