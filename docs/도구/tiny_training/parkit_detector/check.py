"""Meaningful invariants: box transforms, decode, loss gradients and NMS."""
import torch
import tempfile
from pathlib import Path
from .model import ParkitTiny,decode,detection_loss,nms,iou
from .data import Cars,letterbox
from PIL import Image

def main():
    torch.set_num_threads(2)
    _,scale,pad=letterbox(Image.new('RGB',(1920,1080)),640)
    source=torch.tensor([[100.,200.,300.,400.]])
    packed=source*torch.tensor([*scale,*scale])+torch.tensor([*pad,*pad])
    restored=(packed-torch.tensor([*pad,*pad]))/torch.tensor([*scale,*scale])
    assert torch.allclose(source,restored,atol=1e-4)
    # 실제 사진이나 실행 위치에 의존하지 않는 임시 이미지·정답.
    with tempfile.TemporaryDirectory() as folder:
        root = Path(folder)
        (root / 'images/train').mkdir(parents=True)
        (root / 'labels/train').mkdir(parents=True)
        Image.new('RGB', (128, 128), (80, 120, 160)).save(root / 'images/train/sample.jpg')
        (root / 'labels/train/sample.txt').write_text('0 0.5 0.5 0.5 0.5\n')
        ds = Cars(root=root, size=128)
        x,gt,_=ds[0]
    assert x.shape==(3,128,128) and len(gt)>0
    assert (gt[:,2:]>gt[:,:2]).all()
    model=ParkitTiny(8);outputs=model(torch.stack([x,x]))
    boxes,logits,points,strides=decode(outputs)
    assert boxes.shape==(2,320,4) and logits.shape==(2,320)
    assert (boxes[...,2:]>boxes[...,:2]).all()
    loss,info=detection_loss(outputs,[gt,gt]);loss.backward()
    assert torch.isfinite(loss) and info['positives']>0
    assert model.heads[0].box[-1].weight.grad.abs().sum()>0
    empty,_=detection_loss(outputs,[gt[:0],gt[:0]])
    assert torch.isfinite(empty)
    b=torch.tensor([[0.,0.,10.,10.],[1.,1.,9.,9.],[30.,30.,40.,40.]])
    assert nms(b,torch.tensor([.9,.8,.7])).tolist()==[0,2]
    assert torch.allclose(iou(b,b),torch.ones(3))
    from unittest.mock import patch
    from .run import evaluate
    prediction=[(torch.tensor([[0.,0.,10.,10.],[50.,50.,60.,60.]]),torch.tensor([.9,.8]))]
    ground=torch.tensor([[0.,0.,10.,10.],[20.,20.,30.,30.]])
    loader=[(torch.zeros(1,3,64,64),[ground],['synthetic.jpg'])]
    with patch('parkit_detector.run.predict',return_value=prediction):
        metrics=evaluate(torch.nn.Identity(),loader,'cpu')
    assert metrics['precision']==.5 and metrics['recall']==.5
    assert metrics['false_positive']==1 and metrics['false_negative']==1
    assert abs(metrics['mAP50']-51/101)<1e-8
    print('PASS: letterbox dataset, output/decode, positive assignment, backward gradients, empty targets, IoU and NMS')

if __name__=='__main__':main()
