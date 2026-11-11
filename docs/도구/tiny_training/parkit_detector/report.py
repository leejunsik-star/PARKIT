from html import escape

def curve(rows,path):
    panels=[]
    for j,key in enumerate(['loss','precision','recall','mAP50']):
        values=[r[key] for r in rows];maximum=max(max(values),1e-8) if key=='loss' else 1
        coords=' '.join(f'{60+i*680/max(len(values)-1,1):.2f},{j*180+155-v/maximum*120:.2f}' for i,v in enumerate(values))
        panels.append(f'<text x="20" y="{j*180+20}">{escape(key)} (max {maximum:.3f})</text><path d="M60 {j*180+30} V{j*180+155} H740" fill="none" stroke="gray"/><polyline points="{coords}" fill="none" stroke="#16806a" stroke-width="2"/>')
    path.write_text('<svg xmlns="http://www.w3.org/2000/svg" width="800" height="740"><rect width="800" height="740" fill="white"/>'+''.join(panels)+'</svg>')
