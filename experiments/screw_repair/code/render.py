"""Public force/preload/opening/support readings; no latent/target pixels."""
import numpy as np
from PIL import Image, ImageDraw

DIGITS=['abcdef','bc','abdeg','abcdg','bcfg','acdfg','acdefg','abc','abcdefg','abcdfg']
SEG={'a':(1,0,5,0),'b':(6,1,6,4),'c':(6,6,6,9),'d':(1,10,5,10),
     'e':(0,6,0,9),'f':(0,1,0,4),'g':(1,5,5,5)}
ROWS=[137,151,165,179]; START=120

def render(values,event,appearance,opening=0,braced=False):
    im=Image.new('RGB',(224,192),tuple(map(int,appearance))); dr=ImageDraw.Draw(im)
    y=45-int(round(min(opening,1.6)*17))
    dr.rectangle((44,80,159,91),fill=(88,109,125))
    dr.rectangle((50,y,151,y+12),fill=(170,183,193))
    dr.rectangle((96,y-5,108,80),fill=(97,114,130))
    dr.rectangle((82,y-13,122,y-5),fill=(132,148,160))
    for yy in range(y+13,78,5): dr.line((95,yy+3,110,yy),fill=(54,72,86))
    if braced: dr.line((48,80,53,y+11),fill=(140,113,63),width=5)
    if event=='repair': dr.line((115,y-12,182,y-21),fill=(64,79,95),width=5)
    dr.text((7,5),event.upper(),fill=(24,44,61))
    # An identical generic repair annotation is used for every operator. Operator
    # identity is never an input label; post-repair measurements identify state.
    for row,label,val in zip(ROWS,['F N','P N','x mm','support N/mm'],values):
        n=round(float(val)*1000); assert 0<=n<=999999,(values,n)
        dr.rectangle((0,row-1,223,row+11),fill=(245,247,248))
        dr.text((3,row),label,fill=(32,43,53))
        for j,char in enumerate(f'{n:06d}'):
            base=START+j*12
            for name in DIGITS[int(char)]:
                x1,y1,x2,y2=SEG[name]; dr.line((base+x1,row+y1,base+x2,row+y2),fill=(10,10,10))
        dr.rectangle((START+2*12+9,row+9,START+2*12+10,row+10),fill=(10,10,10))
    return np.asarray(im)

def decode(rgb):
    result=[]
    for row in ROWS:
        digits=''
        for j in range(6):
            active=''
            for name,(x1,y1,x2,y2) in SEG.items():
                x=START+j*12+(x1+x2)//2; y=row+(y1+y2)//2
                if np.array_equal(rgb[y,x],(10,10,10)): active+=name
            digits+=str(next(i for i,s in enumerate(DIGITS) if set(s)==set(active)))
        result.append(int(digits)/1000)
    return np.asarray(result,float)

def verify():
    v=np.array([372.123,160.0,.12,240.]); im=render(v,'probe',(249,248,247))
    assert np.array_equal(decode(im),v)
    return {'seven_segment_roundtrip':True}
