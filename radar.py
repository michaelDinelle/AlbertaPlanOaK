from pathlib import Path
import math, statistics, html, re
out=Path(__file__).resolve().parent
source=(out/'README.md').read_text()
count=int(re.search(r'all ([\d,]+) environments',source).group(1).replace(',',''))
axes=['Basic','Credit assignment','Exploration','Generalization','Memory','Noise','Scale']
groups={k:[] for k in axes}
for line in source.splitlines():
    p=[v.strip() for v in line.split('|')]
    if len(p)!=8 or not p[1].endswith('-v0'): continue
    n=p[1].removesuffix('-cont-v0'); oak,oracle,random,ppo,score=map(float,p[2:7])
    if n=='Catch-scale': g='Scale'
    elif '-slip' in n or '-noise' in n: g='Noise'
    elif n.startswith(('MemoryChain','TMaze','Repeat','CountRecall','Recite','Tiger','AliasedRing','HiddenMode','CoinSwap','Nexting')) or '-position' in n: g='Memory'
    elif n.startswith(('DeepSea','LargeDeepSea','JellyBeanWorld','RiverSwim','BlockingMaze','ShortcutMaze','Sokoban','FourRoomsHall','Taxi','Forager','ScentPuck')): g='Exploration'
    elif n.startswith(('Chain','Operant','Umbrella','Discount','TwoLoop','KeyDoor','Password')): g='Credit assignment'
    elif n.startswith(('Regression','Parity','SubsetParity','Permuted','Covariate','Garnet','HashedMDP','RockPaperScissors','WalkBet')) or '-drift' in n: g='Generalization'
    else: g='Basic'
    groups[g].append({'axis':g,'environment':p[1],'OaK':(oak-random)/(oracle-random),'Oracle':1.,'PPO':(ppo-random)/(oracle-random),'Random':0.})
rows=[r for rs in groups.values() for r in rs]
assert len(rows)==count and len({r['environment'] for r in rows})==count
agents=['OaK','Oracle','PPO','Random']
values={a:[statistics.mean(r[a] for r in groups[g]) for g in axes] for a in agents}
assert min(min(v) for v in values.values())>=-.25 and max(max(v) for v in values.values())<=1
s=['<svg xmlns="http://www.w3.org/2000/svg" width="1250" height="930" viewBox="0 0 1250 930" role="img" aria-labelledby="title desc">','<title id="title">OaK performance across seven capability axes</title>',f'<desc id="desc">Exploratory grouping of all {count} README environments. Mean normalized scores for OaK, Oracle, PPO and Random. The radial scale runs from minus 0.25 to 1. Random is zero, Oracle is one. The overall ranking is the equal-weight mean over all {count} environments, independent of the interpretive axis grouping. The source is the final 100,000 steps of the README million-step recording.</desc>','<rect width="1250" height="930" fill="#ffffff"/>','<g font-family="Arial, Helvetica, sans-serif" fill="#243044">']
def text(x,y,t,size=18,fill='#243044',anchor='start',weight='400'):
    s.append(f'<text x="{x}" y="{y}" font-size="{size}" fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{html.escape(t)}</text>')
text(54,55,'Policy comparison',30,weight='600')
text(54,86,f'Mean normalized reward · {count} environments',17,fill='#637084')
cx,cy,R=430,390,240
angles=[-2*math.pi*i/7 for i in range(7)]
def xy(i,value):
    radius=R*(value+.25)/1.25
    return cx+radius*math.cos(angles[i]),cy+radius*math.sin(angles[i])
def points(v):return ' '.join(f'{x:.2f},{y:.2f}' for i,val in enumerate(v) for x,y in [xy(i,val)])
for val in [-.0,.25,.5,.75,1.]:
    rr=R*(val+.25)/1.25
    s.append(f'<circle cx="{cx}" cy="{cy}" r="{rr}" fill="none" stroke="#cbd2dc" stroke-width="1" stroke-dasharray="2 4"/>')
for i,g in enumerate(axes):
    x,y=xy(i,1)
    s.append(f'<path d="M {cx} {cy} L {x} {y}" stroke="#d0d6df" stroke-width="1"/>')
colors={'OaK':'#5548b4','Oracle':'#52a96a','PPO':'#68a8cc','Random':'#e88152'}
for a,opacity in [('PPO',.14),('OaK',.16)]:
    s.append(f'<polygon points="{points(values[a])}" fill="{colors[a]}" fill-opacity="{opacity}" stroke="none"/>')
for a in ['Oracle','Random','PPO','OaK']:
    dash='stroke-dasharray="9 6"' if a=='Oracle' else 'stroke-dasharray="3 5"' if a=='Random' else ''
    s.append(f'<polygon points="{points(values[a])}" fill="none" stroke="{colors[a]}" stroke-width="3.5" stroke-linejoin="round" {dash}/>')
    for i,v in enumerate(values[a]):
        x,y=xy(i,v)
        if a=='PPO':s.append(f'<rect x="{x-3.5}" y="{y-3.5}" width="7" height="7" fill="{colors[a]}"/>')
        elif a=='OaK':s.append(f'<circle cx="{x}" cy="{y}" r="4.5" fill="{colors[a]}"/>')
for val in [0,.25,.5,.75,1.]:
    rr=R*(val+.25)/1.25
    text(cx+rr-2,cy+21,f'{val:g}',13,fill='#647084',anchor='middle')
text(cx-8,cy+22,'−0.25',12,fill='#647084',anchor='end')
for i,g in enumerate(axes):
    x=cx+(R+40)*math.cos(angles[i]);y=cy+(R+40)*math.sin(angles[i])
    anchor='start' if math.cos(angles[i])>.35 else 'end' if math.cos(angles[i])<-.35 else 'middle'
    text(round(x,2),round(y,2),g.title() if g=='Credit assignment' else g,20,anchor=anchor)
    text(round(x,2),round(y+21,2),f'n = {len(groups[g])}',14,fill='#687489',anchor=anchor)
overall={a:statistics.mean(r[a] for r in rows) for a in agents}
ranking=sorted(agents,key=overall.get,reverse=True)
assert all(overall[a]>=0 for a in agents)
sps={'OaK':re.search(r'([\d,]+) SPS on the [\d,]+-environment schedule',source).group(1)}
for agent in ['PPO','Random','Oracle']:
    sps[agent]=re.search(rf'{agent} ([\d,]+) steps/s',source).group(1)
text(900,222,'Overall ranking',24,weight='600')
for j,a in enumerate(ranking):
    y=278+j*76
    text(900,y,f'{j+1}. {a}',20,fill=colors[a],weight='600')
    text(1180,y,f'{overall[a]:.3f}',20,anchor='end')
    if overall[a]>0:
        s.append(f'<rect x="900" y="{y+12}" width="{280*overall[a]:.2f}" height="13" fill="{colors[a]}"/>')
    else:
        s.append(f'<path d="M 900 {y+10} v 17" stroke="{colors[a]}" stroke-width="2"/>')
    text(900,y+46,f'{sps[a]} SPS' if a in sps else 'SPS not reported',16,fill='#637084')
text(900,599,'M4 Max · steps per second',15,fill='#637084')
text(54,745,'Axis scores',17,weight='600')
columns=[245,395,545,715,875,1025,1175]
for i,g in enumerate(axes):
    text(columns[i],745,'Credit' if g=='Credit assignment' else g,15,anchor='middle')
for j,a in enumerate(ranking):
    y=775+j*26
    text(54,y,a,17,fill=colors[a],weight='600')
    for i,v in enumerate(values[a]):
        text(columns[i],y,f'{v:.3f}',16,anchor='middle')
text(54,904,'Equal weight per environment · Random = 0 · Oracle = 1 · README: final 100,000 / 1,000,000 steps',15,fill='#637084')
s.append('</g></svg>')
(out/'oak-performance-radar.svg').write_text('\n'.join(s))
print(out/'oak-performance-radar.svg')
