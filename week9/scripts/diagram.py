"""Draw the AD tuberculosis query with explicit hyperedge junctions."""
from pathlib import Path
from xml.sax.saxutils import escape
out=Path(__file__).resolve().parents[1]/'evidence/graphs.svg'
parts=['<svg xmlns="http://www.w3.org/2000/svg" width="1100" height="680" viewBox="0 0 1100 680">','<rect width="1100" height="680" fill="white"/>','<style>text{font-family:Arial,sans-serif;font-size:16px}.edge{stroke:#65778a;stroke-width:2;fill:none}</style>']
def text(x,y,s,size=16):parts.append(f'<text x="{x}" y="{y}" text-anchor="middle" style="font-size:{size}px">{escape(s)}</text>')
text(550,35,'ASIA: p(T, A=0, D=0); only T remains open',23)
# Coordinates separate variable junctions from factor boxes.
vars={'A':(110,180),'T':(300,275),'S':(670,160),'L':(510,275),'B':(880,280),'E':(540,440),'X':(320,560),'D':(820,560)}
factors=[(110,90,'p(A)',['A']),(215,210,'p(T|A)',['T','A']),(670,90,'p(S)',['S']),(560,210,'p(L|S)',['L','S']),(800,210,'p(B|S)',['B','S']),(420,355,'p(E|L,T)',['E','L','T']),(360,480,'p(X|E)',['X','E']),(740,440,'p(D|E,B)',['D','E','B'])]
for x,y,label,scope in factors:
    for v in scope:
        vx,vy=vars[v];parts.append(f'<path class="edge" d="M{x},{y} L{vx},{vy}"/>')
for v,(x,y) in vars.items():
    parts.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#203c57"/>');text(x+14,y-12,v)
for x,y,label,scope in factors:
    parts.append(f'<rect x="{x-62}" y="{y-24}" width="124" height="48" rx="5" fill="#e7f1f8" stroke="#203c57"/>')
    text(x,y-3,label);text(x,y+17,'axes: '+','.join(scope),12)
for v,x,y,label in [('A',50,270,'(1,0)'),('D',960,560,'(1,0)'),('X',320,625,'(1,1)')]:
    vx,vy=vars[v];parts.append(f'<path class="edge" d="M{vx},{vy} L{x},{y}"/><rect x="{x-28}" y="{y-14}" width="56" height="28" fill="#ffe6bd" stroke="#a76813"/>');text(x,y+5,label)
x,y=vars['T'];parts.append(f'<path d="M{x},{y} L180,335" stroke="#c54830" stroke-width="4"/>');text(150,360,'T output',17)
text(750,635,'Junctions identify one shared variable, not extra tensors.',15)
text(750,660,'Divide the two output weights by p(A=0,D=0).',15)
parts.append('</svg>');out.write_text('\n'.join(parts))
