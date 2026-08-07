const tip=document.getElementById('tip');
function mtip(html,x,y){tip.innerHTML=html;tip.style.opacity=1;tip.style.left=(x+15)+'px';tip.style.top=(y+15)+'px';}
document.querySelectorAll('[data-tip]').forEach(e=>{
  e.addEventListener('mousemove',ev=>mtip(e.getAttribute('data-tip'),ev.clientX,ev.clientY));
  e.addEventListener('mouseleave',()=>tip.style.opacity=0);});
for(const key in LINES){
  const m=LINES[key],hit=document.getElementById(key+'-hit'),svg=hit.ownerSVGElement,
    dot=document.getElementById(key+'-dot'),cross=document.getElementById(key+'-cross');
  const X=x=>m.l+(x-m.xmin)/(m.xmax-m.xmin)*(m.W-m.l-m.r);
  const Y=v=>{const vv=m.log?Math.log(v):v;return m.H-m.b-(vv-m.ymin)/(m.ymax-m.ymin)*(m.H-m.t-m.b);};
  hit.addEventListener('mousemove',ev=>{
    const r=svg.getBoundingClientRect(),sx=(ev.clientX-r.left)/r.width*m.W;
    const dx=m.xmin+(sx-m.l)/(m.W-m.l-m.r)*(m.xmax-m.xmin);
    let bi=0,bd=1e18;for(let i=0;i<m.pts.length;i++){const d=Math.abs(m.pts[i][0]-dx);if(d<bd){bd=d;bi=i;}}
    const p=m.pts[bi],px=X(p[0]),py=Y(p[1]);
    dot.setAttribute('cx',px);dot.setAttribute('cy',py);dot.style.opacity=1;
    cross.setAttribute('x1',px);cross.setAttribute('x2',px);cross.style.opacity=1;
    const dt=new Date(p[0]).toLocaleDateString('en',{year:'numeric',month:'short',day:'numeric'});
    mtip('<b>'+dt+'</b><br>'+(m.pct?(p[1]*100).toFixed(1)+'%':p[1].toFixed(2)),ev.clientX,ev.clientY);});
  hit.addEventListener('mouseleave',()=>{tip.style.opacity=0;dot.style.opacity=0;cross.style.opacity=0;});
}
