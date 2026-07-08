import json, sys
payload_path = sys.argv[1]
out_path = sys.argv[2]
data = open(payload_path, encoding="utf-8").read()

HTML = r'''<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Edge mapping \u2014 __STAGE__</title>
<style>
*{box-sizing:border-box}
body{margin:0;font:15px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;background:#f1f5f9;color:#0f172a}
#top{position:sticky;top:0;background:#fff;border-bottom:1px solid #e2e8f0;padding:10px 16px;
 display:flex;gap:14px;align-items:center;flex-wrap:wrap;z-index:5}
#top b{font-size:16px}
.prog{flex:1;min-width:140px;height:8px;background:#e2e8f0;border-radius:6px;overflow:hidden}
.prog>i{display:block;height:100%;background:#38bdf8;width:0}
button{font:inherit;border:1px solid #cbd5e1;background:#fff;border-radius:8px;padding:6px 11px;cursor:pointer}
button:hover{background:#f8fafc}
button.pri{background:#38bdf8;border-color:#38bdf8;color:#04293b;font-weight:600}
#wrap{max-width:1180px;margin:18px auto;padding:0 14px;display:grid;grid-template-columns:360px 1fr;gap:16px}
.panel{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:16px 18px}
.left{position:sticky;top:64px;align-self:start;max-height:calc(100vh - 84px);overflow:auto}
.theme{display:inline-block;font-size:12px;color:#0369a1;background:#e0f2fe;padding:2px 9px;border-radius:20px}
h2{margin:8px 0 4px;font-size:20px;line-height:1.25}
.def{color:#475569;font-size:13.5px;margin:4px 0}
.ex{color:#64748b;font-size:12.5px;font-style:italic;border-left:3px solid #e2e8f0;padding-left:9px;margin-top:8px}
.tallybox{margin-top:14px;font-size:13px;color:#334155;background:#f8fafc;border-radius:8px;padding:8px 10px}
.tallybox span{display:inline-block;margin-right:10px;font-weight:600}
.search{width:100%;padding:8px 10px;border:1px solid #cbd5e1;border-radius:8px;font:inherit;margin-bottom:8px}
.filters{display:flex;gap:6px;margin-bottom:10px;flex-wrap:wrap;font-size:12px}
.filters button{padding:3px 9px;border-radius:16px}
.filters button.on{background:#334155;color:#fff;border-color:#334155}
.edge{display:flex;align-items:center;gap:8px;border:1px solid #e8eef4;border-radius:9px;padding:7px 9px;margin:4px 0}
.edge.full{border-color:#16a34a;background:#f0fdf4}
.edge.partial{border-color:#eab308;background:#fefce8}
.edge.negation{border-color:#dc2626;background:#fef2f2}
.sim{font-variant-numeric:tabular-nums;font-weight:600;font-size:12px;min-width:42px;text-align:center;border-radius:6px;padding:2px 3px}
.s-hi{background:#bbf7d0;color:#166534}.s-md{background:#fef08a;color:#854d0e}.s-lo{background:#fed7aa;color:#9a3412}.s-xx{background:#eef2f6;color:#64748b}
.etext{flex:1;font-size:13.5px}
.btns{display:flex;gap:4px}
.btns button{padding:2px 8px;font-size:12px;border-radius:6px}
.btns button.F.on{background:#16a34a;color:#fff;border-color:#16a34a}
.btns button.P.on{background:#eab308;color:#fff;border-color:#eab308}
.btns button.N.on{background:#dc2626;color:#fff;border-color:#dc2626}
.nav{display:flex;justify-content:space-between;margin-top:14px}
.kbd{font:11px monospace;background:#f1f5f9;border:1px solid #cbd5e1;border-radius:4px;padding:1px 5px;color:#475569}
.hint{font-size:12px;color:#94a3b8;margin-top:10px;line-height:1.7}
.done{color:#16a34a;font-weight:600}
.count{color:#94a3b8;font-size:12px;margin:6px 0}
</style></head><body>
<div id="top">
 <b>Edge mapping \u2014 __STAGE__</b>
 <span id="counter" style="color:#64748b"></span>
 <div class="prog"><i id="bar"></i></div>
 <span id="savestate" style="font-size:12px;color:#64748b"></span>
 <button onclick="exp('json')">Export JSON</button>
 <button onclick="exp('csv')">Export CSV</button>
 <button onclick="if(confirm('Clear ALL edge labels?')){localStorage.removeItem(KEY);location.reload()}">Reset</button>
</div>
<div id="wrap">
 <div class="panel left" id="left"></div>
 <div class="panel" id="right"></div>
</div>
<script>
const D = __DATA__;
const KEY = "edgemap_"+D.stage+"_v1";
// edges[humanIdx] = { llmIdx: "full"|"partial"|"negation" }   (absence = no match)
let edges = JSON.parse(localStorage.getItem(KEY) || "{}");
let idx = +(localStorage.getItem(KEY+"_idx")||0);
let filter = "all";
let query = "";

const esc=s=>(s==null?"":String(s)).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));
const simClass=v=>v>=.55?"s-hi":v>=.45?"s-md":v>=.35?"s-lo":"s-xx";
const LT={full:"F",partial:"P",negation:"N"};

function eObj(i){ if(!edges[i]) edges[i]={}; return edges[i]; }
function save(){
  localStorage.setItem(KEY, JSON.stringify(edges));
  localStorage.setItem(KEY+"_idx", idx);
  let touched=0, full=0, part=0, neg=0;
  for(const k in edges) for(const j in edges[k]){ touched++;
    if(edges[k][j]==="full")full++; else if(edges[k][j]==="partial")part++; else neg++; }
  const humansTouched=Object.values(edges).filter(o=>Object.keys(o).length).length;
  document.getElementById('savestate').innerHTML=
    `<span class="done">\u2713</span> ${humansTouched}/${D.human.length} coded \u00b7 ${full}F ${part}P ${neg}N`;
}

function setEdge(hi, li, type){
  const o=eObj(hi);
  if(o[li]===type){ delete o[li]; }     // toggle off
  else { o[li]=type; }
  renderRight(); save();
}

function renderLeft(){
  const h=D.human[idx];
  const o=edges[idx]||{};
  let f=0,p=0,n=0; for(const j in o){ if(o[j]==="full")f++;else if(o[j]==="partial")p++;else n++; }
  document.getElementById('left').innerHTML=`
    <span class="theme">${esc(h.theme||'(no theme)')}</span>
    <h2>${esc(h.code)}</h2>
    ${h.definition?`<div class="def">${esc(h.definition)}</div>`:""}
    ${h.example?`<div class="ex">${esc(h.example)}</div>`:""}
    <div class="tallybox">edges on this code:
      <span style="color:#16a34a">${f} full</span>
      <span style="color:#a16207">${p} partial</span>
      <span style="color:#b91c1c">${n} negation</span></div>
    <div class="nav">
      <button onclick="go(-1)">\u2190 Prev</button>
      <button class="pri" onclick="go(1)">Next human code \u2192</button></div>
    <div class="hint">Type edges on the right. Untouched LLM codes = <b>no match</b> (implicit).<br>
      Keys: <span class="kbd">j/k</span> or <span class="kbd">\u2193/\u2191</span> move highlight \u00b7
      <span class="kbd">f</span> full \u00b7 <span class="kbd">p</span> partial \u00b7 <span class="kbd">n</span> negation \u00b7
      <span class="kbd">\u2190\u2192</span> prev/next code</div>`;
  document.getElementById('counter').textContent=`${idx+1} / ${D.human.length}`;
  document.getElementById('bar').style.width=(100*(idx+1)/D.human.length)+"%";
}

let hi_row = 0;   // keyboard-highlighted visible row
function renderRight(){
  const rank=D.ranked[idx];        // [[llmIdx, sim], ...] nearest first
  const o=edges[idx]||{};
  const q=query.toLowerCase();
  let visible=rank.filter(([j,s])=>{
    if(query && !D.llm[j].toLowerCase().includes(q)) return false;
    if(filter==="typed" && !o[j]) return false;
    if(filter==="untyped" && o[j]) return false;
    return true;
  });
  if(hi_row>=visible.length) hi_row=Math.max(0,visible.length-1);
  const rows=visible.map(([j,s],vi)=>{
    const t=o[j]||"";
    const cls=t?` ${t}`:"";
    const hl=vi===hi_row?" style=\"outline:2px solid #38bdf8\"":"";
    return `<div class="edge${cls}"${hl} data-vi="${vi}" data-j="${j}">
      <span class="sim ${simClass(s)}">${s.toFixed(3)}</span>
      <span class="etext">${esc(D.llm[j])}</span>
      <span class="btns">
        <button class="F${t==='full'?' on':''}" onclick="setEdge(${idx},${j},'full')">full</button>
        <button class="P${t==='partial'?' on':''}" onclick="setEdge(${idx},${j},'partial')">partial</button>
        <button class="N${t==='negation'?' on':''}" onclick="setEdge(${idx},${j},'negation')">neg</button>
      </span></div>`;
  }).join("");
  document.getElementById('right').innerHTML=`
    <input class="search" id="q" placeholder="search ${D.llm.length} ${esc(D.llm_label)}s\u2026" value="${esc(query)}"
       oninput="query=this.value;renderRight();document.getElementById('q').focus()">
    <div class="filters">
      ${["all","typed","untyped"].map(f=>`<button class="${filter===f?'on':''}" onclick="filter='${f}';renderRight()">${f}</button>`).join("")}
      <span class="count">${visible.length} shown / ${D.llm.length}</span>
    </div>
    ${rows||'<div class="count">no matches for this search</div>'}`;
}

function render(){ renderLeft(); renderRight(); save(); }
function go(d){ idx=Math.max(0,Math.min(D.human.length-1,idx+d)); hi_row=0; query=""; filter="all"; render(); window.scrollTo(0,0); }

document.addEventListener('keydown',e=>{
  if(e.target.id==='q'){ if(e.key==='Escape'){e.target.blur();} return; }
  if(e.key==='ArrowRight'){e.preventDefault();go(1);}
  else if(e.key==='ArrowLeft'){go(-1);}
  else if(e.key==='j'||e.key==='ArrowDown'){e.preventDefault();hi_row++;renderRight();}
  else if(e.key==='k'||e.key==='ArrowUp'){e.preventDefault();hi_row=Math.max(0,hi_row-1);renderRight();}
  else if(e.key==='f'||e.key==='p'||e.key==='n'){
    const el=document.querySelector(`.edge[data-vi="${hi_row}"]`);
    if(el){ const j=+el.dataset.j; setEdge(idx,j,{f:'full',p:'partial',n:'negation'}[e.key]); }
  } else if(e.key==='/'){ e.preventDefault(); document.getElementById('q').focus(); }
});

function exp(fmt){
  const out=[];
  for(let i=0;i<D.human.length;i++){
    const o=edges[i]||{};
    for(const j in o){
      const sim=(D.ranked[i].find(r=>r[0]==+j)||[0,null])[1];
      out.push({human_code:D.human[i].code, theme:D.human[i].theme,
        llm_code:D.llm[+j], edge:o[j], similarity:sim});
    }
  }
  let blob;
  if(fmt==='json') blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'});
  else{ const cols=["human_code","theme","llm_code","edge","similarity"];
    const csv=[cols.join(",")].concat(out.map(r=>cols.map(k=>{
      let v=(r[k]==null?"":String(r[k])); if(/[",\n]/.test(v))v='"'+v.replace(/"/g,'""')+'"'; return v;
    }).join(","))).join("\n");
    blob=new Blob([csv],{type:'text/csv'}); }
  const a=document.createElement('a'); a.href=URL.createObjectURL(blob);
  a.download="edgemap_"+D.stage+"."+fmt; a.click();
}
render();
</script></body></html>'''

html = HTML.replace("__DATA__", data)
import json as _j
stage = _j.loads(data)["stage"]
html = html.replace("__STAGE__", stage)
open(out_path,"w",encoding="utf-8").write(html)
print("wrote", out_path, "| stage", stage)
