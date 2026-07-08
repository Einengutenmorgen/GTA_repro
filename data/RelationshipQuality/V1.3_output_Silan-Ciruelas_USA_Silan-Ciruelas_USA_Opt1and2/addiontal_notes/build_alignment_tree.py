#!/usr/bin/env python3
"""
Two hierarchical trees side by side with cross-level match lines and an
in-page threshold slider. Interactive standalone HTML (plotly.js).

  LEFT  = human codebook :  Theme  -> Code
  RIGHT = LLM output     :  Axial  -> Open code

Every node is drawn whether it matches or not. Cross-tree lines connect
matches at the corresponding levels:

  * Theme  <-> Axial      (upper-level match lines)
  * Code   <-> Open code  (lower-level match lines)

The similarity is computed ONCE per level (full matrix), embedded in the
page, and thresholded live by the slider in the browser -- no recompute,
no re-embedding when you drag it.

Inputs are the pipeline JSON files directly (raw shapes accepted via
alignment.normalize_codes). The LLM axial JSON must carry
`supporting_open_codes` so the Axial->Open tree edges can be drawn.

Deps: sentence-transformers, numpy, plotly (all already in the project).

Usage
-----
  python build_alignment_tree.py \
      --human-open  human_open_USA.json \
      --llm-open    output_open_codes.json \
      --llm-axial   output_axial_codes.json \
      --out         alignment_tree_USA.html \
      [--init-threshold 0.5] [--model all-MiniLM-L6-v2]
"""
import argparse
import json

import numpy as np

from alignment import normalize_codes, _embed_texts, DEFAULT_MODEL, DEFAULT_EMBED_CACHE


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _texts(codes, enrich=True):
    out = []
    for c in codes:
        code = str(c["code"]).strip()
        d = c.get("definition")
        out.append(f"{code}: {str(d).strip()}" if (enrich and d) else code)
    return out


def _matrix(a_codes, b_codes, model, cache_path):
    """Full cosine matrix (len(a) x len(b)); rows a, cols b."""
    if not a_codes or not b_codes:
        return np.zeros((len(a_codes), len(b_codes)), dtype=np.float32)
    av = _embed_texts(_texts(a_codes), model, cache_path)
    bv = _embed_texts(_texts(b_codes), model, cache_path)
    return (av @ bv.T).astype(np.float32)


def build_payload(human_open, llm_open, llm_axial, model, cache_path):
    # ---- human tree: Theme -> Code ---------------------------------------
    human = normalize_codes(human_open, "human_open")
    human_themes = list(dict.fromkeys(
        (h.get("theme") or "(no theme)") for h in human))
    human_codes = [str(h["code"]).strip() for h in human]
    human_code_theme = [(h.get("theme") or "(no theme)") for h in human]

    # theme-level "codes" for embedding = the theme label itself
    human_theme_dicts = [{"code": t, "definition": ""} for t in human_themes]

    # ---- LLM tree: Axial -> Open -----------------------------------------
    axial = normalize_codes(llm_axial, "llm_axial")
    axial_names = [str(a["code"]).strip() for a in axial]
    # supporting_open_codes lives on the ORIGINAL axial rows (pre-normalize)
    support = {}
    for row in llm_axial:
        name = str(row.get("axial_category", "")).strip()
        if name:
            support[name] = [str(s).strip() for s in row.get("supporting_open_codes", [])]

    opens = normalize_codes(llm_open, "llm_open")
    open_names = [str(o["code"]).strip() for o in opens]
    open_set = set(open_names)

    # open -> parent axial (an open code may be claimed by >=0 axial cats)
    open_parent = {o: None for o in open_names}
    for ax_name, kids in support.items():
        for k in kids:
            if k in open_parent:
                open_parent[k] = ax_name

    # ---- similarity matrices (computed once) -----------------------------
    m_theme_axial = _matrix(human_theme_dicts, axial, model, cache_path)   # themes x axial
    m_code_open = _matrix(human, opens, model, cache_path)                 # codes  x open

    return {
        "human_themes": human_themes,
        "human_codes": human_codes,
        "human_code_theme": human_code_theme,
        "axial_names": axial_names,
        "open_names": open_names,
        "open_parent": [open_parent[o] for o in open_names],
        "axial_support": {a: [k for k in support.get(a, []) if k in open_set]
                          for a in axial_names},
        "m_theme_axial": m_theme_axial.round(4).tolist(),
        "m_code_open": m_code_open.round(4).tolist(),
    }


HTML = r"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Alignment trees</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
 body{font:14px -apple-system,Segoe UI,Roboto,sans-serif;margin:0;background:#fff}
 #bar{position:sticky;top:0;background:#f8fafc;border-bottom:1px solid #e2e8f0;
      padding:10px 16px;display:flex;gap:14px;align-items:center;z-index:10}
 #bar b{font-size:15px}
 #thr{width:280px}
 #val{font-variant-numeric:tabular-nums;font-weight:600;min-width:36px}
 .lg{font-size:12px;color:#555;margin-left:auto}
 .sw{display:inline-block;width:22px;height:0;border-top:3px solid;vertical-align:middle;margin-right:3px}
 #chart{width:100%}
</style></head><body>
<div id="bar">
 <b>Alignment trees</b>
 <label>threshold <input id="thr" type="range" min="0.2" max="0.8" step="0.01" value="__INIT__"></label>
 <span id="val">__INIT__</span>
 <span id="cov"></span>
 <span class="lg">
   <span class="sw" style="border-color:#64748b"></span>tree edge
   <span class="sw" style="border-color:#16a34a"></span>theme\u2013axial
   <span class="sw" style="border-color:#0ea5e9"></span>code\u2013open
 </span>
</div>
<div id="chart"></div>
<script>
const D = __PAYLOAD__;
const XH_T=0.0, XH_C=1.3, XL_A=3.5, XL_O=2.2;   // x: humanTheme, humanCode, llmAxial, llmOpen
const GAP=1.0;

// ---- y layout: codes stacked, grouped by theme; opens stacked, grouped by axial
function humanLayout(){
  const codeY={}, themeY={}, order=[];
  const byTheme={};
  D.human_themes.forEach(t=>byTheme[t]=[]);
  D.human_codes.forEach((c,i)=>byTheme[D.human_code_theme[i]].push(c));
  let y=0;
  D.human_themes.forEach(t=>{
    const ys=[];
    byTheme[t].forEach(c=>{ codeY[c]=-y; ys.push(-y); y++; });
    themeY[t]= ys.length? ys.reduce((a,b)=>a+b,0)/ys.length : -y;
    y+=0.6; // gap between theme groups
  });
  return {codeY,themeY,height:y};
}
function llmLayout(){
  const openY={}, axialY={};
  const byAx={}; D.axial_names.forEach(a=>byAx[a]=[]);
  const orphans=[];
  D.open_names.forEach((o,i)=>{ const p=D.open_parent[i]; (p&&byAx[p]?byAx[p]:orphans).push(o); });
  let y=0;
  D.axial_names.forEach(a=>{
    const ys=[];
    byAx[a].forEach(o=>{ openY[o]=-y; ys.push(-y); y++; });
    axialY[a]= ys.length? ys.reduce((a,b)=>a+b,0)/ys.length : -y;
    y+=0.6;
  });
  orphans.forEach(o=>{ openY[o]=-y; y++; });   // unclaimed open codes at bottom
  return {openY,axialY,height:y};
}

const H=humanLayout(), L=llmLayout();

function build(thr){
  const traces=[];
  const line=(x0,y0,x1,y1,col,w)=>({x:[x0,x1],y:[y0,y1],mode:'lines',
     line:{color:col,width:w},hoverinfo:'none',showlegend:false});

  // tree edges (grey) : theme->code
  let tx=[],ty=[];
  D.human_codes.forEach((c,i)=>{const t=D.human_code_theme[i];
     tx.push(XH_T,XH_C,null); ty.push(H.themeY[t],H.codeY[c],null);});
  traces.push({x:tx,y:ty,mode:'lines',line:{color:'rgba(100,116,139,.45)',width:1},hoverinfo:'none',showlegend:false});
  // tree edges : axial->open
  let lx=[],ly=[];
  D.axial_names.forEach(a=>{(D.axial_support[a]||[]).forEach(o=>{
     if(o in L.openY){lx.push(XL_A,XL_O,null); ly.push(L.axialY[a],L.openY[o],null);}});});
  traces.push({x:lx,y:ly,mode:'lines',line:{color:'rgba(100,116,139,.45)',width:1},hoverinfo:'none',showlegend:false});

  // cross match lines: theme<->axial (green), above threshold
  let gx=[],gy=[];
  const matchedTheme=new Set();
  D.human_themes.forEach((t,i)=>D.axial_names.forEach((a,j)=>{
     if(D.m_theme_axial[i][j]>=thr){gx.push(XH_T,XL_A,null);gy.push(H.themeY[t],L.axialY[a],null);matchedTheme.add(i);}}));
  traces.push({x:gx,y:gy,mode:'lines',line:{color:'rgba(22,163,74,.5)',width:1.5},hoverinfo:'none',showlegend:false});

  // cross match lines: code<->open (blue)
  let bx=[],by=[]; const matchedCode=new Set();
  D.human_codes.forEach((c,i)=>D.open_names.forEach((o,j)=>{
     if(D.m_code_open[i][j]>=thr){bx.push(XH_C,XL_O,null);by.push(H.codeY[c],L.openY[o],null);matchedCode.add(i);}}));
  traces.push({x:bx,y:by,mode:'lines',line:{color:'rgba(14,165,233,.4)',width:1},hoverinfo:'none',showlegend:false});

  // nodes
  const node=(xs,ys,txt,col,sz,pos,name)=>({x:xs,y:ys,text:txt,mode:'markers+text',
     textposition:pos,textfont:{size:9},marker:{size:sz,color:col},
     hovertext:txt,hoverinfo:'text',name:name,showlegend:false});

  traces.push(node(D.human_themes.map(_=>XH_T),D.human_themes.map(t=>H.themeY[t]),
     D.human_themes,'#334155',12,'middle left','themes'));
  const cCol=D.human_codes.map((_,i)=>matchedCode.has(i)?'#0ea5e9':'#cbd5e1');
  traces.push(node(D.human_codes.map(_=>XH_C),D.human_codes.map(c=>H.codeY[c]),
     D.human_codes,cCol,7,'middle left','codes'));
  traces.push(node(D.axial_names.map(_=>XL_A),D.axial_names.map(a=>L.axialY[a]),
     D.axial_names,'#d97706',13,'middle right','axial'));
  const oCol=D.open_names.map((_,j)=>{
     for(let i=0;i<D.human_codes.length;i++) if(D.m_code_open[i][j]>=thr) return '#f59e0b';
     return '#e5c9a0';});
  traces.push(node(D.open_names.map(_=>XL_O),D.open_names.map(o=>L.openY[o]),
     D.open_names,oCol,6,'middle right','open'));

  const covC=(matchedCode.size/Math.max(1,D.human_codes.length));
  document.getElementById('cov').textContent=
     `\u2014 code coverage ${(covC*100).toFixed(0)}% (${matchedCode.size}/${D.human_codes.length})`;

  const height=Math.max(700, Math.max(H.height,L.height)*24);
  const layout={height,margin:{l:10,r:10,t:10,b:10},plot_bgcolor:'#fff',paper_bgcolor:'#fff',
     xaxis:{visible:false,range:[XH_T-1.5,XL_A+1.5]},yaxis:{visible:false},
     annotations:[
       {x:XH_T,y:GAP,text:'<b>THEME</b>',showarrow:false,font:{size:11,color:'#666'}},
       {x:XH_C,y:GAP,text:'<b>HUMAN CODE</b>',showarrow:false,font:{size:11,color:'#666'}},
       {x:XL_O,y:GAP,text:'<b>LLM OPEN</b>',showarrow:false,font:{size:11,color:'#666'}},
       {x:XL_A,y:GAP,text:'<b>LLM AXIAL</b>',showarrow:false,font:{size:11,color:'#666'}}]};
  Plotly.react('chart',traces,layout,{responsive:true,displaylogo:false});
}

const thr=document.getElementById('thr'), val=document.getElementById('val');
thr.addEventListener('input',()=>{val.textContent=(+thr.value).toFixed(2);build(+thr.value);});
build(+thr.value);
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--human-open", required=True)
    ap.add_argument("--llm-open", required=True)
    ap.add_argument("--llm-axial", required=True)
    ap.add_argument("--out", default="alignment_tree.html")
    ap.add_argument("--init-threshold", type=float, default=0.5)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--embed-cache", default=DEFAULT_EMBED_CACHE)
    a = ap.parse_args()

    payload = build_payload(_load(a.human_open), _load(a.llm_open),
                            _load(a.llm_axial), a.model, a.embed_cache)
    html = (HTML.replace("__PAYLOAD__", json.dumps(payload))
                .replace("__INIT__", f"{a.init_threshold:.2f}"))
    with open(a.out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {a.out}: {len(payload['human_themes'])} themes, "
          f"{len(payload['human_codes'])} human codes, "
          f"{len(payload['axial_names'])} axial, {len(payload['open_names'])} open")


if __name__ == "__main__":
    main()