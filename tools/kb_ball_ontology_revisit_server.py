#!/usr/bin/env python
"""
BALL ONTOLOGY REVISIT -- what KIND of ball was each Round-0 finding?

Round 0 measured one thing and measured it honestly: 84 of 300 images hold at
least one visible football that is not annotated, 128 objects in all. That
result stands and this tool does not touch it.

But "visible football" turned out to bundle two different objects. The reviewer
noticed during the sweep that many of the 128 were balls held by ball boys,
spares behind the goal, or reserve balls on the touchline -- real footballs,
correctly found, and almost certainly not what a match-analysis detector is
being asked to track. The active match ball, by the reviewer's impression, was
usually annotated. If that impression holds, then "28% missing-ball rate" and
"28% active-ball failure" are very different claims, and only the first is
supported.

So this asks a SECOND question about the SAME findings:

    A  ACTIVE_MATCH_BALL      the ball the current phase of play is centred on
    X  NON_ACTIVE_EXTRA_BALL  a real football, not the ball in play
    U  UNSURE                 the image does not settle it -- do not guess

THE UNIT IS THE OBJECT, NOT THE IMAGE. Round 0's endpoint was the image; this
one is per drawn box, because an image can hold an active ball AND two spares,
and collapsing those to one answer would destroy exactly the distinction the
revisit exists to make. The image-level rate is derived afterwards by folding
objects up, never by asking the human an image-level question.

A SIDE CHANNEL FOR EXISTING GT. While looking hard at these images the reviewer
also sees existing ball annotations that are plainly wrong -- a player annotated
as a football, or a real ball with a box that does not fit it. Those are worth
capturing at the moment they are noticed, because nobody will be looking at
these frames this closely again. They are recorded as FLAGS and nothing else:
no annotation is changed, no queue is advanced, no ontology answer is implied,
and the 128-object denominator is untouched. A flag is an observation with a
timestamp, and the cleanup queue that consumes it does not exist yet.

WHY THE QUEUE IS BUILT FROM THE EFFECTIVE FOLD. Six images were answered more
than once during Round 0. Reading missing_balls straight from the log yields 135
objects; the effective state holds 128. The seven extras are superseded drafts
that no longer exist as findings, and putting them in front of a reviewer would
invent work and inflate the denominator of the secondary analysis.

    python tools/kb_ball_ontology_revisit_server.py

No geometry is created, edited or deleted here. Every event is append-only and
carries the original Round-0 bbox unchanged, so the ontology answer can always
be traced back to the box it describes.
"""

import argparse
import hashlib
import json
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import kb_ball_qa_sample                                          # noqa: E402
import kb_ball_qa_server                                          # noqa: E402
import kb_decisions                                               # noqa: E402
import kb_images                                                  # noqa: E402

PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
DECISIONS = PKG / 'decisions.json'
ROUND0_RESULT = PKG / 'BALL_QA_ROUND0_RESULT.json'
LOCK = threading.Lock()

ONTOLOGY_MODE = 'ball_ontology_revisit'
ACTIVE = 'ACTIVE_MATCH_BALL'
NON_ACTIVE = 'NON_ACTIVE_EXTRA_BALL'
UNSURE = 'UNSURE'
ROLES = (ACTIVE, NON_ACTIVE, UNSURE)

# Flags on EXISTING ball GT. A separate mode, so no fold that reads ontology
# answers can ever see one, and vice versa.
FLAG_MODE = 'ball_gt_flag'
FLAG_RETRACT_MODE = 'ball_gt_flag_retraction'
FALSE_BALL = 'SUSPECT_FALSE_BALL_GT'
BAD_BOX = 'SUSPECT_BAD_BALL_BOX'
# Not a defect at all: a correct annotation of a real ball that simply is not
# the one in play. It belongs in the same side channel because it is the same
# kind of statement -- an observation about existing GT -- and because the
# ontology question the 128 objects answer will be meaningless later if the
# existing annotations it is compared against were never sorted the same way.
EXISTING_NON_ACTIVE = 'EXISTING_NON_ACTIVE_BALL_GT'
FLAG_TYPES = (FALSE_BALL, BAD_BOX, EXISTING_NON_ACTIVE)

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>ball ontology revisit</title>
<style>
 body{margin:0;background:#0e0e0e;color:#e6e6e6;font:13px system-ui,sans-serif}
 #top{display:flex;gap:14px;align-items:center;padding:7px 12px;background:#000;
      position:sticky;top:0;z-index:9;flex-wrap:wrap;border-bottom:1px solid #222}
 .pill{padding:1px 7px;border-radius:9px;background:#1c1c1c;border:1px solid #333}
 #bar{height:4px;background:#222;width:160px;border-radius:2px;overflow:hidden}
 #bar>i{display:block;height:100%;background:#3ddc57;width:0}
 #wrap{position:relative;margin:10px 300px 10px 10px;width:fit-content;
       overflow:auto;max-height:82vh;max-width:calc(100vw - 330px)}
 #stage{position:relative;transform-origin:0 0}
 img{display:block}
 /* the object under review: unmistakable, and the only magenta thing on screen */
 .target{position:absolute;border:3px solid #ff2fd0;box-sizing:border-box;
         pointer-events:none;box-shadow:0 0 0 2px #000,0 0 14px #ff2fd0}
 /* the other Round-0 findings in this image, deliberately quiet */
 .sibling{position:absolute;border:1px dashed #7a7a7a;box-sizing:border-box;
          pointer-events:none}
 .ballgt{position:absolute;border:2px solid #4aa3ff;box-sizing:border-box;
         pointer-events:none;box-shadow:0 0 0 1px #000}
 .ballgt.sel{border-color:#fff;border-width:3px;box-shadow:0 0 0 2px #000,
             0 0 12px #fff}
 .ballgt.flagged{border-style:dashed;border-color:#ff6a3d}
 .ballgt.flaggedbox{border-style:dashed;border-color:#c08a2a}
 .ballgt.flaggednon{border-style:dashed;border-color:#9a7fd0}
 /* #ov had no position, so it sat in normal flow BELOW the image and made the
    stage taller than the picture. #hit's inset:0 then resolved against that
    taller box, so its top-left was not the image's top-left and every click
    mapped to the wrong image coordinate -- boxes were visible and unclickable.
    Both are now pinned to the image origin. */
 #ov{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none}
 #hit{position:absolute;left:0;top:0;width:100%;height:100%;cursor:pointer;
      z-index:1}
 .ctx{position:absolute;border:1px solid #333;box-sizing:border-box;
      pointer-events:none}
 .tag{position:absolute;font:11px/1.3 monospace;background:#000d;padding:1px 4px;
      transform:translateY(-100%);white-space:nowrap;pointer-events:none}
 #panel{position:fixed;right:10px;top:52px;width:276px;background:#141414;
        border:1px solid #2a2a2a;border-radius:6px;padding:9px;z-index:8;
        max-height:84vh;overflow:auto}
 button{background:#1e1e1e;color:#ddd;border:1px solid #3a3a3a;border-radius:4px;
        padding:3px 8px;cursor:pointer;font:12px system-ui}
 button:hover{background:#2a2a2a}
 button.big{width:100%;margin-top:6px;padding:7px;text-align:left}
 button.act{border-color:#2f5a2f}
 button.non{border-color:#6a5a20}
 .k{background:#222;border:1px solid #3a3a3a;border-radius:3px;padding:0 6px;
    font:12px monospace;margin-right:6px}
 .ok{background:#132a13;border:1px solid #2f5a2f;border-radius:4px;padding:6px;
     margin:6px 0;font-size:11px}
 .non{background:#2f2a10;border:1px solid #6a5a20;border-radius:4px;padding:6px;
      margin:6px 0;font-size:11px}
 .warn{background:#3a1414;border:1px solid #7a3a3a;border-radius:4px;padding:6px;
       margin:6px 0;font-size:11px}
 .note{color:#8a8a8a;font-size:11px;margin:-2px 0 8px}
 .qbox{font-size:12px;font-weight:600;margin:2px 0 6px}
 #tiles{display:flex;gap:3px;flex-wrap:wrap;margin:5px 0}
 #tiles button{padding:2px 7px;font:11px monospace}
 #tiles button.on{background:#20402a;border-color:#3ddc57;color:#cfc}
 #legend{font-size:11px;color:#888;margin:6px 0}
 #legend i{display:inline-block;width:10px;height:10px;margin-right:4px;
           vertical-align:middle}
 #gthelp{font-size:11px;color:#aaa;line-height:1.7}
 #gthelp .k{min-width:16px;display:inline-block;text-align:center}
 #selhdr{background:#10243a;border:1px solid #2f5a8a;border-radius:4px;
         padding:6px;margin:6px 0;font-size:11px}
 #selhdr b{color:#8fc4ff}
</style></head><body>
<div id="top">
 <b>BALL ONTOLOGY REVISIT</b>
 <span id="pos" class="pill"></span>
 <span id="rem" class="pill"></span>
 <span id="img" class="pill" style="font:11px monospace"></span>
 <div id="bar"><i></i></div>
 <span id="zoomp" class="pill"></span>
 <span id="flagc" class="pill" style="font-size:11px"></span>
 <span class="pill" style="font:10px monospace;color:#777"
       title="the build this server process is actually running">build
  __BUILD__</span>
 <span style="color:#8a8a8a;font-size:11px">magenta: A active &middot; X
  non-active &middot; U unsure &nbsp;|&nbsp; blue GT: [ ] select &middot; E
  non-active &middot; F false &middot; V bad box &middot; C retract
  &nbsp;|&nbsp; M/&rarr; next &middot; B/&larr; prev &middot; J next
  unclassified</span>
</div>
<div id="imgerr" style="display:none;margin:10px 300px 10px 10px;background:#3a1414;
     border:1px solid #7a3a3a;border-radius:6px;padding:12px;max-width:640px"></div>
<div id="wrap"><div id="stage"><img id="im"><div id="ov"></div><div id="hit"></div></div></div>
<div id="panel">
 <div class="qbox">Is the <span style="color:#ff2fd0">highlighted</span> ball the
  ACTIVE MATCH BALL?</div>
 <div id="legend">
  <div><i style="background:#ff2fd0"></i>the object being classified</div>
  <div><i style="background:#7a7a7a"></i>other Round-0 findings in this image</div>
  <div><i style="background:#4aa3ff"></i>existing ball annotations
   &mdash; click to select</div>
 </div>
 <div id="state"></div>
 <button class="big act" id="aA"><span class="k">A</span>ACTIVE MATCH BALL</button>
 <div class="note">the ball the current phase of play is centred on</div>
 <button class="big non" id="aX"><span class="k">X</span>NON-ACTIVE EXTRA BALL</button>
 <div class="note">real football, not in play: ball boy, spare behind the goal,
  touchline or warm-up ball</div>
 <button class="big" id="aU"><span class="k">U</span>UNSURE</button>
 <div class="note">the image does not settle it &mdash; do not guess</div>
 <div style="border-top:1px solid #2a2a2a;margin:9px 0 6px"></div>
 <div style="font-size:11px;font-weight:600;color:#4aa3ff;margin-bottom:3px">
  EXISTING BLUE BALL GT</div>
 <div id="gthelp">
  <div><span class="k">[</span><span class="k">]</span>previous / next GT</div>
  <div><span class="k">E</span>NON-ACTIVE EXISTING BALL</div>
  <div><span class="k">F</span>FALSE BALL</div>
  <div><span class="k">V</span>BAD BOX</div>
  <div><span class="k">C</span>RETRACT</div>
  <div><span class="k">Esc</span>deselect</div>
 </div>
 <div class="note" style="margin-top:5px">observation only &mdash; no annotation
  is changed and the magenta object is not answered</div>
 <div id="gtsel"></div>
 <div style="border-top:1px solid #2a2a2a;margin:9px 0 6px"></div>
 <div style="font-size:11px;color:#8a8a8a">tile sweep</div>
 <div id="tiles"></div>
 <button class="big" id="bNx"><span class="k">M</span>NEXT IMAGE</button>
 <button class="big" id="bB"><span class="k">B</span>PREVIOUS</button>
 <button class="big" id="bJ"><span class="k">J</span>NEXT UNCLASSIFIED</button>
 <div class="note">NEXT IMAGE steps one at a time through all 128, answered or
  not. NEXT UNCLASSIFIED skips ahead to work still outstanding.</div>
 <div id="meta" style="font-size:11px;color:#777;margin-top:8px"></div>
 <div id="hist" style="font-size:11px;color:#888;margin-top:8px"></div>
</div>
<script>
let S=null,i=0,zoom=1,tile=-1,seen={},selGT=null,cycle=0;
const TILE_ZOOM=4,TCOLS=4,TROWS=3,OVERLAP=0.12;
var boot=async function(){
 S=await (await fetch('/api/state')).json();
 i=S.items.findIndex(x=>!x.role); if(i<0)i=0;
 flagcounts();
 render();
};
function cur(){return S.items[i];}
function remaining(){return S.items.filter(x=>!x.role).length;}
function render(){
 const t=cur(); if(!t)return;
 zoom=1; tile=-1; selGT=null; cycle=0;
 if(!seen[t.object_id])seen[t.object_id]={};
 document.getElementById('pos').textContent=`object ${i+1}/${S.items.length}`;
 document.getElementById('rem').textContent=`${remaining()} remaining`;
 document.getElementById('img').textContent=t.IMAGE;
 const im=document.getElementById('im');
 const err=document.getElementById('imgerr');
 err.style.display='none'; im.style.display='block';
 im.onload=()=>{applyZoom();draw();};
 im.onerror=async()=>{im.style.display='none';
  let why='the server did not return an image';
  try{const r=await fetch('/img/'+t.IMAGE);const j=await r.json();
      why=j.error||why;}catch(e){}
  err.style.display='block';
  err.innerHTML='<b>IMAGE COULD NOT BE LOADED</b><br><span style="font:11px '
   +'monospace">'+t.IMAGE+'</span><br>'+why+'<br><br>Do not classify this object.';};
 im.src='/img/'+t.IMAGE;
 tilebar(); stats();
}
function applyZoom(){
 const st=document.getElementById('stage');
 st.style.transform='scale('+zoom+')';
 const im=document.getElementById('im');
 st.style.width=im.naturalWidth+'px'; st.style.height=im.naturalHeight+'px';
 document.getElementById('zoomp').textContent='zoom '+zoom.toFixed(2)+'x';
}
function tilebar(){
 const t=cur(),bar=document.getElementById('tiles');
 bar.innerHTML='';
 const mk=(label,idx)=>{
  const b=document.createElement('button');
  b.textContent=label;
  if(idx===tile)b.className='on';
  b.onclick=()=>gotoTile(idx);
  bar.appendChild(b);
 };
 mk('FIT',-1);
 for(let k=0;k<TCOLS*TROWS;k++)mk(String(k+1),k);
 mk('BALL',-2);
}
function gotoTile(idx){
 const t=cur(),im=document.getElementById('im');
 const W=im.naturalWidth||t.img_w,H=im.naturalHeight||t.img_h;
 const wrap=document.getElementById('wrap');
 if(idx===-2){                       // centre on the object under review
  tile=-2; zoom=TILE_ZOOM; applyZoom();
  const b=t.bbox_xywh;
  wrap.scrollLeft=Math.max(0,(b[0]+b[2]/2)*zoom-wrap.clientWidth/2);
  wrap.scrollTop =Math.max(0,(b[1]+b[3]/2)*zoom-wrap.clientHeight/2);
  tilebar(); return;
 }
 if(idx<0){tile=-1;zoom=1;applyZoom();tilebar();return;}
 tile=idx; seen[t.object_id][idx]=true;
 zoom=TILE_ZOOM; applyZoom();
 const cw=W/TCOLS,ch=H/TROWS;
 const cx=(idx%TCOLS)*cw,cy=Math.floor(idx/TCOLS)*ch;
 wrap.scrollLeft=Math.max(0,(cx-cw*OVERLAP)*zoom);
 wrap.scrollTop =Math.max(0,(cy-ch*OVERLAP)*zoom);
 tilebar();
}
function draw(){
 const t=cur(),ov=document.getElementById('ov');
 ov.innerHTML='';
 const add=(cls,b,label,col)=>{
  const e=document.createElement('div'); e.className=cls;
  e.style.cssText=`left:${b[0]}px;top:${b[1]}px;width:${b[2]}px;height:${b[3]}px;`;
  ov.appendChild(e);
  if(label){const g=document.createElement('div');g.className='tag';
   g.style.cssText=`left:${b[0]}px;top:${b[1]}px;color:${col||'#999'}`;
   g.textContent=label; ov.appendChild(g);}
 };
 if(S.show_context)t.context.forEach(b=>add('ctx',b.bbox,null,null));
 t.ball_gt.forEach((b,n)=>{
  let cls='ballgt';
  if(b.flag===S.FALSE_BALL)cls+=' flagged';
  else if(b.flag===S.BAD_BOX)cls+=' flaggedbox';
  else if(b.flag===S.EXISTING_NON_ACTIVE)cls+=' flaggednon';
  if(selGT===b.BOX_ID)cls+=' sel';
  let lab='ball GT '+(n+1)+'/'+t.ball_gt.length;
  if(b.flag===S.FALSE_BALL)lab='FALSE BALL';
  else if(b.flag===S.BAD_BOX)lab='BAD BOX';
  else if(b.flag===S.EXISTING_NON_ACTIVE)lab='NON-ACTIVE';
  if(selGT===b.BOX_ID)lab='SELECTED · '+lab;
  add(cls,b.bbox,lab,b.flag?(b.flag===S.EXISTING_NON_ACTIVE?'#9a7fd0':'#ff6a3d')
                          :'#4aa3ff');
 });
 t.siblings.forEach((b,k)=>add('sibling',b.bbox_xywh,
   'other finding'+(b.role?' → '+b.short:''),'#7a7a7a'));
 add('target',t.bbox_xywh,'CLASSIFY THIS','#ff2fd0');
 panel();
}
// Geometry hit-testing, not DOM order: the overlay divs are painted in a fixed
// sequence, so a small ball inside a larger flagged box would be unreachable if
// selection followed paint order. Smallest area first, then cycling.
function gtAt(x,y){
 return cur().ball_gt.filter(b=>x>=b.bbox[0]&&y>=b.bbox[1]&&
   x<=b.bbox[0]+b.bbox[2]&&y<=b.bbox[1]+b.bbox[3])
  .sort((p,q)=>(p.bbox[2]*p.bbox[3])-(q.bbox[2]*q.bbox[3]));
}
// Stepping is the reliable path: many of these balls are 4-8 px, which is 2-3
// screen pixels at fit zoom, so clicking one is a test of mouse precision
// rather than of judgement. [ and ] walk the list in a fixed order and bring
// the box into view, so no ball is unreachable however small it is.
function stepGT(dir){
 const g=cur().ball_gt;
 if(!g.length){selGT=null;draw();return;}
 const at=g.findIndex(b=>b.BOX_ID===selGT);
 const n=at<0?(dir>0?0:g.length-1):((at+dir)%g.length+g.length)%g.length;
 selGT=g[n].BOX_ID;
 revealGT(g[n]);
 draw();
}
function revealGT(b){
 const wrap=document.getElementById('wrap');
 if(zoom<TILE_ZOOM){zoom=TILE_ZOOM;applyZoom();tile=-3;tilebar();}
 wrap.scrollLeft=Math.max(0,(b.bbox[0]+b.bbox[2]/2)*zoom-wrap.clientWidth/2);
 wrap.scrollTop =Math.max(0,(b.bbox[1]+b.bbox[3]/2)*zoom-wrap.clientHeight/2);
}
function pick(ev){
 const im=document.getElementById('im');
 const r=im.getBoundingClientRect();
 const x=(ev.clientX-r.left)/zoom, y=(ev.clientY-r.top)/zoom;
 // a generous pad so a 5px ball is clickable at fit zoom without the pad
 // ever letting a click reach a box it is not actually near
 const pad=Math.max(0,(6/zoom)-1);
 let hits=gtAt(x,y);
 if(!hits.length)hits=cur().ball_gt.filter(b=>
   x>=b.bbox[0]-pad&&y>=b.bbox[1]-pad&&
   x<=b.bbox[0]+b.bbox[2]+pad&&y<=b.bbox[1]+b.bbox[3]+pad)
  .sort((p,q)=>(p.bbox[2]*p.bbox[3])-(q.bbox[2]*q.bbox[3]));
 if(!hits.length){selGT=null;cycle=0;draw();return;}
 // clicking the same spot again cycles through the boxes under the cursor
 const idx=hits.findIndex(b=>b.BOX_ID===selGT);
 selGT=hits[idx>=0?(idx+1)%hits.length:0].BOX_ID;
 cycle=hits.length;
 draw();
}
function panel(){
 const t=cur();
 const st=document.getElementById('state');
 if(t.role===S.ACTIVE)
  st.innerHTML='<div class="ok">classified <b>ACTIVE MATCH BALL</b>. '
   +'Answering again appends a new event; nothing is rewritten.</div>';
 else if(t.role===S.NON_ACTIVE)
  st.innerHTML='<div class="non">classified <b>NON-ACTIVE EXTRA BALL</b>.</div>';
 else if(t.role===S.UNSURE)
  st.innerHTML='<div class="warn">classified <b>UNSURE</b> &mdash; reported '
   +'separately, never folded into either category.</div>';
 else st.innerHTML='<div class="warn">not classified yet. Press <b>Z</b> to jump '
   +'to this object at 4x.</div>';
 document.getElementById('meta').innerHTML=
  `box <b>${t.bbox_xywh.map(v=>Math.round(v)).join(', ')}</b>
   (${Math.round(t.bbox_xywh[2])}x${Math.round(t.bbox_xywh[3])} px)
   <div>split <b>${t.split}</b> · run <b>${t.run||'?'}</b> · view
    <b>${t.view_proxy||'?'}</b> · ball GT in image <b>${t.ball_gt.length}</b>
    · findings in image <b>${t.siblings.length+1}</b></div>
   <div style="color:#666">metadata is secondary; classify from the image.</div>`;
 document.getElementById('hist').innerHTML=!t.history.length?'':
  '<b>history</b>'+t.history.map(h=>`<div>${(h.recorded_utc||'').slice(11,19)}
   ${h.role}</div>`).join('');
 gtpanel();
}
function gtpanel(){
 const t=cur(),el=document.getElementById('gtsel');
 const b=t.ball_gt.find(x=>x.BOX_ID===selGT);
 const nav=t.ball_gt.length
  ?`<div class="note"><span class="k">[</span><span class="k">]</span>step
     through the ${t.ball_gt.length} ball GT in this image &mdash; easier than
     clicking a 4 px box</div>`:'';
 if(!b){
  el.innerHTML=!t.ball_gt.length
   ?'<div class="note">this image has no ball annotation to observe</div>'
   :'<div class="note">click a blue ball GT box, or press <b>]</b>, to select it'
    +(t.ball_gt.length>1?' ('+t.ball_gt.length+' in this image)':'')+'</div>'+nav;
  return;
 }
 const NAME={};
 NAME[S.FALSE_BALL]='SUSPECT FALSE BALL';
 NAME[S.BAD_BOX]='SUSPECT BAD BOX';
 NAME[S.EXISTING_NON_ACTIVE]='EXISTING NON-ACTIVE BALL';
 const at=t.ball_gt.findIndex(x=>x.BOX_ID===b.BOX_ID);
 el.innerHTML=`
  <div id="selhdr">
   SELECTED GT: <b>${b.BOX_ID}</b><br>
   GT <b>${at+1}/${t.ball_gt.length}</b> in this image · class <b>football</b><br>
   box <span style="font:11px monospace">${b.bbox.map(v=>Math.round(v)).join(', ')}</span>
   (${Math.round(b.bbox[2])}x${Math.round(b.bbox[3])} px)<br>
   observation: <b>${b.flag?NAME[b.flag]:'none yet'}</b>
  </div>
  <button class="big" id="fE"><span class="k">E</span>EXISTING NON-ACTIVE BALL</button>
  <div class="note">correct annotation of a real ball that is not the one in
   play &mdash; not a defect</div>
  <button class="big" id="fF"><span class="k">F</span>SUSPECT FALSE BALL</button>
  <button class="big" id="fV"><span class="k">V</span>SUSPECT BAD BOX</button>
  <button class="big" id="fC"><span class="k">C</span>RETRACT OBSERVATION</button>
  ${nav}
  <div class="note">an observation is recorded for later. It changes no
   annotation, answers no ontology object, and does not advance the queue.</div>`;
 document.getElementById('fE').onclick=()=>flag(S.EXISTING_NON_ACTIVE);
 document.getElementById('fF').onclick=()=>flag(S.FALSE_BALL);
 document.getElementById('fV').onclick=()=>flag(S.BAD_BOX);
 document.getElementById('fC').onclick=()=>flag(null);
 document.getElementById('fC').disabled=!b.flag;
}
function flagcounts(){
 const c=S.flag_counts||{false:0,bad_box:0,non_active:0};
 document.getElementById('flagc').textContent=
  `GT obs — non-active ${c.non_active||0} · false ${c['false']} · `
  +`bad box ${c.bad_box}`;
}
async function flag(kind){
 const t=cur(),b=t.ball_gt.find(x=>x.BOX_ID===selGT);
 if(!b){alert('select an existing ball GT box first');return;}
 const body={BOX_ID:b.BOX_ID};
 if(kind===null)body.retract=true; else body.flag_type=kind;
 const r=await fetch('/api/flag',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const j=await r.json();
 if(!r.ok){alert(j.error||'refused');return;}
 // reflect on every item that shows this annotation, not just this one
 S.items.forEach(it=>it.ball_gt.forEach(g=>{
   if(g.BOX_ID===b.BOX_ID)g.flag=j.flag_type;}));
 if(j.counts)S.flag_counts=j.counts;
 flagcounts();
 draw();                       // deliberately no next(): a flag is not an answer
}
function stats(){
 const d=S.items.filter(x=>x.role).length;
 document.getElementById('bar').firstElementChild.style.width=
   (100*d/S.items.length)+'%';
 document.getElementById('rem').textContent=`${remaining()} remaining`;
}
async function classify(role){
 const t=cur();
 const r=await fetch('/api/ontology',{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({object_id:t.object_id,HUMAN_BALL_ROLE:role})});
 const j=await r.json();
 if(!r.ok){alert(j.error||'refused');return;}
 t.role=role;
 t.history.push({role:role,recorded_utc:j.recorded_utc});
 // keep the sibling badges in the same image in step
 S.items.forEach(o=>{if(o.IMAGE===t.IMAGE)
   o.siblings.forEach(s=>{if(s.object_id===t.object_id){
     s.role=role; s.short=role===S.ACTIVE?'ACTIVE':
       (role===S.NON_ACTIVE?'non-active':'unsure');}});});
 draw(); stats(); next();
}
function next(){
 for(let k=i+1;k<S.items.length;k++){if(!S.items[k].role){i=k;render();return;}}
 for(let k=0;k<S.items.length;k++){if(!S.items[k].role){i=k;render();return;}}
 render();
}
document.getElementById('hit').onclick=pick;
document.getElementById('aA').onclick=()=>classify(S.ACTIVE);
document.getElementById('aX').onclick=()=>classify(S.NON_ACTIVE);
document.getElementById('aU').onclick=()=>classify(S.UNSURE);
// step one forward, whether or not it is answered -- the mirror of PREVIOUS.
// next() skips to outstanding work, which with 114 of 128 done jumps over
// nearly everything and makes the queue impossible to simply page through.
function step(d){const n=i+d; if(n>=0&&n<S.items.length){i=n;render();}}
document.getElementById('bNx').onclick=()=>step(1);
document.getElementById('bJ').onclick=next;
document.getElementById('bB').onclick=()=>step(-1);
document.onkeydown=e=>{
 const k=e.key.toLowerCase();
 // 'n' is deliberately NOT bound. It reads as both NEXT and NON-ACTIVE, and
 // either meaning would be a silent misclassification of a real finding.
 if(k==='a')classify(S.ACTIVE);
 else if(k==='x')classify(S.NON_ACTIVE);
 else if(k==='u')classify(S.UNSURE);
 else if(k==='j'){e.preventDefault();next();}
 else if(k==='m'){e.preventDefault();step(1);}
 else if(e.key==='ArrowRight'){e.preventDefault();step(1);}
 else if(e.key==='ArrowLeft'){e.preventDefault();step(-1);}
 else if(k==='k'||k==='b'){step(-1);}
 else if(k==='t'){gotoTile(tile+1>=TCOLS*TROWS?-1:tile+1);}
 else if(k==='z'){gotoTile(-2);}
 // the observation keys act on the SELECTED existing GT and never on the
 // magenta Round-0 object, which has no BOX_ID and is not in ball_gt at all
 else if(k==='e'){if(selGT)flag(S.EXISTING_NON_ACTIVE);
                  else alert('select an existing ball GT box first ( ] )');}
 else if(k==='f'){if(selGT)flag(S.FALSE_BALL);
                  else alert('select an existing ball GT box first ( ] )');}
 else if(k==='v'){if(selGT)flag(S.BAD_BOX);
                  else alert('select an existing ball GT box first ( ] )');}
 else if(k==='c'){if(selGT)flag(null);}
 else if(e.key===']'){e.preventDefault();stepGT(1);}
 else if(e.key==='['){e.preventDefault();stepGT(-1);}
 else if(e.key==='Escape'){selGT=null;draw();}
 else if(e.key==='+'||e.key==='='){zoom=Math.min(zoom*1.25,12);applyZoom();}
 else if(e.key==='-'){zoom=Math.max(zoom/1.25,0.25);applyZoom();}
 else if(k==='0'){zoom=1;tile=-1;applyZoom();tilebar();}
};
boot();
</script></body></html>"""


def object_id(image, bbox):
    """Stable identity for a drawn box: image + geometry, hashed.

    Not the index in the list. Indices renumber if an earlier image is ever
    re-answered, which would silently reattach an ontology answer to a
    different ball. Geometry is what the human actually pointed at, so it is
    what the identity is built from.
    """
    key = f'{image}|' + ','.join(f'{float(v):.2f}' for v in bbox)
    return 'BALLOBJ:' + hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]


def round0_objects(decisions: Path = DECISIONS):
    """The Round-0 findings as they EFFECTIVELY stand. One entry per drawn box.

    Built from kb_ball_qa_server.answers(), which folds the log to the latest
    answer per image. Six images were answered more than once; the superseded
    drafts contribute 7 boxes that are no longer findings and must not appear.
    """
    out = []
    for image, a in sorted(kb_ball_qa_server.answers(decisions).items()):
        if a['answer'] != 'MISSING_BALL':
            continue
        for k, m in enumerate(a['missing']):
            bbox = m['bbox_xywh']
            out.append({'object_id': object_id(image, bbox), 'IMAGE': image,
                        'bbox_xywh': bbox, 'index_in_image': k,
                        'round0_recorded_utc': a['recorded_utc']})
    return out


def ontology(decisions: Path = DECISIONS):
    """Effective ontology answer per object. Latest human event wins."""
    per = {}
    for d in kb_decisions.read_log(decisions):
        if d.get('mode') == ONTOLOGY_MODE:
            per.setdefault(d['missing_object_id'], []).append(d)
    out = {}
    for oid, evs in per.items():
        evs = sorted(evs, key=lambda d: (d.get('recorded_utc') or '', d['_line']))
        out[oid] = {
            'role': evs[-1].get('HUMAN_BALL_ROLE'),
            'recorded_utc': evs[-1].get('recorded_utc'),
            'history': [{'role': e.get('HUMAN_BALL_ROLE'),
                         'recorded_utc': e.get('recorded_utc')} for e in evs],
        }
    return out


def gt_flags(decisions: Path = DECISIONS):
    """Effective flag state per existing ball annotation. Latest event wins.

    Keyed by `<split>:<annotation id>`, the same BOX_ID convention the exporter
    uses. Raw COCO ids are NOT unique across splits -- train and valid share
    4,508 of them -- so a flag keyed on the bare id would land on an unrelated
    annotation in another split.

    Retraction is an event, not a deletion: a flag that was raised and withdrawn
    stays in the log with both halves visible, because "somebody looked at this
    and changed their mind" is itself worth knowing when the cleanup queue is
    finally built.

    Returns {BOX_ID: {flag_type or None, IMAGE, bbox_xywh, retracted, history}}.
    """
    per = {}
    for d in kb_decisions.read_log(decisions):
        if d.get('mode') in (FLAG_MODE, FLAG_RETRACT_MODE):
            per.setdefault(d['BOX_ID'], []).append(d)
    out = {}
    for box, evs in per.items():
        evs = sorted(evs, key=lambda d: (d.get('recorded_utc') or '', d['_line']))
        win = evs[-1]
        retracted = win['mode'] == FLAG_RETRACT_MODE
        last_flag = next((e for e in reversed(evs) if e['mode'] == FLAG_MODE),
                         None)
        out[box] = {
            'flag_type': None if retracted else win.get('flag_type'),
            'IMAGE': (last_flag or win).get('IMAGE'),
            'bbox_xywh': (last_flag or win).get('bbox_xywh'),
            'annotation_id': (last_flag or win).get('annotation_id'),
            'retracted': retracted,
            'recorded_utc': win.get('recorded_utc'),
            'history': [{'mode': e['mode'], 'flag_type': e.get('flag_type'),
                         'recorded_utc': e.get('recorded_utc')} for e in evs],
        }
    return out


def flag_counts(decisions: Path = DECISIONS):
    """Effective counts for the header. Retracted flags count as neither."""
    eff = gt_flags(decisions)
    return {
        'false': sum(1 for v in eff.values() if v['flag_type'] == FALSE_BALL),
        'bad_box': sum(1 for v in eff.values() if v['flag_type'] == BAD_BOX),
        'non_active': sum(1 for v in eff.values()
                          if v['flag_type'] == EXISTING_NON_ACTIVE),
        'retracted': sum(1 for v in eff.values() if v['retracted']),
    }


def build_state(show_context=False, decisions: Path = DECISIONS):
    objs = round0_objects(decisions)
    onto = ontology(decisions)
    flags = gt_flags(decisions)
    man = json.loads(kb_ball_qa_sample.MANIFEST.read_text(encoding='utf-8'))
    meta = {r['IMAGE']: r for r in man['sample']}

    by_image = {}
    for o in objs:
        by_image.setdefault(o['IMAGE'], []).append(o)

    doc_cache = {}
    items = []
    short = {ACTIVE: 'ACTIVE', NON_ACTIVE: 'non-active', UNSURE: 'unsure'}
    for o in objs:
        m = meta[o['IMAGE']]
        split = m['split']
        if split not in doc_cache:
            doc = json.loads((kb_ball_qa_sample.EXPORT /
                              f'{split}_annotations.coco.json')
                             .read_text(encoding='utf-8'))
            bc = kb_ball_qa_sample._ball_category(doc['categories'])
            per_img = {}
            for a in doc['annotations']:
                per_img.setdefault(a['image_id'], []).append(a)
            doc_cache[split] = (bc, per_img)
        bc, per_img = doc_cache[split]
        rows = per_img.get(m['coco_image_id'], [])
        sibs = []
        for s in by_image[o['IMAGE']]:
            if s['object_id'] == o['object_id']:
                continue
            r = onto.get(s['object_id'], {}).get('role')
            sibs.append({'object_id': s['object_id'], 'bbox_xywh': s['bbox_xywh'],
                         'role': r, 'short': short.get(r, '')})
        rec = onto.get(o['object_id'], {})
        items.append({
            **o,
            'split': split, 'run': m['run'], 'view_proxy': m['view_proxy'],
            'gt_state': m['gt_state'], 'img_w': m['img_w'], 'img_h': m['img_h'],
            'ball_gt': [{'bbox': x['bbox'],
                         'annotation_id': x['id'],
                         'BOX_ID': f'{split}:{x["id"]}',
                         'cls': 'football',
                         'flag': flags.get(f'{split}:{x["id"]}', {}).get(
                             'flag_type')}
                        for x in rows if x['category_id'] == bc],
            'context': [{'bbox': x['bbox']} for x in rows
                        if x['category_id'] != bc],
            'siblings': sibs,
            'role': rec.get('role'),
            'history': rec.get('history', []),
        })
    return {'items': items, 'show_context': show_context,
            'ACTIVE': ACTIVE, 'NON_ACTIVE': NON_ACTIVE, 'UNSURE': UNSURE,
            'FALSE_BALL': FALSE_BALL, 'BAD_BOX': BAD_BOX,
            'EXISTING_NON_ACTIVE': EXISTING_NON_ACTIVE,
            'flag_counts': flag_counts(decisions),
            'source_log': kb_decisions.log_version(decisions)}


def append(rec):
    with LOCK:
        with open(DECISIONS, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec) + '\n')
            fh.flush()


def build_id_info():
    """Identity of the page this process is actually serving.

    The first version hashed __file__ at request time, which reported the file
    ON DISK -- so a server started before an edit advertised the NEW hash while
    serving the OLD page, the exact confusion the stamp exists to prevent. It
    now hashes PAGE, the in-memory string this process will actually send, and
    reports the on-disk hash separately so a mismatch is visible rather than
    hidden behind one number.
    """
    page_sha = hashlib.sha256(PAGE.encode('utf-8')).hexdigest()
    try:
        disk_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    except OSError:
        disk_sha = None
    return {'build': page_sha[:12],
            'page_sha256': page_sha,
            'source_file_sha256_on_disk': disk_sha,
            'has_gt_navigation': "e.key===']'" in PAGE,
            'has_gt_observations': "k==='e'" in PAGE,
            'has_next_image': 'bNx' in PAGE,
            'flag_types': list(FLAG_TYPES)}


def build_id():
    return build_id_info()['build']


def ball_gt_index(decisions: Path = DECISIONS):
    """{BOX_ID: {IMAGE, bbox, annotation_id, split}} for every EXISTING ball GT
    in the images that hold a Round-0 finding.

    The server checks a flag target against this rather than trusting the
    client. A Round-0 missing box is a human drawing that exists only in the
    decision log, has no annotation id, and is not in here -- so it cannot be
    flagged as suspect GT even if a crafted request names it. The two
    populations must never merge: one is what the dataset claims, the other is
    what the dataset missed.
    """
    man = json.loads(kb_ball_qa_sample.MANIFEST.read_text(encoding='utf-8'))
    meta = {r['IMAGE']: r for r in man['sample']}
    images = {o['IMAGE'] for o in round0_objects(decisions)}
    out, cache = {}, {}
    for image in sorted(images):
        m = meta[image]
        split = m['split']
        if split not in cache:
            doc = json.loads((kb_ball_qa_sample.EXPORT /
                              f'{split}_annotations.coco.json')
                             .read_text(encoding='utf-8'))
            bc = kb_ball_qa_sample._ball_category(doc['categories'])
            per = {}
            for a in doc['annotations']:
                if a['category_id'] == bc:
                    per.setdefault(a['image_id'], []).append(a)
            cache[split] = per
        for a in cache[split].get(m['coco_image_id'], []):
            out[f'{split}:{a["id"]}'] = {
                'IMAGE': image, 'split': split, 'annotation_id': a['id'],
                'bbox_xywh': a['bbox']}
    return out


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    SHOW_CONTEXT = False
    OBJECTS = {}
    BALL_GT = {}

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        v = memoryview(body)
        for i in range(0, len(v), 1 << 16):
            self.wfile.write(v[i:i + (1 << 16)])
        self.wfile.flush()

    def do_GET(self):
        p = unquote(urlparse(self.path).path)
        if p == '/':
            # A long-running server holds PAGE in memory from whenever it
            # started. Editing this file changes nothing until it restarts, and
            # the symptom is a UI that silently lacks features that demonstrably
            # exist in the source -- which is exactly what happened. The stamp
            # makes the served build identifiable without guessing.
            body = PAGE.replace('__BUILD__', build_id()).encode('utf-8')
            return self._send(200, body, 'text/html; charset=utf-8')
        if p == '/api/build':
            return self._send(200, json.dumps(build_id_info()).encode())
        if p == '/api/state':
            return self._send(200, json.dumps(
                build_state(self.SHOW_CONTEXT)).encode())
        if p.startswith('/img/'):
            want = p[len('/img/'):]
            if want not in {o['IMAGE'] for o in self.OBJECTS.values()}:
                return self._send(403, json.dumps(
                    {'error': 'image holds no Round-0 missing object',
                     'IMAGE': want}).encode())
            try:
                body, ctype = kb_images.read(want)
            except kb_images.ImageError as e:
                print(f'IMAGE 404  {want}  --  {e}', flush=True)
                return self._send(404, json.dumps(
                    {'error': str(e), 'IMAGE': want}).encode())
            return self._send(200, body, ctype)
        return self._send(404, b'not found', 'text/plain')

    def do_POST(self):
        route = urlparse(self.path).path
        n = int(self.headers.get('Content-Length', 0))
        d = json.loads(self.rfile.read(n) or b'{}')

        if route == '/api/flag':
            return self._flag(d)
        if route != '/api/ontology':
            return self._send(404, b'', 'text/plain')

        oid = str(d.get('object_id', ''))
        if oid not in self.OBJECTS:
            return self._send(400, json.dumps(
                {'error': 'not one of the Round-0 missing objects; only the '
                          'boxes a human already drew can be classified',
                 'object_id': oid}).encode())
        role = d.get('HUMAN_BALL_ROLE')
        if role not in ROLES:
            return self._send(400, json.dumps(
                {'error': f'HUMAN_BALL_ROLE must be one of {ROLES}'}).encode())

        o = self.OBJECTS[oid]
        now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        append({
            'mode': ONTOLOGY_MODE,
            # BOX_ID mirrors the object id so the shared log reader keeps
            # working. This is not a decision about any existing annotation.
            'BOX_ID': oid,
            'missing_object_id': oid,
            'IMAGE': o['IMAGE'],
            'round0_bbox_xywh': o['bbox_xywh'],
            'HUMAN_BALL_ROLE': role,
            'HUMAN_FINAL_CLASS': None,
            'geometry_unchanged': True,
            'no_new_geometry_created': True,
            'no_model_proposal_used': True,
            'classifies': 'a Round-0 missing-ball finding, not an annotation',
            'recorded_utc': now, 'author': 'human reviewer',
        })
        return self._send(200, json.dumps(
            {'ok': True, 'HUMAN_BALL_ROLE': role, 'recorded_utc': now}).encode())

    def _flag(self, d):
        """Flag or retract a flag on an EXISTING ball annotation.

        Nothing is changed by this route. It appends an observation, and the
        ontology queue, the Round-0 counts and the export are all untouched.
        """
        box = str(d.get('BOX_ID', ''))
        if box not in self.BALL_GT:
            # This is the guard that matters: object_ids from the 128 drawn
            # boxes start with BALLOBJ: and are never in BALL_GT, so a
            # mis-wired client cannot flag a missing-ball finding as bad GT.
            return self._send(400, json.dumps(
                {'error': 'not an existing ball annotation in a reviewed image; '
                          'only annotations already in the export can be '
                          'flagged, never a Round-0 missing-ball drawing',
                 'BOX_ID': box}).encode())
        gt = self.BALL_GT[box]
        now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

        if d.get('retract'):
            eff = gt_flags(DECISIONS).get(box)
            if not eff or not eff['flag_type']:
                return self._send(400, json.dumps(
                    {'error': 'no effective flag on this annotation to '
                              'retract'}).encode())
            append({
                'mode': FLAG_RETRACT_MODE, 'BOX_ID': box,
                # kb_decisions.resolve() reads this field on every row it sees,
                # so an event that omits it crashes the shared reader for the
                # whole log -- not just for this mode.
                'HUMAN_FINAL_CLASS': None,
                'annotation_id': gt['annotation_id'], 'IMAGE': gt['IMAGE'],
                'target_flag_event': eff['recorded_utc'],
                'retracts_flag_type': eff['flag_type'],
                'bbox_xywh': gt['bbox_xywh'],
                'reason': 'human correction',
                'annotation_unchanged': True,
                'recorded_utc': now, 'author': 'human reviewer'})
            return self._send(200, json.dumps(
                {'ok': True, 'flag_type': None, 'recorded_utc': now}).encode())

        ft = d.get('flag_type')
        if ft not in FLAG_TYPES:
            return self._send(400, json.dumps(
                {'error': f'flag_type must be one of {FLAG_TYPES}'}).encode())
        append({
            'mode': FLAG_MODE, 'BOX_ID': box,
            'flag_type': ft,
            'annotation_id': gt['annotation_id'],
            'IMAGE': gt['IMAGE'], 'split': gt['split'],
            'bbox_xywh': gt['bbox_xywh'],
            'current_class': 'football',
            'reason': 'human visual review',
            'HUMAN_FINAL_CLASS': None,
            'annotation_unchanged': True,
            'no_annotation_modified': True,
            'no_model_proposal_used': True,
            'observation_only': ('recorded for a later cleanup queue; this pass '
                                 'changes nothing'),
            'recorded_utc': now, 'author': 'human reviewer'})
        return self._send(200, json.dumps(
            {'ok': True, 'flag_type': ft, 'recorded_utc': now,
             'counts': flag_counts(DECISIONS)}).encode())


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8745)
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--show-context', action='store_true',
                    help='outline non-ball annotations faintly as context')
    args = ap.parse_args()

    if not ROUND0_RESULT.is_file():
        print(f'REFUSING: no frozen Round-0 result at {ROUND0_RESULT}.\n'
              f'The ontology revisit describes Round-0 findings and must not '
              f'run before Round 0 is frozen.')
        sys.exit(1)
    res = json.loads(ROUND0_RESULT.read_text(encoding='utf-8'))

    objs = round0_objects()
    H.SHOW_CONTEXT = args.show_context
    H.OBJECTS = {o['object_id']: o for o in objs}
    H.BALL_GT = ball_gt_index()
    if len(H.OBJECTS) != len(objs):
        print(f'REFUSING: {len(objs) - len(H.OBJECTS)} objects share an id; '
              f'two identical boxes were drawn in one image.')
        sys.exit(1)

    want = res['secondary']['total_missing_objects']
    if len(objs) != want:
        print(f'REFUSING: the frozen result records {want} missing objects but '
              f'the log yields {len(objs)}. The queue must describe exactly the '
              f'findings the frozen result counted.')
        sys.exit(1)

    ok, problems = kb_images.preflight(sorted({o['IMAGE'] for o in objs}))
    if not ok:
        print(f'REFUSING TO START: {len(problems)} image(s) cannot be resolved.')
        for im, why in problems[:10]:
            print(f'  {im}\n    {why}')
        sys.exit(1)

    onto = ontology()
    done = sum(1 for o in objs if o['object_id'] in onto)
    imgs = len({o['IMAGE'] for o in objs})
    fc = flag_counts()
    print(f'BALL ONTOLOGY REVISIT -- {len(objs)} Round-0 missing objects '
          f'across {imgs} images')
    print(f'frozen Round 0 stands unchanged: '
          f'{res["primary"]["positive_images"]}/{res["primary"]["n"]} positive '
          f'images, {want} objects')
    print(f'{done} classified, {len(objs) - done} outstanding')
    print(f'{len(H.BALL_GT)} existing ball annotations in these images are '
          f'flaggable')
    print(f'ball GT observations so far: non-active {fc["non_active"]}, '
          f'false {fc["false"]}, bad box {fc["bad_box"]}'
          + (f', retracted {fc["retracted"]}' if fc['retracted'] else ''))
    print('preflight: every image resolves')
    print('\nNO GEOMETRY IS CREATED OR EDITED. No annotation is changed by a '
          'flag. No detector is loaded.')
    print("Keys: A active  X non-active  U unsure  J next  B/K previous"
          "  ('n' is unbound on purpose)")
    print('      [ ] step through the blue ball GT, then E existing non-active '
          'ball  F suspect false ball')
    print('      V suspect bad box  C retract   (clicking a box also selects it)')
    url = f'http://127.0.0.1:{args.port}/'
    print(f'\nreview at {url}   Ctrl-C to stop')
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(('127.0.0.1', args.port), H).serve_forever()


if __name__ == '__main__':
    main()
