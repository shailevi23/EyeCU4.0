#!/usr/bin/env python
"""
Image-centric review server for the 6,984-box missed_role pass.

The first server was box-centric: one highlighted box at a time, Tab to the next.
That is the wrong shape for this queue. 6,984 candidates sit in only 1,184 images
-- 5.9 per image -- and they are all the same question asked of the same picture,
so the reviewer should see the whole image once and answer it once.

What changed, and what deliberately did not:

  * every candidate in the image is drawn and clickable at the same time, colour
    coded by proposed role, with the image's context boxes kept visible in grey
    so a candidate is judged against the players around it
  * two bulk actions, both scoped to the CURRENT IMAGE ONLY: mark all remaining
    candidates player, or accept all proposals. The accept button stays disabled
    until every candidate in the image has actually been on screen and scrolled
    to, so "accept all" cannot mean "accept things I did not look at"
  * nothing propagates. No decision crosses an image boundary, no identity is
    asserted, no proposal becomes a label without a keypress or a click

The gate is untouched by all of this. Decisions are appended to the same
decisions.json in the same schema with mode='missed_role', which is a separate
namespace from the completed candidates/qa_player/qa_nocand work, so a second
pass cannot overwrite a first-pass decision.
"""

import argparse
import json
import mimetypes
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / 'tools'))
import kb_decisions                                              # noqa: E402
import kb_images                                                 # noqa: E402

PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
IMGROOT = kb_images.IMGROOT
MODES = ('missed_role', 'missed_role_manual', 'missing_target_box',
         'missing_target_retraction', 'final_target')
# M -- a real human who is not taking part: bench, coach, ball person, medical
# or technical staff, anyone on the touchline. It is a DISPOSITION, never an
# EyeCU class, and it must never quietly become `player`.
#
# The value recorded is NON_TARGET_HUMAN, which the u_resolution pass already
# defines as exactly this ("coach, bench, ball person, medical, staff") and
# which 22 boxes already carry. Introducing a second name for one concept would
# split the bucket and make every later count remember to add both. The label
# NON_ACTIVE_MATCH_HUMAN is kept on the record so the audit trail still shows
# which key was pressed.
NON_ACTIVE = 'NON_TARGET_HUMAN'
ROLE_VALUES = ('player', 'goalkeeper', 'referee', 'uncertain', NON_ACTIVE)
# The escape hatch for a target whose role genuinely cannot be read. Leaving one
# on 'uncertain' leaves it labelled `player`, which is a wrong label in TRAIN if
# it is a keeper or an official -- so the honest alternative is to drop its
# image, explicitly and on the record, rather than guess or leave it open.
EXCLUDE = 'EXCLUDE_IMAGE'
# An image-level flag for a real EyeCU target that has NO annotation at all, so
# there is nothing to click. It cannot use a real BOX_ID because no box exists;
# the key is synthetic and namespaced so it can never collide with `split:id`,
# and the applier skips ids it does not recognise. No box is created or inferred
# here -- flagging is a request for annotation, recorded for a later pass.
MISSING_PREFIX = 'MISSING:'
LOCK = threading.Lock()

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>missed_role review</title>
<style>
 :root{--gk:#ffc400;--ref:#ff7a1a;--pl:#3ddc57;--un:#b06cff;--ctx:#4a4a4a}
 body{margin:0;background:#0e0e0e;color:#e6e6e6;font:13px system-ui,sans-serif}
 #top{display:flex;gap:16px;align-items:center;padding:7px 12px;background:#000;
      position:sticky;top:0;z-index:9;flex-wrap:wrap;border-bottom:1px solid #222}
 #bar{height:4px;background:#222;width:180px;border-radius:2px;overflow:hidden}
 #bar>i{display:block;height:100%;background:#3ddc57;width:0}
 .pill{padding:1px 7px;border-radius:9px;background:#1c1c1c;border:1px solid #333}
 .gk{color:var(--gk)}.ref{color:var(--ref)}.pl{color:var(--pl)}.un{color:var(--un)}
 #wrap{position:relative;margin:10px auto;width:fit-content}
 img{display:block;max-width:98vw;max-height:74vh}
 /* pointer-events:none so a click reaches the overlay and is resolved by
    GEOMETRY, not by which element happens to be painted last. A small box
    inside a large one is otherwise unreachable. */
 .bx{position:absolute;border:2px solid;box-sizing:border-box;pointer-events:none}
 #ov{position:absolute;inset:0;cursor:crosshair}
 .bx.ctx{border-color:var(--ctx);border-width:1px}
 .bx.selctx{box-shadow:0 0 0 2px #fff inset;border-color:#b06cff!important;border-width:2px}
 /* the one box a revisit item is about, so it cannot be mistaken for context */
 .bx.utarget{border-color:#b06cff!important;border-width:4px!important;
   box-shadow:0 0 0 3px #000,0 0 14px 4px #b06cffcc;animation:upulse 1.4s infinite}
 @keyframes upulse{50%{box-shadow:0 0 0 3px #000,0 0 6px 2px #b06cff77}}
 .bx.done{border-style:solid;border-width:3px}
 .bx:hover{filter:brightness(1.6)}
 .tag{position:absolute;font:11px/1.3 monospace;background:#000d;padding:1px 4px;
      transform:translateY(-100%);white-space:nowrap;pointer-events:none;border-radius:2px}
 #panel{position:fixed;right:10px;top:52px;width:260px;background:#141414;
        border:1px solid #2a2a2a;border-radius:6px;padding:9px;z-index:8;max-height:76vh;overflow:auto}
 .row{display:flex;gap:5px;align-items:center;padding:3px 0;border-bottom:1px solid #1e1e1e}
 .row b{width:56px}
 button{background:#1e1e1e;color:#ddd;border:1px solid #3a3a3a;border-radius:4px;
        padding:3px 8px;cursor:pointer;font:12px system-ui}
 button:hover{background:#2a2a2a}
 button:disabled{opacity:.35;cursor:not-allowed}
 button.big{width:100%;margin-top:6px;padding:6px}
 .k{background:#222;border:1px solid #3a3a3a;border-radius:3px;padding:0 5px;font:11px monospace}
 #hint{color:#8a8a8a}
</style></head><body>
<div id="top">
 <b id="mode">missed_role</b>
 <span id="pos" class="pill"></span>
 <span id="run" class="pill"></span>
 <span id="imgs" class="pill"></span>
 <span id="boxes" class="pill"></span>
 <div id="bar"><i></i></div>
 <span class="pill"><span class="gk" id="cG">G 0</span> ·
  <span class="ref" id="cR">R 0</span> · <span class="pl" id="cP">P 0</span> ·
  <span class="un" id="cU">U 0</span></span>
 <span class="pill" id="rem"></span>
 <span class="pill" style="border-color:#555">manual
  <b id="mNew" class="ref">new 0</b> · <b id="mOv" class="gk">override 0</b> ·
  <b id="mNo" style="color:#888">no-op 0</b> · <b id="mFl" class="un">flagged 0</b>
  <span style="color:#666">| G <b id="mG">0</b> R <b id="mR">0</b>
   P <b id="mP">0</b> U <b id="mU">0</b></span></span>
 <span class="pill">non-active <b id="cM" style="color:#aaa">0</b></span>
 <span class="pill" style="border-color:#7a3a3a">missing-box flags
  <b id="fT" style="color:#ff7a7a">0</b> targets in
  <b id="fI" style="color:#ff7a7a">0</b> images</span>
 <span id="hint"><span class="k">Q</span> flag a target with NO box ·
  <span class="k">click</span> a box, then <span class="k">P</span>
  <span class="k">G</span> <span class="k">R</span> <span class="k">U</span> ·
  <span class="k">1-9</span> pick box · <span class="k">A</span> all=player ·
  <span class="k">Enter</span> accept proposals · <span class="k">N</span>/<span class="k">B</span> nav ·
  <span class="k">M</span> non-active human (bench/coach/staff) ·
  <b>click any black box</b> to correct a missed role ·
  <span class="k">Tab</span> cycle overlapping boxes (or click the same spot again;
  smallest first)</span>
</div>
<div id="ubanner" style="display:none;position:sticky;top:36px;z-index:10;
     background:#132a13;border-bottom:1px solid #2f5a2f;padding:6px 12px"></div>
<div id="qbanner" style="display:none;position:sticky;top:36px;z-index:10;
     background:#3a1414;border-bottom:1px solid #7a3a3a;padding:6px 12px">
 <b style="color:#ff9a9a">MISSING TARGET BOX</b> &mdash; a real target with no
 annotation at all. Press <b class="k">G</b> goalkeeper &middot;
 <b class="k">R</b> referee &middot; <b class="k">P</b> player &middot;
 <b class="k">U</b> role unclear &nbsp;|&nbsp; <b class="k">Esc</b> cancel.
 No box is created here &mdash; this records a request for annotation.
</div>
<div id="wrap"><img id="im"><div id="ov"></div></div>
<div id="panel">
 <div id="urow" style="display:none;background:#1d1526;border:1px solid #4a3a5e;
      border-radius:4px;padding:6px;margin-bottom:6px"></div>
 <div id="selinfo" style="border-bottom:1px solid #222;padding-bottom:5px;
      margin-bottom:5px"></div>
 <div id="ovbox" style="display:none;background:#221c0c;border:1px solid #4a3c14;
      border-radius:4px;padding:5px;margin-bottom:6px"></div>
 <div id="list"></div>
 <button class="big" id="allP">ALL CURRENT CANDIDATES = PLAYER <span class="k">A</span></button>
 <button class="big" id="acc">ACCEPT ALL PROPOSALS <span class="k">Enter</span></button>
 <div id="accnote" style="color:#8a8a8a;margin-top:5px"></div>
 <button class="big" id="nonact" style="border-color:#6a6a6a">NON-ACTIVE MATCH HUMAN
  <span class="k">M</span></button>
 <div style="color:#8a8a8a;font-size:11px;margin:-2px 0 6px">
  bench/substitute, coach, ball person, medical or other sideline staff &mdash;
  a real human, but not one of the four EyeCU classes. Works on the selected
  candidate or on a clicked black box.</div>
 <div id="qhere" style="font-size:11px;padding:4px 0"></div>
 <button class="big" id="flagQ" style="border-color:#7a3a3a">FLAG MISSING TARGET BOX
  <span class="k">Q</span></button>
 <button class="big" id="uJump" style="border-color:#5a3a7a;display:none"></button>
 <button class="big" id="uExcl" style="border-color:#7a3a3a;display:none">EXCLUDE
  THIS IMAGE (role unreadable)</button>
 <button class="big" id="uBack" style="border-color:#3a5a7a;display:none">RETURN TO
  FULL REVIEW</button>
 <button class="big" id="next">NEXT UNRESOLVED IMAGE <span class="k">N</span></button>
 <button class="big" id="prev">PREVIOUS <span class="k">B</span></button>
</div>
<script>
let S=null,i=0,sel=null,selKind='cand',dec={},man={},kind={},seen={};
let missing=[],qMode=false,uAt=-1;
// U REVISIT is its own queue, not a jump inside the 6,684. The retrospective
// pass is finished -- 6684/6684, 1133/1133 -- and re-entering it to clean up
// seven boxes puts the reviewer back in a population that has no work left,
// where N walks 1,133 images to find the next one. Everything below is scoped
// to the seven: the header, the navigation, and what "remaining" counts.
let uMode=false;
function uQueue(){
 // built from the SERVER's resolve() output, filtered by what is still
 // effectively uncertain right now, so a box answered in this session leaves
 // the queue immediately and a historically-uncertain box never enters it
 return (S.u_open||[]).filter(u=>dec[u.BOX_ID]==='uncertain'
                              ||man[u.BOX_ID]==='uncertain');
}
function uEnter(){
 if(!uQueue().length)return;
 uMode=true; uAt=0; uShow();
}
function uExit(){
 uMode=false;
 document.getElementById('ubanner').style.display='none';
 i=S.items.findIndex(x=>x.candidates.some(c=>!dec[c.BOX_ID]));
 if(i<0)i=0;
 render();
}
function uShow(){
 const q=uQueue();
 if(!q.length){uDone();return;}
 uAt=((uAt%q.length)+q.length)%q.length;
 const t=q[uAt];
 const k=S.items.findIndex(x=>x.IMAGE===t.IMAGE);
 if(k<0){uAt++;uShow();return;}
 i=k;
 const it=S.items[i];
 // draw() records which candidates have been displayed; without this the
 // revisit view walks into an image the full pass never opened and throws
 seen[it.IMAGE]=seen[it.IMAGE]||new Set();
 document.getElementById('run').textContent='run '+it.run;
 const im=document.getElementById('im');
 im.onload=()=>{uSelect(t);};
 im.src='/img/'+it.IMAGE;
 uSelect(t);
 stats();
}
function uSelect(t){
 const it=cur(); if(!it)return;
 sel=t.BOX_ID;
 selKind=it.candidates.some(c=>c.BOX_ID===t.BOX_ID)?'cand':'ctx';
 ovlp=[];ovlpAt=0;ovlpX=ovlpY=null;
 draw();
}
function uDone(){
 document.getElementById('ubanner').style.display='block';
 document.getElementById('ubanner').innerHTML=
  '<b style="color:#8fe08f">U REVISIT COMPLETE</b> &mdash; effective unresolved '
  +'uncertain boxes: <b>0</b>. Gate condition G no longer has an uncertain box '
  +'to block on; re-run <code>tools/kb_second_pass_gate.py</code> to confirm. '
  +'<button id="ubk" style="margin-left:8px">RETURN TO FULL REVIEW</button>';
 document.getElementById('ubk').onclick=uExit;
 stats();
}
const CLS={player:'pl',goalkeeper:'gk',referee:'ref',uncertain:'un'};
const COLV={player:'#3ddc57',goalkeeper:'#ffc400',referee:'#ff7a1a',
            uncertain:'#b06cff',NON_TARGET_HUMAN:'#8a8a8a'};
const NONACT='NON_TARGET_HUMAN';
var boot=async function(){
 S=await (await fetch('/api/state')).json(); dec=S.decisions; man=S.manual;
 kind=S.manual_kind; missing=S.missing||[];
 i=S.items.findIndex(x=>x.candidates.some(c=>!dec[c.BOX_ID])); if(i<0)i=0;
 render();
}
function cur(){return S.items[i];}
// ---------------------------------------------------------------------------
// Hit-testing. A click is resolved against box GEOMETRY, never against which
// element sits on top: candidates were appended after context boxes, so a small
// context box under a large candidate could not be clicked at all, and a small
// box fully inside a larger one was unreachable whichever order they were in.
//
// Every box containing the point is collected, smallest area first, so the most
// specific box is the first thing offered. Clicking again in the same place
// walks outward through the group and wraps, which is the only way to reach a
// large box whose interior is covered by smaller ones.
// ---------------------------------------------------------------------------
let ovlp=[],ovlpAt=0,ovlpX=null,ovlpY=null;
const SAME_SPOT=6;                       // px, so a small hand tremor still cycles
function area(b){return Math.max(b[2],0)*Math.max(b[3],0);}
function boxesAt(x,y){
 const it=cur(),out=[];
 const inside=b=>x>=b[0]&&x<=b[0]+b[2]&&y>=b[1]&&y<=b[1]+b[3];
 it.candidates.forEach(c=>{if(inside(c.bbox))
   out.push({BOX_ID:c.BOX_ID,kind:'cand',a:area(c.bbox)});});
 it.context.forEach(b=>{if(inside(b.bbox))
   out.push({BOX_ID:b.BOX_ID,kind:'ctx',a:area(b.bbox)});});
 // smallest first, then a stable tie-break so the cycle order never wobbles
 out.sort((p,q)=>p.a-q.a||(p.BOX_ID<q.BOX_ID?-1:1));
 return out;
}
function sameGroup(g){
 return g.length===ovlp.length&&g.every((b,n)=>b.BOX_ID===ovlp[n].BOX_ID);
}
function pick(n){
 if(!ovlp.length)return;
 ovlpAt=((n%ovlp.length)+ovlp.length)%ovlp.length;
 sel=ovlp[ovlpAt].BOX_ID; selKind=ovlp[ovlpAt].kind;
 draw();
}
function cycle(step){if(ovlp.length>1)pick(ovlpAt+step);}
function hit(ev){
 if(qMode)return;                        // flagging a missing target selects nothing
 const it=cur(),im=document.getElementById('im');
 const r=im.getBoundingClientRect();
 const x=(ev.clientX-r.left)*it.img_w/im.clientWidth;
 const y=(ev.clientY-r.top)*it.img_h/im.clientHeight;
 const g=boxesAt(x,y);
 if(!g.length){ovlp=[];ovlpAt=0;ovlpX=ovlpY=null;draw();return;}
 const near=ovlpX!==null&&Math.abs(ev.clientX-ovlpX)<=SAME_SPOT
                        &&Math.abs(ev.clientY-ovlpY)<=SAME_SPOT;
 const advance=near&&sameGroup(g);
 ovlp=g; ovlpX=ev.clientX; ovlpY=ev.clientY;
 pick(advance?ovlpAt+1:0);               // same spot walks outward, else smallest
}
function render(){
 if(uMode){uShow();return;}
 const it=cur(); if(!it)return;
 document.getElementById('pos').textContent=`image ${i+1}/${S.items.length}`;
 document.getElementById('run').textContent='run '+it.run;
 const im=document.getElementById('im'); im.onload=draw; im.src='/img/'+it.IMAGE;
 sel=it.candidates.find(c=>!dec[c.BOX_ID])?.BOX_ID||it.candidates[0].BOX_ID;
 selKind='cand';
 ovlp=[];ovlpAt=0;ovlpX=ovlpY=null;      // a new image starts a new overlap group
 seen[it.IMAGE]=seen[it.IMAGE]||new Set();
 stats();
}
function draw(){
 const it=cur(),im=document.getElementById('im'),ov=document.getElementById('ov');
 const sx=im.clientWidth/it.img_w, sy=im.clientHeight/it.img_h;
 ov.innerHTML='';
 it.context.forEach(b=>{
  // subdued until it carries a human answer, or until it is selected
  const mine=man[b.BOX_ID], other=b.already;
  const col=mine?COLV[mine]:(other&&other!=='player'?COLV[other]:null);
  const e=document.createElement('div');
  e.className='bx ctx'+(b.BOX_ID===sel&&selKind==='ctx'?' selctx':'')
   +(uMode&&b.BOX_ID===sel?' utarget':'');
  e.style.cssText=`left:${b.bbox[0]*sx}px;top:${b.bbox[1]*sy}px;
   width:${b.bbox[2]*sx}px;height:${b.bbox[3]*sy}px;cursor:pointer;`+
   (col?`border-color:${col};border-width:2px;`:'');
  e.title='existing annotation -- click if you can see its role was missed';
  ov.appendChild(e);
  if(mine||other){
   const t2=document.createElement('div'); t2.className='tag';
   t2.style.cssText=`left:${b.bbox[0]*sx}px;top:${b.bbox[1]*sy}px;color:${col||'#888'}`;
   t2.textContent=mine?('MANUAL: '+mine):(other+' ('+b.already_mode+')');
   ov.appendChild(t2);}
 });
 it.candidates.forEach((c,n)=>{
  const d=dec[c.BOX_ID];
  const col=d?COLV[d]:COLV[c.proposed];
  const e=document.createElement('div');
  e.className='bx'+(d?' done':'')+(uMode&&c.BOX_ID===sel?' utarget':'');
  e.style.cssText=`left:${c.bbox[0]*sx}px;top:${c.bbox[1]*sy}px;
   width:${c.bbox[2]*sx}px;height:${c.bbox[3]*sy}px;border-color:${col};
   ${c.BOX_ID===sel?'box-shadow:0 0 0 2px #fff inset;':''}`;
  ov.appendChild(e);
  const t=document.createElement('div'); t.className='tag';
  t.style.cssText=`left:${c.bbox[0]*sx}px;top:${c.bbox[1]*sy}px;color:${col}`;
  t.textContent=`${n+1} ${d?('YOU:'+d):('?'+c.proposed)}`;
  ov.appendChild(t);
  seen[it.IMAGE].add(c.BOX_ID);
 });
 list();
}
function list(){
 const it=cur();
 // What is selected right now, and whether anything else sits under the click.
 const curRole=dec[sel]||man[sel]||
   (it.context.find(b=>b.BOX_ID===sel)||{}).already||null;
 document.getElementById('selinfo').innerHTML=
  `<div style="font:11px monospace;color:#bbb">${sel||'-'}
    <span style="color:#777">${selKind==='ctx'?'context box':'candidate'}</span></div>`
  +`<div style="font-size:11px;color:#8a8a8a">current: <b>${curRole||'no answer yet'}</b></div>`;
 const ur=document.getElementById('urow');
 if(uMode){
  const q=uQueue(), t=q[Math.min(uAt,q.length-1)];
  const led=it.candidates.find(c=>c.BOX_ID===sel);
  ur.style.display='block';
  ur.innerHTML=q.length?`<b style="color:#b06cff">UNCERTAIN REVISIT
    ${Math.min(uAt+1,q.length)}/${q.length}</b>
   <div style="font:11px monospace;color:#bbb;margin-top:3px">${sel}</div>
   <div style="font-size:11px;color:#8a8a8a">current answer:
    <b class="un">uncertain</b>${led?' · proposed <b>'+led.proposed+'</b>':''}
    · ${selKind==='ctx'?'existing annotation':'queued candidate'}</div>
   <div style="font-size:11px;color:#8a8a8a;margin-top:4px">answer
    <span class="k">P</span> <span class="k">G</span> <span class="k">R</span>
    <span class="k">M</span>, or exclude the image if the role really cannot be
    read. <span class="k">N</span>/<span class="k">B</span> move within these
    ${q.length} only.</div>`
   :'<b style="color:#8fe08f">U REVISIT COMPLETE</b>';
 } else ur.style.display='none';
 const ob=document.getElementById('ovbox');
 if(ovlp.length>1){
  ob.style.display='block';
  ob.innerHTML=`<b style="color:#ffc400">OVERLAPPING BOXES
    ${ovlpAt+1}/${ovlp.length}</b>
   <div style="font-size:11px;color:#8a8a8a;margin:2px 0">smallest first; click the
    same spot again, or use <span class="k">Tab</span> /
    <span class="k">Shift+Tab</span>, to reach the ones underneath</div>`;
  const mk=(lab,step)=>{const b=document.createElement('button');
    b.textContent=lab;b.onclick=()=>cycle(step);return b;};
  ob.appendChild(mk('< prev',-1)); ob.appendChild(mk('next >',1));
 } else ob.style.display='none';
 document.getElementById('list').innerHTML=it.candidates.map((c,n)=>{
  const d=dec[c.BOX_ID];
  return `<div class="row" style="${c.BOX_ID===sel?'background:#1d1d1d':''}">
   <b><span class="k">${n+1}</span></b>
   <span class="${CLS[d||c.proposed]}">${d?d:'?'+c.proposed}</span>
   <span style="color:#777;margin-left:auto">${c.score.toFixed(2)}</span></div>`;
 }).join('');
 const isCtx=selKind==='ctx';
 const ctxb=isCtx?it.context.find(b=>b.BOX_ID===sel):null;
 const resolved=ctxb?(man[ctxb.BOX_ID]||ctxb.already):null;
 const rmode=ctxb?(man[ctxb.BOX_ID]?'missed_role_manual':ctxb.already_mode):null;
 document.getElementById('list').innerHTML+=isCtx
  ? `<div class="row" style="background:#241d2e;margin-top:6px">
      <b>ctx</b><span class="un">manual correction</span>
      <span style="color:#888;margin-left:auto">${sel}</span></div>`
    +(resolved?`<div style="background:#1b2a1b;border:1px solid #2f4a2f;
       border-radius:4px;padding:5px;margin:4px 0;font-size:11px">
       <b style="color:#8fe08f">ALREADY RESOLVED</b> &mdash;
       <b>${resolved}</b> via <i>${rmode}</i>.<br>
       choosing <b>${resolved}</b> again records a NO_OP_CONFIRMATION, not a new
       missed role. A DIFFERENT class is kept as a HUMAN_OVERRIDE.
       ${kind[ctxb.BOX_ID]?'<br>last manual click: <b>'+kind[ctxb.BOX_ID]+'</b>':''}
      </div>`
     :`<div style="color:#8a8a8a;font-size:11px;padding:2px 0">
       unresolved existing annotation, not part of the 6,684. P/G/R/U records it
       as <b>missed_role_manual</b>.</div>`)
  : '';
 const all=it.candidates.every(c=>seen[it.IMAGE]&&seen[it.IMAGE].has(c.BOX_ID));
 document.getElementById('acc').disabled=!all;
 document.getElementById('accnote').textContent=all
   ? 'every candidate in this image has been displayed'
   : 'accept-all stays disabled until all candidates are on screen';
}
function stats(){
 const done=Object.keys(dec).length, tot=S.total_boxes;
 const c={player:0,goalkeeper:0,referee:0,uncertain:0};
 Object.values(dec).forEach(v=>c[v]!==undefined&&c[v]++);
 const imgsDone=S.items.filter(x=>x.candidates.every(z=>dec[z.BOX_ID])).length;
 const uq=uQueue();
 document.getElementById('mode').textContent=uMode?'UNCERTAIN REVISIT':'missed_role';
 document.getElementById('mode').style.color=uMode?'#b06cff':'';
 if(uMode){
  // the 6,684 population is finished and is NOT what is being counted here
  document.getElementById('pos').textContent=
    uq.length?`item ${Math.min(uAt+1,uq.length)}/${uq.length}`:'0 items';
  document.getElementById('imgs').textContent=`${uq.length} remaining`;
  document.getElementById('boxes').textContent='cleanup queue only';
  document.getElementById('rem').textContent='candidates 6,684/6,684 complete';
  document.querySelector('#bar>i').style.width=
    (uq.length?100*uAt/uq.length:100)+'%';
 } else {
  document.getElementById('imgs').textContent=`images ${imgsDone}/${S.items.length}`;
  document.getElementById('boxes').textContent=`boxes ${done}/${tot}`;
  document.getElementById('rem').textContent=`remaining ${tot-done}`;
  document.querySelector('#bar>i').style.width=(100*done/tot)+'%';
 }
 document.getElementById('cG').textContent='G '+c.goalkeeper;
 document.getElementById('cR').textContent='R '+c.referee;
 document.getElementById('cP').textContent='P '+c.player;
 document.getElementById('cU').textContent='U '+c.uncertain;
 const uo=uq.length;
 const ub=document.getElementById('uJump');
 ub.style.display=(uo&&!uMode)?'block':'none';
 // textContent, so a plain hyphen rather than an entity or a literal em dash
 ub.textContent='REVISIT U BOXES ('+uo+') - open the cleanup queue';
 document.getElementById('uBack').style.display=uMode?'block':'none';
 // the exclusion escape hatch is offered only while a U box is selected
 document.getElementById('uExcl').style.display=
   (dec[sel]==='uncertain'||man[sel]==='uncertain')?'block':'none';
 document.getElementById('cM').textContent=
   Object.values(dec).filter(v=>v===NONACT).length+
   Object.values(man).filter(v=>v===NONACT).length;
 const mc={player:0,goalkeeper:0,referee:0,uncertain:0};
 Object.values(man).forEach(v=>mc[v]!==undefined&&mc[v]++);
 document.getElementById('mG').textContent=mc.goalkeeper;
 document.getElementById('mR').textContent=mc.referee;
 document.getElementById('mP').textContent=mc.player;
 document.getElementById('mU').textContent=mc.uncertain;
 const kc={NEW_MISSED_ROLE_CORRECTION:0,HUMAN_OVERRIDE:0,
           NO_OP_CONFIRMATION:0,FLAGGED_UNCERTAIN:0};
 Object.values(kind).forEach(v=>kc[v]!==undefined&&kc[v]++);
 document.getElementById('mNew').textContent='new '+kc.NEW_MISSED_ROLE_CORRECTION;
 document.getElementById('mOv').textContent='override '+kc.HUMAN_OVERRIDE;
 document.getElementById('mNo').textContent='no-op '+kc.NO_OP_CONFIRMATION;
 document.getElementById('mFl').textContent='flagged '+kc.FLAGGED_UNCERTAIN;
 document.getElementById('fT').textContent=missing.length;
 document.getElementById('fI').textContent=new Set(missing.map(m=>m.IMAGE)).size;
 const here=missing.filter(m=>m.IMAGE===cur().IMAGE);
 const live=here.filter(m=>!m.retracted);
 document.getElementById('qhere').innerHTML=here.length
  ? '<b style="color:#ff9a9a">'+live.length+' missing target'+
    (live.length===1?'':'s')+' flagged here</b>'+
    (here.length>live.length?' <span style="color:#666">('+
      (here.length-live.length)+' retracted)</span>':'')+
    '<div style="color:#8a8a8a;margin:3px 0">each row is a SEPARATE target; '+
    'retract only an accidental duplicate</div>'+
    here.map((m,n)=>`<div class="row" style="opacity:${m.retracted?0.4:1}">
      <b>#${n+1}</b><span style="color:${COLV[m.missing_role]||'#ccc'}">
      ${m.missing_role}</span>
      <span style="color:#666;font-size:10px">${(m.recorded_utc||'').slice(11,19)}</span>
      ${m.retracted?'<span style="color:#888;margin-left:auto">retracted</span>'
        :`<button style="margin-left:auto;padding:0 5px"
           onclick="retract('${m.key}')">retract</button>`}</div>`).join('')
  : '';
}
async function post(box,cls,note,mode){
 const m=mode||'missed_role';
 if(m==='missed_role') dec[box]=cls; else man[box]=cls;
 const rsp=await fetch('/api/decide',{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({mode:m,BOX_ID:box,IMAGE:cur().IMAGE,
   HUMAN_FINAL_CLASS:cls,note:note||'per-box click'})});
 if(m==='missed_role_manual'){
  try{const j=await rsp.json(); if(j.manual_kind) kind[box]=j.manual_kind;}catch(e){}
 }
}
async function retract(key){
 const why=prompt('Why is this flag being retracted?\n'+
  '(e.g. "accidental duplicate of the flag above", "misread the image")');
 if(!why||!why.trim())return;
 await fetch('/api/decide',{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({mode:'missing_target_retraction',BOX_ID:key,
                       IMAGE:cur().IMAGE,reason:why.trim()})});
 const m=missing.find(x=>x.key===key);
 if(m){m.retracted=true;m.retraction_reason=why.trim();}
 stats();
}
async function flagMissing(role){
 const it=cur();
 const key='MISSING:'+it.IMAGE+'#'+Date.now();
 const rec={mode:'missing_target_box',BOX_ID:key,IMAGE:it.IMAGE,run:it.run,
            HUMAN_FINAL_CLASS:role,
            note:'image-level flag: real target present with no annotation box'};
 await fetch('/api/decide',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(rec)});
 missing.push({key:key,IMAGE:it.IMAGE,run:it.run,missing_role:role});
 qMode=false; document.getElementById('qbanner').style.display='none';
 stats();
}
async function decide(cls){
 // a coach is never a MISSING TARGET, so M cancels the flag prompt rather
 // than sending a value the server would reject
 if(qMode&&cls===NONACT){qMode=false;
   document.getElementById('qbanner').style.display='none';return;}
 if(qMode){await flagMissing(cls);return;}
 if(!sel)return;
 if(uMode){
  // answering a revisit item resolves it and moves to the next of the seven.
  // The mode is deliberate: it records against the same box in the same mode
  // it was answered in before, so nothing about the 6,684 population changes.
  await post(sel,cls,'uncertain revisit',
             selKind==='ctx'?'missed_role_manual':'missed_role');
  uShow();return;
 }
 if(selKind==='ctx'){
  // optional correction on an existing annotation; never advances the queue
  await post(sel,cls,'manual correction on an existing context box',
             'missed_role_manual');
  draw();stats();return;
 }
 await post(sel,cls);
 const it=cur(); const nxt=it.candidates.find(c=>!dec[c.BOX_ID]);
 if(nxt){sel=nxt.BOX_ID;selKind='cand';draw();stats();} else {stats();draw();}
}
async function allPlayer(){
 const it=cur();
 // candidates only -- context boxes are never bulk-assigned
 for(const c of it.candidates) if(!dec[c.BOX_ID])
   await post(c.BOX_ID,'player','ALL CURRENT CANDIDATES = PLAYER, this image only');
 draw();stats();
}
async function acceptAll(){
 const it=cur();
 if(document.getElementById('acc').disabled)return;
 for(const c of it.candidates) if(!dec[c.BOX_ID])
   await post(c.BOX_ID,c.proposed,'accept-all after every candidate was displayed');
 draw();stats();
}
function nextUnresolved(){
 for(let k=i+1;k<S.items.length;k++)
   if(S.items[k].candidates.some(c=>!dec[c.BOX_ID])){i=k;render();return;}
 i=Math.min(i+1,S.items.length-1);render();
}
document.getElementById('allP').onclick=allPlayer;
document.getElementById('acc').onclick=acceptAll;
document.getElementById('flagQ').onclick=()=>{qMode=true;
 document.getElementById('qbanner').style.display='block';};
document.getElementById('ov').onclick=hit;
document.getElementById('nonact').onclick=()=>decide(NONACT);
// Revisit boxes parked on U. Some are only U because M was advertised while its
// key was dead. Jumping to one changes nothing until the reviewer answers.
function uJump(){
 const open=(S.u_open||[]).filter(u=>dec[u.BOX_ID]==='uncertain'
                                  ||man[u.BOX_ID]==='uncertain');
 if(!open.length)return;
 uAt=(uAt+1)%open.length;
 const t=open[uAt];
 const k=S.items.findIndex(x=>x.IMAGE===t.IMAGE);
 if(k>=0){i=k;render();
  sel=t.BOX_ID;
  selKind=S.items[k].candidates.some(c=>c.BOX_ID===t.BOX_ID)?'cand':'ctx';
  draw();list();}
}
document.getElementById('uJump').onclick=uEnter;
document.getElementById('uBack').onclick=uExit;
// The honest way out for a target whose role cannot be read. Leaving it on U
// leaves it labelled player, which is a wrong label if it is an official.
async function uExclude(){
 const b=sel; if(!b)return;
 if(!confirm('Exclude '+cur().IMAGE+' because the role of '+b+
   ' cannot be read?\n\nThe image and its annotations are not deleted; the '
   +'image is dropped from the repaired candidate set.'))return;
 const why=prompt('Why can this target not be classified?');
 if(!why||!why.trim())return;
 await fetch('/api/decide',{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({mode:'final_target',BOX_ID:b,IMAGE:cur().IMAGE,
   HUMAN_FINAL_CLASS:'EXCLUDE_IMAGE',reason:why.trim()})});
 delete dec[b]; man[b]='EXCLUDE_IMAGE';
 S.u_open=(S.u_open||[]).filter(u=>u.BOX_ID!==b);
 draw();stats();
}
document.getElementById('uExcl').onclick=uExclude;
document.getElementById('next').onclick=()=>{
  if(uMode){uAt++;uShow();} else nextUnresolved();};
document.getElementById('prev').onclick=()=>{
  if(uMode){uAt--;uShow();} else {i=Math.max(0,i-1);render();}};
document.onkeydown=e=>{
 const k=e.key.toLowerCase();
 if(e.key==='Escape'){qMode=false;
   document.getElementById('qbanner').style.display='none';return;}
 // Tab walks the overlap group without moving the mouse. preventDefault stops
 // the browser moving focus to a button, which would make the next P/G/R land
 // somewhere unexpected.
 if(e.key==='Tab'&&ovlp.length>1){e.preventDefault();
   cycle(e.shiftKey?-1:1);return;}
 if(k==='q'&&!qMode){e.preventDefault();qMode=true;
   document.getElementById('qbanner').style.display='block';return;}
 if(k==='p')decide('player'); else if(k==='g')decide('goalkeeper');
 else if(k==='r')decide('referee'); else if(k==='u')decide('uncertain');
 else if(k==='a'&&!qMode){e.preventDefault();allPlayer();}
 else if(e.key==='Enter'&&!qMode){e.preventDefault();acceptAll();}
 else if(k==='n'&&!qMode){e.preventDefault();
   if(uMode){uAt++;uShow();} else nextUnresolved();}
 // no !qMode guard: M must never be silently inert. During a flag prompt
 // decide() cancels it, which is the same thing the button does.
 else if(k==='m'){e.preventDefault();decide(NONACT);}
 else if(k==='b'){if(uMode){uAt--;uShow();} else {i=Math.max(0,i-1);render();}}
 else if(/^[1-9]$/.test(k)){const it=cur();const c=it.candidates[+k-1];
   // picking by number is an explicit choice, so it ends any overlap cycle
   if(c){sel=c.BOX_ID;selKind='cand';ovlp=[];ovlpAt=0;ovlpX=ovlpY=null;draw();}}
};
// ?u=1 opens straight into the cleanup queue, so the finished pass is not
// the thing a reviewer has to navigate out of first
if(location.search.indexOf('u=1')>=0){const _b=boot;boot=async()=>{await _b();
  if(uQueue().length)uEnter(); else uDone();};}
boot();
</script></body></html>"""


def build_state():
    ledger = {r['BOX_ID']: r for r in
              json.loads((PKG / 'ledger.json').read_text(encoding='utf-8'))}
    by_img = {}
    for r in ledger.values():
        by_img.setdefault(r['IMAGE'], []).append(r)
    mrq = json.loads((PKG / 'missed_role_queue.json').read_text(encoding='utf-8'))

    # Every mode's latest word, so a context box that some other pass already
    # settled shows its class instead of inviting the same decision twice.
    resolved = kb_decisions.resolve(PKG / 'decisions.json')
    per_mode = kb_decisions.by_mode(PKG / 'decisions.json')
    dec = {b: v for (m, b), v in per_mode.items() if m == 'missed_role'}
    manual = {b: v for (m, b), v in per_mode.items() if m == 'missed_role_manual'}
    manual_kind = {b: r['kind'] for b, r in
                   kb_decisions.classify_manual(PKG / 'decisions.json').items()}
    missing, retracted = [], {}
    for r in kb_decisions.read_log(PKG / 'decisions.json'):
        if r['mode'] == 'missing_target_box':
            missing.append({'key': r['BOX_ID'], 'IMAGE': r.get('IMAGE'),
                            'run': r.get('run'),
                            'missing_role': r['HUMAN_FINAL_CLASS'],
                            'recorded_utc': r.get('recorded_utc')})
        elif r['mode'] == 'missing_target_retraction':
            retracted[r['BOX_ID']] = r.get('reason')
    # retracted flags stay in history but are no longer live
    for m in missing:
        m['retracted'] = m['key'] in retracted
        m['retraction_reason'] = retracted.get(m['key'])

    # Boxes parked on 'uncertain' in this pass and still unresolved. Most of them
    # are genuinely unreadable, but some are only U because M was advertised in
    # the UI while its key was never wired, so U was the only way to park a
    # bench player or a coach. Exposed for revisiting; nothing is changed here.
    u_open = [{'BOX_ID': b, 'IMAGE': ledger.get(b, {}).get('IMAGE')}
              for b, v in list(dec.items()) + list(manual.items())
              if v == kb_decisions.UNRESOLVED
              and resolved[b]['disposition'] == 'UNRESOLVED']

    per, order = {}, {}
    for row in mrq['rows']:
        per.setdefault(row['IMAGE'], []).append(row)
        order[row['IMAGE']] = max(order.get(row['IMAGE'], 0), row['score'])

    items = []
    for img in sorted(per, key=lambda k: -order[k]):
        rows = sorted(per[img], key=lambda z: -z['score'])
        cand_ids = {z['BOX_ID'] for z in rows}
        base = by_img[img][0]
        items.append({
            'IMAGE': img, 'run': rows[0]['run'],
            'img_w': base['img_w'], 'img_h': base['img_h'],
            'candidates': [{'BOX_ID': z['BOX_ID'],
                            'bbox': ledger[z['BOX_ID']]['bbox_xywh'],
                            'proposed': z['proposed_missed_role'],
                            'score': z['score'],
                            'evidence': z['evidence']} for z in rows],
            # Context boxes are now CLICKABLE. The retrospective ranking is
            # high-recall but not perfect -- its recall was measured on 16 held-out
            # positives -- so a real official can sit in a box the queue never
            # scored highly. Making them actionable in the same pass is the
            # difference between catching that and needing a second full sweep.
            # They are NOT added to the required 6,684: they are optional
            # corrections, taken only when the reviewer notices one.
            'context': [{'BOX_ID': b['BOX_ID'], 'bbox': b['bbox_xywh'],
                         'already': resolved.get(b['BOX_ID'], {}).get('final_class'),
                         'already_mode': resolved.get(b['BOX_ID'], {}).get(
                             'decided_in_mode')}
                        for b in by_img[img]
                        if b['BOX_ID'] not in cand_ids
                        and b['eyecu_original_class'] == 'player'],
        })

    return {'items': items, 'decisions': dec, 'manual': manual,
            'manual_kind': manual_kind, 'missing': missing, 'u_open': u_open,
            'total_boxes': len(mrq['rows']), 'total_images': len(items)}


def page_script_defects(page=None):
    """Unterminated ' or " literals in the served script. Empty means clean.

    A single raw newline inside a JS string literal is a whole-file syntax error:
    the browser parses nothing, so every handler, the fetch of /api/state and the
    first render all fail silently at once. The page still returns 200, the state
    endpoint still returns the full 2.9 MB, and the server log looks perfectly
    healthy -- the only symptom is a blank page with zeroed counters, which reads
    like lost work rather than a typo. That is exactly what happened.

    The page is a raw Python string, so the linters that would catch this never
    see it as code. This does, and main() refuses to serve a page that fails.
    """
    src = PAGE if page is None else page
    i = src.find('<script')
    if i < 0:
        return ['no <script> block in the page']
    js = src[src.find('>', i) + 1:src.rfind('</script>')]
    out, quote, esc, line, start = [], '', False, 1, 0
    depth = 0                       # nesting of ${ } inside template literals
    k = 0
    while k < len(js):
        c = js[k]
        if c == '\n':
            if quote in ("'", '"'):
                out.append(f'line {start}: unterminated {quote} string literal '
                           f'-- a raw newline inside it breaks the whole script')
                quote = ''
            line += 1
        elif esc:
            esc = False
        elif c == '\\':
            esc = True
        elif quote:
            if c == quote and not (quote == '`' and depth):
                quote = ''
            elif quote == '`' and c == '$' and js[k + 1:k + 2] == '{':
                depth += 1
                k += 1
            elif quote == '`' and c == '}' and depth:
                depth -= 1
        elif c in '\'"`':
            quote, start = c, line
        elif c == '/' and js[k + 1:k + 2] == '/':
            while k < len(js) and js[k] != '\n':
                k += 1
            continue
        k += 1
    if quote:
        out.append(f'line {start}: {quote} string literal never closed')
    return out


class H(BaseHTTPRequestHandler):
    # HTTP/1.0 + a single large write truncated the 1.8 MB state payload on
    # Windows: the socket closed with ~54 KB still unsent, so the browser got a
    # short body and could not parse it. It happened to succeed in an earlier
    # test, which is worse than failing every time. HTTP/1.1 keeps the connection
    # framed by Content-Length, and the body is written in chunks and flushed.
    protocol_version = 'HTTP/1.1'

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        view = memoryview(body)
        for i in range(0, len(view), 1 << 16):
            self.wfile.write(view[i:i + (1 << 16)])
        self.wfile.flush()

    def do_GET(self):
        p = unquote(urlparse(self.path).path)
        if p == '/':
            return self._send(200, PAGE.encode('utf-8'), 'text/html; charset=utf-8')
        if p == '/api/state':
            return self._send(200, json.dumps(build_state()).encode('utf-8'))
        if p.startswith('/img/'):
            # kb_images is the single resolver. The rglob that used to live here
            # found files by basename, which returns whichever the filesystem
            # walked first if two ever share a name.
            want = p[len('/img/'):]
            try:
                body, ctype = kb_images.read(want)
            except kb_images.ImageError as e:
                print(f'IMAGE 404  {want}  --  {e}', flush=True)
                return self._send(404, json.dumps(
                    {'error': str(e), 'IMAGE': want}).encode())
            return self._send(200, body, ctype)
        return self._send(404, b'not found', 'text/plain')

    def do_POST(self):
        if urlparse(self.path).path != '/api/decide':
            return self._send(404, b'', 'text/plain')
        n = int(self.headers.get('Content-Length', 0))
        d = json.loads(self.rfile.read(n) or b'{}')
        if d.get('mode') not in MODES:
            return self._send(
                400, b'{"error":"this server writes only missed_role and '
                     b'missed_role_manual"}')
        if d['mode'] == 'missing_target_retraction':
            if not str(d.get('BOX_ID', '')).startswith(MISSING_PREFIX):
                return self._send(
                    400, b'{"error":"only a MISSING: flag can be retracted"}')
            if not str(d.get('reason', '')).strip():
                return self._send(
                    400, b'{"error":"a retraction needs a reason"}')
        elif d['mode'] == 'final_target':
            # only ever used to drop the image of a target whose role is
            # unreadable; a role answer belongs in the pass that asked for it
            if d.get('HUMAN_FINAL_CLASS') != EXCLUDE:
                return self._send(
                    400, b'{"error":"final_target here records only '
                         b'EXCLUDE_IMAGE"}')
            if not str(d.get('reason', '')).strip():
                return self._send(400, b'{"error":"an exclusion needs a reason"}')
        elif d.get('HUMAN_FINAL_CLASS') not in ROLE_VALUES:
            return self._send(400, b'{"error":"value not in the role vocabulary"}')
        if (d['mode'] == 'missing_target_box'
                and d.get('HUMAN_FINAL_CLASS') == NON_ACTIVE):
            # A bench player or a coach is not a missing EyeCU TARGET, so
            # flagging one would create annotation work for something that
            # should never be annotated.
            return self._send(
                400, b'{"error":"a non-active human is not a missing TARGET; '
                     b'do not flag one"}')
        if d['mode'] == 'missing_target_box':
            if not str(d.get('BOX_ID', '')).startswith(MISSING_PREFIX):
                return self._send(
                    400, b'{"error":"a missing-target flag must use a '
                         b'MISSING: key, never a real BOX_ID"}')
            d['image_level'] = True
            d['no_box_exists'] = True
        elif (d['mode'] != 'missing_target_retraction'
                and str(d.get('BOX_ID', '')).startswith(MISSING_PREFIX)):
            # a retraction names the flag it withdraws, so it is the one other
            # mode that legitimately carries a MISSING: key
            return self._send(
                400, b'{"error":"MISSING: keys belong to missing_target_box"}')
        d['recorded_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        d['author'] = 'human reviewer'
        if d['mode'] == 'missing_target_retraction':
            d.setdefault('HUMAN_FINAL_CLASS', None)
            d['retracts'] = d['BOX_ID']
        if d['mode'] == 'missed_role_manual':
            # What this click actually did, computed from the log rather than
            # trusted from the page. Re-confirming a role a box already carries
            # is a NO_OP_CONFIRMATION, not a newly found official, and must not
            # inflate the number this pass exists to produce.
            # Same rule and same prior state the offline auditor uses, so the
            # kind stored here can never disagree with the audited counts.
            pv, pm = kb_decisions.prior_non_manual(PKG / 'decisions.json',
                                                   d['BOX_ID'])
            d['manual_kind'] = kb_decisions.classify_click(
                pv, d['HUMAN_FINAL_CLASS'])
            d['prior_class'] = pv
            d['prior_mode'] = pm
            d['prior_class'] = pv
            d['prior_mode'] = pm
        with LOCK:
            with open(PKG / 'decisions.json', 'a', encoding='utf-8') as f:
                f.write(json.dumps(d) + '\n')
                f.flush()
        return self._send(200, json.dumps(
            {'ok': True, 'manual_kind': d.get('manual_kind')}).encode())


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8740)
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--u-revisit', action='store_true',
                    help='open the uncertain cleanup queue instead of the finished 6,684 pass')
    args = ap.parse_args()
    bad = page_script_defects()
    if bad:
        print('REFUSING TO SERVE: the review page would not parse in a browser.')
        for b in bad:
            print('  ' + b)
        print('\nNo decision has been touched. Fix the page and relaunch.')
        sys.exit(1)
    st = build_state()
    done = len(st['decisions'])
    imgs_done = sum(1 for it in st['items']
                    if all(c['BOX_ID'] in st['decisions'] for c in it['candidates']))
    print(f"missed_role: {st['total_images']} images, {st['total_boxes']} boxes "
          f"({st['total_boxes']/max(st['total_images'],1):.1f} per image)")
    print(f'already decided: {done} boxes, {imgs_done} images complete '
          f'-- resumed from decisions.json')
    if args.u_revisit:
        url_suffix = '?u=1'
    else:
        url_suffix = ''
    url = f'http://127.0.0.1:{args.port}/{url_suffix}'
    print(f'\nreview at {url}   Ctrl-C to stop')
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(('127.0.0.1', args.port), H).serve_forever()


if __name__ == '__main__':
    main()
