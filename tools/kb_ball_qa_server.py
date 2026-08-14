#!/usr/bin/env python
"""
BALL QA ROUND 0 -- the review UI for the 300-image model-independent sample.

One question per image:

    Is there any visible football in this image that is NOT annotated?

    NO MISSING BALL / MISSING BALL / UNSURE

NO DETECTOR IS LOADED HERE, and that is a property of the tool rather than a
promise in a docstring: nothing in this module imports torch or ultralytics,
nothing reads a .pt file, and no proposal of any kind is shown to the reviewer.
Round 0's independence is a property of the REVIEW, not only of the sampling --
a reviewer primed by a model's guesses inherits the model's blind spots, which
is precisely the circularity Round 0 exists to escape.

THE ENDPOINT IS THE IMAGE. An image holding three missing balls is ONE positive
image and three objects. The two are recorded separately and never added: only
the image has a defined inclusion probability, so only the image can carry a
confidence interval.

TILE SWEEP. A 4 px ball on a 1280x720 frame is about three screen pixels at
fit-to-window -- genuinely invisible, not merely small. A generic zoom control
is not enough, because it leaves the reviewer to remember which parts of a large
canvas they have already looked at. So the UI offers a guided sweep: FIT, then
tile 1..N at high magnification, each tile a fixed region with overlap so a ball
on a seam is not cut in half. The sweep does not gate the answer -- the UI
cannot actually prove an eye rested on a tile, and a checkbox claiming otherwise
would be a false record.

    python tools/kb_ball_qa_server.py

Findings are QA findings, appended to decisions.json under ball_qa_r0 modes.
They are NOT promoted into repaired_export by this tool. The measurement is
frozen before any correction is applied, because a defect that is corrected
still happened, and removing it from the numerator would mean measuring the
dataset after fixing exactly the parts we looked at.
"""

import argparse
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
import kb_decisions                                               # noqa: E402
import kb_images                                                  # noqa: E402
from kb_missing_target_server import validate_bbox                # noqa: E402

PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
DECISIONS = PKG / 'decisions.json'
LOCK = threading.Lock()

QA_MODE = 'ball_qa_r0'
ANSWERS = ('NO_MISSING_BALL', 'MISSING_BALL', 'UNSURE')

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>ball QA round 0</title>
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
 .ex{position:absolute;border:1px solid #3a3a3a;box-sizing:border-box;
     pointer-events:none}
 .ball{position:absolute;border:2px solid #4aa3ff;box-sizing:border-box;
       pointer-events:none;box-shadow:0 0 0 1px #000}
 .new{position:absolute;border:2px dashed #3ddc57;box-sizing:border-box;
      pointer-events:none;background:#3ddc5722}
 .saved{position:absolute;border:2px solid #3ddc57;box-sizing:border-box;
        pointer-events:none}
 .tag{position:absolute;font:11px/1.3 monospace;background:#000d;padding:1px 4px;
      transform:translateY(-100%);white-space:nowrap;pointer-events:none}
 #hit{position:absolute;inset:0}
 #hit.draw{cursor:crosshair}
 #panel{position:fixed;right:10px;top:52px;width:276px;background:#141414;
        border:1px solid #2a2a2a;border-radius:6px;padding:9px;z-index:8;
        max-height:84vh;overflow:auto}
 button{background:#1e1e1e;color:#ddd;border:1px solid #3a3a3a;border-radius:4px;
        padding:3px 8px;cursor:pointer;font:12px system-ui}
 button:hover{background:#2a2a2a}
 button:disabled{opacity:.35;cursor:not-allowed}
 button.big{width:100%;margin-top:6px;padding:6px}
 .k{background:#222;border:1px solid #3a3a3a;border-radius:3px;padding:0 5px;
    font:11px monospace}
 .warn{background:#3a1414;border:1px solid #7a3a3a;border-radius:4px;padding:6px;
       margin:6px 0;font-size:11px}
 .ok{background:#132a13;border:1px solid #2f5a2f;border-radius:4px;padding:6px;
     margin:6px 0;font-size:11px}
 .unsure{background:#2f2a10;border:1px solid #6a5a20;border-radius:4px;
         padding:6px;margin:6px 0;font-size:11px}
 .note{color:#8a8a8a;font-size:11px;margin:-2px 0 8px}
 #tiles{display:flex;gap:3px;flex-wrap:wrap;margin:5px 0}
 #tiles button{padding:2px 7px;font:11px monospace}
 #tiles button.on{background:#20402a;border-color:#3ddc57;color:#cfc}
 #tiles button.seen{border-color:#4a5a4a}
 .qbox{font-size:12px;font-weight:600;margin:2px 0 6px}
</style></head><body>
<div id="top">
 <b>BALL QA ROUND 0</b>
 <span id="pos" class="pill"></span>
 <span id="rem" class="pill"></span>
 <span id="img" class="pill" style="font:11px monospace"></span>
 <div id="bar"><i></i></div>
 <span id="zoomp" class="pill"></span>
 <span id="hint" style="color:#8a8a8a;font-size:11px"></span>
</div>
<div id="imgerr" style="display:none;margin:10px 300px 10px 10px;background:#3a1414;
     border:1px solid #7a3a3a;border-radius:6px;padding:12px;max-width:640px"></div>
<div id="wrap"><div id="stage"><img id="im"><div id="ov"></div><div id="hit"></div></div></div>
<div id="panel">
 <div class="qbox">Is there any visible football in this image that is NOT
  annotated?</div>
 <div id="state"></div>
 <button class="big" id="aNo">NO MISSING BALL <span class="k">1</span></button>
 <button class="big" id="aYes">MISSING BALL <span class="k">2</span></button>
 <button class="big" id="aUns">UNSURE <span class="k">3</span></button>
 <div id="drawwrap" style="display:none">
  <div class="note" style="margin-top:8px">drag on the image to draw each
   missing ball, then FINALISE. Every ball drawn is one object; the image counts
   as one positive either way.</div>
  <div id="drawn" style="font:11px monospace;color:#bbb"></div>
  <button class="big" id="bUndo">UNDO LAST <span class="k">Z</span></button>
  <button class="big" id="bClear">CLEAR ALL</button>
  <button class="big" id="bFin" style="border-color:#2f5a2f">FINALISE POSITIVE
   <span class="k">Enter</span></button>
  <button class="big" id="bCancel">CANCEL</button>
 </div>
 <div style="border-top:1px solid #2a2a2a;margin:9px 0 6px"></div>
 <div style="font-size:11px;color:#8a8a8a">tile sweep &mdash; inspect the whole
  frame at high magnification</div>
 <div id="tiles"></div>
 <div class="note" id="tilenote"></div>
 <button class="big" id="bN">NEXT UNRESOLVED <span class="k">N</span></button>
 <button class="big" id="bB">PREVIOUS <span class="k">B</span></button>
 <div id="meta" style="font-size:11px;color:#777;margin-top:8px"></div>
 <div id="hist" style="font-size:11px;color:#888;margin-top:8px"></div>
</div>
<script>
let S=null,i=0,zoom=1,mode='view',draft=null,drag=null,drawn=[],tile=-1,seen={};
const TILE_ZOOM=4, TCOLS=4, TROWS=3, OVERLAP=0.12;
var boot=async function(){
 S=await (await fetch('/api/state')).json();
 i=S.items.findIndex(x=>!x.answer); if(i<0)i=0;
 render();
};
function cur(){return S.items[i];}
function render(){
 const t=cur(); if(!t)return;
 mode='view'; draft=null; drawn=[]; tile=-1; zoom=1;
 if(!seen[t.IMAGE])seen[t.IMAGE]={};
 document.getElementById('pos').textContent=`image ${i+1}/${S.items.length}`;
 document.getElementById('rem').textContent=`${remaining()} remaining`;
 document.getElementById('img').textContent=t.IMAGE;
 document.getElementById('hint').textContent=
  '1 none · 2 missing · 3 unsure · T tile sweep · +/- zoom · 0 fit';
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
   +'monospace">'+t.IMAGE+'</span><br>'+why+'<br><br>Do not answer this image.';};
 im.src='/img/'+t.IMAGE;
 tilebar(); stats();
}
function remaining(){return S.items.filter(x=>!x.answer||x.answer==='UNSURE').length;}
function applyZoom(){
 const st=document.getElementById('stage');
 st.style.transform='scale('+zoom+')';
 const im=document.getElementById('im');
 st.style.width=im.naturalWidth+'px'; st.style.height=im.naturalHeight+'px';
 document.getElementById('zoomp').textContent='zoom '+zoom.toFixed(2)+'x';
}
function toImage(ev){
 const im=document.getElementById('im');
 const r=im.getBoundingClientRect();
 return [(ev.clientX-r.left)/zoom,(ev.clientY-r.top)/zoom];
}
function tilebar(){
 const t=cur(),bar=document.getElementById('tiles');
 bar.innerHTML='';
 const mk=(label,idx)=>{
  const b=document.createElement('button');
  b.textContent=label;
  if(idx===tile)b.className='on';
  else if(idx>=0&&seen[t.IMAGE]&&seen[t.IMAGE][idx])b.className='seen';
  b.onclick=()=>gotoTile(idx);
  bar.appendChild(b);
 };
 mk('FIT',-1);
 for(let k=0;k<TCOLS*TROWS;k++)mk(String(k+1),k);
 const n=Object.keys(seen[t.IMAGE]||{}).length;
 document.getElementById('tilenote').textContent=
  n+' of '+(TCOLS*TROWS)+' tiles visited on this image (not required, and not '
  +'recorded as evidence)';
}
function gotoTile(idx){
 const t=cur(),im=document.getElementById('im');
 const W=im.naturalWidth||t.img_w, H=im.naturalHeight||t.img_h;
 if(idx<0){tile=-1;zoom=1;applyZoom();tilebar();return;}
 tile=idx; seen[t.IMAGE][idx]=true;
 zoom=TILE_ZOOM; applyZoom();
 const cw=W/TCOLS, ch=H/TROWS;
 const cx=(idx%TCOLS)*cw, cy=Math.floor(idx/TCOLS)*ch;
 const wrap=document.getElementById('wrap');
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
 if(S.show_context)t.context.forEach(b=>add('ex',b.bbox,null,null));
 t.ball_gt.forEach(b=>add('ball',b.bbox,'ball GT','#4aa3ff'));
 (t.answer==='MISSING_BALL'?t.missing:[]).forEach((b,k)=>
   add('saved',b.bbox_xywh,'MISSING '+(k+1),'#3ddc57'));
 drawn.forEach((b,k)=>add('saved',b,'new '+(k+1),'#3ddc57'));
 if(draft)add('new',draft,draft.map(v=>Math.round(v)).join(','),'#3ddc57');
 panel();
}
function panel(){
 const t=cur();
 const st=document.getElementById('state');
 if(t.answer==='NO_MISSING_BALL')
  st.innerHTML='<div class="ok">answered <b>NO MISSING BALL</b>. Answering again '
   +'appends a new event; nothing is rewritten.</div>';
 else if(t.answer==='MISSING_BALL')
  st.innerHTML='<div class="ok">answered <b>MISSING BALL</b> with <b>'
   +t.missing.length+'</b> object(s) drawn.<br>This image counts as <b>one</b> '
   +'positive image regardless of the object count.</div>';
 else if(t.answer==='UNSURE')
  st.innerHTML='<div class="unsure">answered <b>UNSURE</b>. This stays '
   +'unresolved and is reported separately &mdash; it is never counted as '
   +'clean.</div>';
 else st.innerHTML='<div class="warn">not answered yet. Sweep the whole frame '
   +'at zoom before answering; a 4 px ball is invisible at fit.</div>';
 document.getElementById('drawwrap').style.display=mode==='draw'?'block':'none';
 document.getElementById('hit').className=mode==='draw'?'draw':'';
 document.getElementById('drawn').innerHTML=drawn.length
  ?drawn.map((b,k)=>(k+1)+': ['+b.map(v=>Math.round(v)).join(', ')+']').join('<br>')
  :'<span style="color:#777">nothing drawn yet</span>';
 document.getElementById('bUndo').disabled=!drawn.length;
 document.getElementById('bFin').disabled=!drawn.length;
 document.getElementById('meta').innerHTML=
  `split <b>${t.split}</b> · run <b>${t.run||'?'}</b> · existing ball GT
   <b>${t.ball_gt.length}</b> · ${t.img_w}x${t.img_h}
   <div style="color:#666">metadata is descriptive; it played no part in
   selecting this image.</div>`;
 document.getElementById('hist').innerHTML=!t.history.length?'':
  '<b>history</b>'+t.history.map(h=>`<div>${(h.recorded_utc||'').slice(11,19)}
   ${h.answer} ${h.n_missing?'('+h.n_missing+' obj)':''}</div>`).join('');
}
function stats(){
 const d=S.items.filter(x=>x.answer).length;
 document.getElementById('bar').firstElementChild.style.width=
   (100*d/S.items.length)+'%';
 document.getElementById('rem').textContent=`${remaining()} remaining`;
}
const hit=document.getElementById('hit');
hit.onmousedown=e=>{if(mode!=='draw')return;drag=toImage(e);draft=null;draw();};
hit.onmousemove=e=>{
 if(mode!=='draw'||!drag)return;
 const p=toImage(e);
 draft=[Math.min(drag[0],p[0]),Math.min(drag[1],p[1]),
        Math.abs(p[0]-drag[0]),Math.abs(p[1]-drag[1])];
 draw();
};
hit.onmouseup=e=>{
 if(mode!=='draw'||!drag)return;
 const p=toImage(e);
 let b=[Math.min(drag[0],p[0]),Math.min(drag[1],p[1]),
        Math.abs(p[0]-drag[0]),Math.abs(p[1]-drag[1])];
 drag=null; draft=null;
 if(b[2]>=1&&b[3]>=1)drawn.push(b);
 draw();
};
async function answer(a){
 const t=cur();
 if(a==='MISSING_BALL'&&mode!=='draw'){mode='draw';drawn=[];draw();return;}
 const body={IMAGE:t.IMAGE,answer:a};
 if(a==='MISSING_BALL')body.missing_balls_xywh=drawn;
 const r=await fetch('/api/answer',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const j=await r.json();
 if(!r.ok){alert(j.error||'refused');return;}
 t.answer=a; t.missing=j.missing||[];
 t.history.push({answer:a,n_missing:(j.missing||[]).length,
                 recorded_utc:j.recorded_utc});
 mode='view'; drawn=[]; draft=null;
 draw(); stats(); next();
}
function next(){
 for(let k=i+1;k<S.items.length;k++){if(!S.items[k].answer){i=k;render();return;}}
 for(let k=0;k<S.items.length;k++){if(!S.items[k].answer){i=k;render();return;}}
 render();
}
document.getElementById('aNo').onclick=()=>answer('NO_MISSING_BALL');
document.getElementById('aYes').onclick=()=>answer('MISSING_BALL');
document.getElementById('aUns').onclick=()=>answer('UNSURE');
document.getElementById('bUndo').onclick=()=>{drawn.pop();draw();};
document.getElementById('bClear').onclick=()=>{drawn=[];draw();};
document.getElementById('bFin').onclick=()=>answer('MISSING_BALL');
document.getElementById('bCancel').onclick=()=>{mode='view';drawn=[];draft=null;draw();};
document.getElementById('bN').onclick=next;
document.getElementById('bB').onclick=()=>{if(i>0){i--;render();}};
document.onkeydown=e=>{
 const k=e.key.toLowerCase();
 if(k==='1')answer('NO_MISSING_BALL');
 else if(k==='2')answer('MISSING_BALL');
 else if(k==='3')answer('UNSURE');
 else if(k==='enter'&&mode==='draw'&&drawn.length)answer('MISSING_BALL');
 else if(k==='escape'&&mode==='draw'){mode='view';drawn=[];draft=null;draw();}
 else if(k==='z'&&mode==='draw'){drawn.pop();draw();}
 else if(k==='t'){gotoTile(tile+1>=TCOLS*TROWS?-1:tile+1);}
 else if(k==='n'){e.preventDefault();next();}
 else if(k==='b'){if(i>0){i--;render();}}
 else if(e.key==='+'||e.key==='='){zoom=Math.min(zoom*1.25,12);applyZoom();}
 else if(e.key==='-'){zoom=Math.max(zoom/1.25,0.25);applyZoom();}
 else if(k==='0'){zoom=1;tile=-1;applyZoom();tilebar();}
};
boot();
</script></body></html>"""


def load_sample():
    """The drawn sample, refusing to proceed if the export has since changed."""
    p = kb_ball_qa_sample.MANIFEST
    if not p.is_file():
        raise SystemExit(
            f'no Round-0 sample at {p}\nrun: python tools/kb_ball_qa_sample.py')
    man = json.loads(p.read_text(encoding='utf-8'))
    live = kb_ball_qa_sample.population_fingerprint()
    if live['population_sha256'] != man['population']['population_sha256']:
        raise SystemExit(
            'REFUSING TO START: the promoted export has changed since this '
            'sample was drawn.\n'
            f'  sample drawn against {man["population"]["population_sha256"][:16]}...\n'
            f'  export is now        {live["population_sha256"][:16]}...\n'
            'The sample is invalid. Answers collected against different '
            'annotations would not be measuring one population.')
    return man


def answers(path: Path = DECISIONS):
    """Effective Round-0 answer per IMAGE. Latest event wins; nothing rewritten.

    Keyed by IMAGE, not BOX_ID: the endpoint is the image. A re-answer appends a
    new event and supersedes the earlier one exactly as kb_decisions.resolve()
    does for boxes, and the whole history stays readable.
    """
    per = {}
    for d in kb_decisions.read_log(path):
        if d.get('mode') == QA_MODE:
            per.setdefault(d['IMAGE'], []).append(d)
    out = {}
    for im, evs in per.items():
        evs = sorted(evs, key=lambda d: (d.get('recorded_utc') or '', d['_line']))
        win = evs[-1]
        out[im] = {
            'answer': win.get('answer'),
            'missing': win.get('missing_balls') or [],
            'recorded_utc': win.get('recorded_utc'),
            'history': [{'answer': e.get('answer'),
                         'n_missing': len(e.get('missing_balls') or []),
                         'recorded_utc': e.get('recorded_utc')} for e in evs],
        }
    return out


def build_state(show_context=False):
    man = load_sample()
    ans = answers()
    doc_cache = {}
    items = []
    for r in man['sample']:
        split = r['split']
        if split not in doc_cache:
            d = json.loads((kb_ball_qa_sample.EXPORT /
                            f'{split}_annotations.coco.json')
                           .read_text(encoding='utf-8'))
            bc = kb_ball_qa_sample._ball_category(d['categories'])
            by_img = {}
            for a in d['annotations']:
                by_img.setdefault(a['image_id'], []).append(a)
            doc_cache[split] = (bc, by_img)
        bc, by_img = doc_cache[split]
        rows = by_img.get(r['coco_image_id'], [])
        a = ans.get(r['IMAGE'], {})
        items.append({
            'IMAGE': r['IMAGE'], 'split': split, 'run': r['run'],
            'img_w': r['img_w'], 'img_h': r['img_h'],
            'ball_gt': [{'bbox': x['bbox']} for x in rows
                        if x['category_id'] == bc],
            'context': [{'bbox': x['bbox']} for x in rows
                        if x['category_id'] != bc],
            'answer': a.get('answer'),
            'missing': a.get('missing', []),
            'history': a.get('history', []),
        })
    return {'round': 0, 'items': items, 'show_context': show_context,
            'sample_seed': man['seed'], 'n': man['n'],
            'population_sha256': man['population']['population_sha256'],
            'source_log': kb_decisions.log_version(DECISIONS)}


def append(rec):
    with LOCK:
        with open(DECISIONS, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec) + '\n')
            fh.flush()


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    SHOW_CONTEXT = False
    SAMPLED = frozenset()
    DIMS = {}

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
            return self._send(200, PAGE.encode('utf-8'),
                              'text/html; charset=utf-8')
        if p == '/api/state':
            return self._send(200, json.dumps(
                build_state(self.SHOW_CONTEXT)).encode())
        if p.startswith('/img/'):
            want = p[len('/img/'):]
            # only the 300 sampled images are reachable. Serving anything else
            # would let a mistyped URL put an unsampled image in front of the
            # reviewer, and an answer on it would not belong to the sample.
            if want not in self.SAMPLED:
                return self._send(403, json.dumps(
                    {'error': 'not in the Round-0 sample', 'IMAGE': want}).encode())
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
        if route != '/api/answer':
            return self._send(404, b'', 'text/plain')

        image = str(d.get('IMAGE', ''))
        if image not in self.SAMPLED:
            return self._send(400, json.dumps(
                {'error': 'image is not in the Round-0 sample; answering it '
                          'would not belong to the measured population',
                 'IMAGE': image}).encode())
        ans = d.get('answer')
        if ans not in ANSWERS:
            return self._send(400, json.dumps(
                {'error': f'answer must be one of {ANSWERS}'}).encode())

        w, h = self.DIMS.get(image, (None, None))
        missing = []
        if ans == 'MISSING_BALL':
            raw = d.get('missing_balls_xywh') or []
            if not raw:
                return self._send(400, json.dumps(
                    {'error': 'MISSING_BALL requires at least one drawn ball; '
                              'use UNSURE if it cannot be located'}).encode())
            for b in raw:
                bbox, err = validate_bbox(b, w, h)
                if err:
                    return self._send(400, json.dumps({'error': err}).encode())
                missing.append({'bbox_xywh': bbox,
                                'geometry_author': 'human drawn',
                                'coordinate_space': 'original image pixels'})

        now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        append({
            'mode': QA_MODE,
            # the endpoint is the image, so the log is keyed by image. BOX_ID
            # mirrors it so the shared reader, which assumes that field, keeps
            # working -- these events are not about any existing annotation.
            'BOX_ID': f'BALLQA0:{image}',
            'IMAGE': image,
            'answer': ans,
            'HUMAN_FINAL_CLASS': None,
            'image_is_positive': ans == 'MISSING_BALL',
            'missing_balls': missing,
            'n_missing_objects': len(missing),
            'img_w': w, 'img_h': h,
            'round': 0,
            'no_model_proposal_used': True,
            'no_detector_consulted': True,
            'recorded_utc': now, 'author': 'human reviewer',
        })
        return self._send(200, json.dumps(
            {'ok': True, 'missing': missing, 'recorded_utc': now}).encode())


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8744)
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--show-context', action='store_true',
                    help='outline non-ball annotations faintly as context')
    args = ap.parse_args()

    man = load_sample()
    H.SHOW_CONTEXT = args.show_context
    H.SAMPLED = frozenset(r['IMAGE'] for r in man['sample'])
    H.DIMS = {r['IMAGE']: (r['img_w'], r['img_h']) for r in man['sample']}

    ok, problems = kb_images.preflight(sorted(H.SAMPLED))
    if not ok:
        print(f'REFUSING TO START: {len(problems)} image(s) cannot be resolved.')
        for im, why in problems[:10]:
            print(f'  {im}\n    {why}')
        sys.exit(1)

    ans = answers()
    done = sum(1 for im in H.SAMPLED if ans.get(im, {}).get('answer'))
    unsure = sum(1 for im in H.SAMPLED
                 if ans.get(im, {}).get('answer') == 'UNSURE')
    print(f'BALL QA ROUND 0 -- {man["n"]} images, seed {man["seed"]}, '
          f'inclusion probability {man["inclusion_probability"]:.6f} each')
    print(f'population fingerprint {man["population"]["population_sha256"][:16]}... '
          f'matches the promoted export')
    print(f'{done} answered, {man["n"] - done} outstanding, {unsure} UNSURE')
    print('preflight: every sampled image resolves')
    print('\nNO DETECTOR IS LOADED. Nothing is proposed to the reviewer.')
    print('Findings are QA findings; they are not promoted into the export.')
    url = f'http://127.0.0.1:{args.port}/'
    print(f'\nreview at {url}   Ctrl-C to stop')
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(('127.0.0.1', args.port), H).serve_forever()


if __name__ == '__main__':
    main()
