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
</style></head><body>
<div id="top">
 <b>BALL ONTOLOGY REVISIT</b>
 <span id="pos" class="pill"></span>
 <span id="rem" class="pill"></span>
 <span id="img" class="pill" style="font:11px monospace"></span>
 <div id="bar"><i></i></div>
 <span id="zoomp" class="pill"></span>
 <span style="color:#8a8a8a;font-size:11px">A active &middot; X non-active
  &middot; U unsure &middot; J/K move &middot; B previous &middot; T tiles</span>
</div>
<div id="imgerr" style="display:none;margin:10px 300px 10px 10px;background:#3a1414;
     border:1px solid #7a3a3a;border-radius:6px;padding:12px;max-width:640px"></div>
<div id="wrap"><div id="stage"><img id="im"><div id="ov"></div></div></div>
<div id="panel">
 <div class="qbox">Is the <span style="color:#ff2fd0">highlighted</span> ball the
  ACTIVE MATCH BALL?</div>
 <div id="legend">
  <div><i style="background:#ff2fd0"></i>the object being classified</div>
  <div><i style="background:#7a7a7a"></i>other Round-0 findings in this image</div>
  <div><i style="background:#4aa3ff"></i>existing ball annotations</div>
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
 <div style="font-size:11px;color:#8a8a8a">tile sweep</div>
 <div id="tiles"></div>
 <button class="big" id="bJ"><span class="k">J</span>NEXT UNCLASSIFIED</button>
 <button class="big" id="bB"><span class="k">B</span>PREVIOUS</button>
 <div id="meta" style="font-size:11px;color:#777;margin-top:8px"></div>
 <div id="hist" style="font-size:11px;color:#888;margin-top:8px"></div>
</div>
<script>
let S=null,i=0,zoom=1,tile=-1,seen={};
const TILE_ZOOM=4,TCOLS=4,TROWS=3,OVERLAP=0.12;
var boot=async function(){
 S=await (await fetch('/api/state')).json();
 i=S.items.findIndex(x=>!x.role); if(i<0)i=0;
 render();
};
function cur(){return S.items[i];}
function remaining(){return S.items.filter(x=>!x.role).length;}
function render(){
 const t=cur(); if(!t)return;
 zoom=1; tile=-1;
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
 t.ball_gt.forEach(b=>add('ballgt',b.bbox,'ball GT','#4aa3ff'));
 t.siblings.forEach((b,k)=>add('sibling',b.bbox_xywh,
   'other finding'+(b.role?' → '+b.short:''),'#7a7a7a'));
 add('target',t.bbox_xywh,'CLASSIFY THIS','#ff2fd0');
 panel();
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
 else st.innerHTML='<div class="warn">not classified yet. Press BALL in the tile '
   +'bar to jump to this object at 4x.</div>';
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
document.getElementById('aA').onclick=()=>classify(S.ACTIVE);
document.getElementById('aX').onclick=()=>classify(S.NON_ACTIVE);
document.getElementById('aU').onclick=()=>classify(S.UNSURE);
document.getElementById('bJ').onclick=next;
document.getElementById('bB').onclick=()=>{if(i>0){i--;render();}};
document.onkeydown=e=>{
 const k=e.key.toLowerCase();
 // 'n' is deliberately NOT bound. It reads as both NEXT and NON-ACTIVE, and
 // either meaning would be a silent misclassification of a real finding.
 if(k==='a')classify(S.ACTIVE);
 else if(k==='x')classify(S.NON_ACTIVE);
 else if(k==='u')classify(S.UNSURE);
 else if(k==='j'){e.preventDefault();next();}
 else if(k==='k'||k==='b'){if(i>0){i--;render();}}
 else if(k==='t'){gotoTile(tile+1>=TCOLS*TROWS?-1:tile+1);}
 else if(k==='c'){gotoTile(-2);}
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


def build_state(show_context=False, decisions: Path = DECISIONS):
    objs = round0_objects(decisions)
    onto = ontology(decisions)
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
            'ball_gt': [{'bbox': x['bbox']} for x in rows
                        if x['category_id'] == bc],
            'context': [{'bbox': x['bbox']} for x in rows
                        if x['category_id'] != bc],
            'siblings': sibs,
            'role': rec.get('role'),
            'history': rec.get('history', []),
        })
    return {'items': items, 'show_context': show_context,
            'ACTIVE': ACTIVE, 'NON_ACTIVE': NON_ACTIVE, 'UNSURE': UNSURE,
            'source_log': kb_decisions.log_version(decisions)}


def append(rec):
    with LOCK:
        with open(DECISIONS, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec) + '\n')
            fh.flush()


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    SHOW_CONTEXT = False
    OBJECTS = {}

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
        if urlparse(self.path).path != '/api/ontology':
            return self._send(404, b'', 'text/plain')
        n = int(self.headers.get('Content-Length', 0))
        d = json.loads(self.rfile.read(n) or b'{}')

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
    print(f'BALL ONTOLOGY REVISIT -- {len(objs)} Round-0 missing objects '
          f'across {imgs} images')
    print(f'frozen Round 0 stands unchanged: '
          f'{res["primary"]["positive_images"]}/{res["primary"]["n"]} positive '
          f'images, {want} objects')
    print(f'{done} classified, {len(objs) - done} outstanding')
    print('preflight: every image resolves')
    print('\nNO GEOMETRY IS CREATED OR EDITED. No detector is loaded.')
    print("Keys: A active  X non-active  U unsure  J next  B/K previous"
          "  ('n' is unbound on purpose)")
    url = f'http://127.0.0.1:{args.port}/'
    print(f'\nreview at {url}   Ctrl-C to stop')
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(('127.0.0.1', args.port), H).serve_forever()


if __name__ == '__main__':
    main()
