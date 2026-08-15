#!/usr/bin/env python
"""
The last two cleanup queues: 7 bad boxes to redraw, 5 ball cases to decide.

Both are tiny, both need a human drawing on an image, and both exist because the
honest answer to a bad annotation is not always "delete it".

  --partial   the 7 PARTIAL_BODY_BAD_BOX annotations. A real person is in each
              box; the box covers a fragment while much more of the body is
              visible. Keeping it teaches a wrong extent, removing it turns a
              visible player into background, and excluding the whole image
              throws away every other correct annotation in it. So it is redrawn.

  --ball      the 5 BALL_WRONG_HUMAN_BOX annotations. The audit found the thing
              that makes "just remove it" wrong: ALL FIVE images contain zero
              ball GT. Removing the mis-drawn human box would leave a visible
              ball unlabelled, which trains the detector that a ball there is
              background -- the exact failure this dataset was wanted to fix.
              A human decides per case, and no ball is ever created automatically.

WHAT NEVER HAPPENS HERE. No model runs, nothing is proposed, no geometry is
inferred, and the source stays immutable. A repair is an append-only event
carrying the ORIGINAL geometry alongside the replacement, so the first bbox is
recoverable from the log forever. Annotation identity is preserved: a repaired
box keeps its BOX_ID, because it is the same annotation of the same object.

    python tools/kb_geometry_repair_server.py --partial
    python tools/kb_geometry_repair_server.py --ball

Nothing is applied. kb_apply_review.py --apply remains the only writer.
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
import kb_decisions                                              # noqa: E402
import kb_images                                                 # noqa: E402
from kb_missing_target_server import validate_bbox               # noqa: E402

PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
LOCK = threading.Lock()
ROLES = ('player', 'goalkeeper', 'referee')

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>geometry repair</title>
<style>
 :root{--gk:#ffc400;--ref:#ff7a1a;--pl:#3ddc57;--ball:#4aa3ff}
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
 .ex{position:absolute;border:1px solid #4a4a4a;box-sizing:border-box;
     pointer-events:none}
 .ex.ball{border-color:var(--ball);border-width:2px}
 /* the annotation being repaired, so it cannot be mistaken for context */
 .bad{position:absolute;border:3px dashed #ff5a5a;box-sizing:border-box;
      pointer-events:none;background:#ff5a5a1a}
 .new{position:absolute;border:2px dashed #3ddc57;box-sizing:border-box;
      pointer-events:none;background:#3ddc5722}
 .saved{position:absolute;border:3px solid #3ddc57;box-sizing:border-box;
        pointer-events:none}
 .tag{position:absolute;font:11px/1.3 monospace;background:#000d;padding:1px 4px;
      transform:translateY(-100%);white-space:nowrap;pointer-events:none}
 #hit{position:absolute;inset:0;cursor:crosshair}
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
 .note{color:#8a8a8a;font-size:11px;margin:-2px 0 8px}
</style></head><body>
<div id="top">
 <b id="mode"></b>
 <span id="pos" class="pill"></span>
 <span id="img" class="pill" style="font:11px monospace"></span>
 <div id="bar"><i></i></div>
 <span id="prog" class="pill"></span>
 <span id="hint" style="color:#8a8a8a;font-size:11px"></span>
</div>
<div id="imgerr" style="display:none;margin:10px 300px 10px 10px;background:#3a1414;
     border:1px solid #7a3a3a;border-radius:6px;padding:12px;max-width:640px"></div>
<div id="wrap"><div id="stage"><img id="im"><div id="ov"></div><div id="hit"></div></div></div>
<div id="panel">
 <div id="who"></div>
 <div id="state"></div>
 <div id="acts"></div>
 <button class="big" id="bZ">UNDO DRAW <span class="k">Z</span></button>
 <button class="big" id="bN">NEXT <span class="k">N</span></button>
 <button class="big" id="bB">PREVIOUS <span class="k">B</span></button>
 <div id="hist" style="font-size:11px;color:#888;margin-top:8px"></div>
</div>
<script>
let S=null,i=0,draft=null,drag=null,zoom=1;
const COL={player:'#3ddc57',goalkeeper:'#ffc400',referee:'#ff7a1a',ball:'#4aa3ff'};
var boot=async function(){
 S=await (await fetch('/api/state')).json();
 i=S.items.findIndex(x=>!x.decision); if(i<0)i=0;
 render();
};
function cur(){return S.items[i];}
function render(){
 const t=cur(); if(!t)return;
 document.getElementById('mode').textContent=
   S.queue==='partial'?'GEOMETRY REPAIR':'BALL CASE REVIEW';
 document.getElementById('pos').textContent=`item ${i+1}/${S.items.length}`;
 document.getElementById('img').textContent=t.IMAGE;
 document.getElementById('hint').textContent=S.queue==='partial'
  ?'drag a correct box around the whole visible person, then P/G/R'
  :'A reclassify this box to ball · D draw the ball · R remove only';
 draft=null;
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
   +'monospace">'+t.IMAGE+'</span><br>'+why+'<br><br>Do not answer this item.';};
 im.src='/img/'+t.IMAGE;
 stats();
}
function applyZoom(){
 const st=document.getElementById('stage');
 st.style.transform='scale('+zoom+')';
 const im=document.getElementById('im');
 st.style.width=im.naturalWidth+'px'; st.style.height=im.naturalHeight+'px';
}
function toImage(ev){
 const im=document.getElementById('im');
 const r=im.getBoundingClientRect();
 return [(ev.clientX-r.left)/zoom,(ev.clientY-r.top)/zoom];
}
function draw(){
 const t=cur(),ov=document.getElementById('ov');
 ov.innerHTML='';
 const add=(cls,b,col,label)=>{
  const e=document.createElement('div'); e.className=cls;
  e.style.cssText=`left:${b[0]}px;top:${b[1]}px;width:${b[2]}px;height:${b[3]}px;`
   +(col?`border-color:${col};`:'');
  ov.appendChild(e);
  if(label){const g=document.createElement('div');g.className='tag';
   g.style.cssText=`left:${b[0]}px;top:${b[1]}px;color:${col||'#999'}`;
   g.textContent=label; ov.appendChild(g);}
 };
 t.existing.forEach(b=>{
  if(b.BOX_ID===t.BOX_ID)return;              // the bad box is drawn separately
  add('ex'+(b.cls==='ball'?' ball':''),b.bbox,null,b.cls==='ball'?'ball GT':null);
 });
 add('bad',t.bbox,null,'ORIGINAL '+t.original_class);
 if(t.decision&&t.decision.bbox) add('saved',t.decision.bbox,
   COL[t.decision.cls],'HUMAN '+t.decision.cls);
 if(draft) add('new',draft,null,'drawn '+draft.map(v=>Math.round(v)).join(','));
 panel();
}
function panel(){
 const t=cur();
 document.getElementById('who').innerHTML=
  `<div style="font-size:11px;color:#8a8a8a">annotation</div>
   <div style="font:11px monospace;color:#bbb">${t.BOX_ID}</div>
   <div style="font-size:11px;margin-top:3px">original class
    <b>${t.original_class}</b> · original box
    <span style="font:11px monospace">${t.bbox.map(v=>Math.round(v)).join(', ')}</span></div>
   <div style="font-size:11px;color:#8a8a8a">the annotation ID is kept; only its
    geometry or class changes, and the original stays in history.</div>`;
 const st=document.getElementById('state');
 const acts=document.getElementById('acts');
 if(S.queue==='partial'){
  st.innerHTML=t.decision
   ?`<div class="ok">REPAIRED to <b>${t.decision.cls}</b> at
      [${t.decision.bbox.map(v=>Math.round(v)).join(', ')}].
      Drawing again records another repair; both stay in history.</div>`
   :(draft?'<div class="ok">box drawn &mdash; now choose the class</div>'
          :'<div class="warn">a real person is in the red box, but it covers a '
           +'fragment. Drag a box around the whole visible extent.</div>');
  acts.innerHTML=`<button class="big" id="aP">PLAYER <span class="k">P</span></button>
   <button class="big" id="aG">GOALKEEPER <span class="k">G</span></button>
   <button class="big" id="aR">REFEREE <span class="k">R</span></button>`;
  for(const [id,role] of [['aP','player'],['aG','goalkeeper'],['aR','referee']]){
   const b=document.getElementById(id); b.disabled=!draft;
   b.onclick=()=>saveRepair(role);
  }
 } else {
  const nb=t.balls_in_image;
  st.innerHTML=t.decision
   ?`<div class="ok">DECIDED <b>${t.decision.action}</b>${t.decision.bbox
      ?' at ['+t.decision.bbox.map(v=>Math.round(v)).join(', ')+']':''}</div>`
   :`<div class="warn">this human box is on the ball, not a person.<br>
      ball annotations already in this image: <b>${nb}</b>${nb?'':' &mdash; '
      +'removing this box alone would leave the ball unlabelled and train it as '
      +'background.'}</div>`;
  acts.innerHTML=`
   <button class="big" id="aA">A &mdash; RECLASSIFY THIS BOX TO BALL</button>
   <div class="note">only if the existing red box accurately bounds the ball.
    Keeps the annotation ID.</div>
   <button class="big" id="aD">D &mdash; DRAW THE CORRECT BALL BOX</button>
   <div class="note">clearly a ball, but the red geometry is unsuitable. Draw
    first, then press D.</div>
   <button class="big" id="aR2" style="border-color:#7a3a3a">R &mdash; REMOVE ONLY</button>
   <div class="note">no valid ball annotation should exist here.</div>`;
  document.getElementById('aA').onclick=()=>saveBall('RECLASSIFY_TO_BALL');
  document.getElementById('aD').onclick=()=>saveBall('DRAW_BALL_BOX');
  document.getElementById('aD').disabled=!draft;
  document.getElementById('aR2').onclick=()=>saveBall('REMOVE_ONLY');
 }
 document.getElementById('bZ').disabled=!draft;
 document.getElementById('hist').innerHTML=!t.history.length?'':
  '<b>history</b>'+t.history.map(h=>`<div>${(h.recorded_utc||'').slice(11,19)}
   ${h.mode} ${h.value||''}</div>`).join('');
}
function stats(){
 const d=S.items.filter(x=>x.decision).length;
 document.getElementById('prog').textContent=`${d}/${S.items.length} decided`;
 document.getElementById('bar').firstElementChild.style.width=
   (100*d/S.items.length)+'%';
}
const hit=document.getElementById('hit');
hit.onmousedown=e=>{drag=toImage(e);draft=null;draw();};
hit.onmousemove=e=>{
 if(!drag)return;
 const p=toImage(e);
 draft=[Math.min(drag[0],p[0]),Math.min(drag[1],p[1]),
        Math.abs(p[0]-drag[0]),Math.abs(p[1]-drag[1])];
 draw();
};
hit.onmouseup=e=>{
 if(!drag)return;
 const p=toImage(e);
 draft=[Math.min(drag[0],p[0]),Math.min(drag[1],p[1]),
        Math.abs(p[0]-drag[0]),Math.abs(p[1]-drag[1])];
 drag=null;
 if(draft[2]<2||draft[3]<2)draft=null;
 draw();
};
async function saveRepair(role){
 const t=cur(); if(!draft)return;
 const r=await fetch('/api/repair',{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({BOX_ID:t.BOX_ID,HUMAN_FINAL_CLASS:role,
                       replacement_bbox_xywh:draft})});
 const j=await r.json();
 if(!r.ok){alert(j.error||'refused');return;}
 t.decision={cls:role,bbox:j.replacement_bbox_xywh};
 t.history.push({mode:'geometry_repair',value:role,recorded_utc:j.recorded_utc});
 draft=null; draw(); stats(); next();
}
async function saveBall(action){
 const t=cur();
 if(action==='DRAW_BALL_BOX'&&!draft){alert('draw the ball box first');return;}
 const body={BOX_ID:t.BOX_ID,action:action};
 if(action==='DRAW_BALL_BOX')body.ball_bbox_xywh=draft;
 const r=await fetch('/api/ball',{method:'POST',
  headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
 const j=await r.json();
 if(!r.ok){alert(j.error||'refused');return;}
 t.decision={action:action,bbox:j.ball_bbox_xywh||null};
 t.history.push({mode:'ball_case_resolution',value:action,
                 recorded_utc:j.recorded_utc});
 draft=null; draw(); stats(); next();
}
function next(){if(i<S.items.length-1){i++;render();}}
document.getElementById('bZ').onclick=()=>{draft=null;draw();};
document.getElementById('bN').onclick=next;
document.getElementById('bB').onclick=()=>{if(i>0){i--;render();}};
document.onkeydown=e=>{
 const k=e.key.toLowerCase();
 if(S.queue==='partial'){
  if(k==='p')saveRepair('player'); else if(k==='g')saveRepair('goalkeeper');
  else if(k==='r')saveRepair('referee');
 } else {
  if(k==='a')saveBall('RECLASSIFY_TO_BALL');
  else if(k==='d')saveBall('DRAW_BALL_BOX');
  else if(k==='r')saveBall('REMOVE_ONLY');
 }
 if(k==='z'){draft=null;draw();}
 else if(k==='n'){e.preventDefault();next();}
 else if(k==='b'){if(i>0){i--;render();}}
 else if(e.key==='+'||e.key==='='){zoom=Math.min(zoom*1.25,8);applyZoom();}
 else if(e.key==='-'){zoom=Math.max(zoom/1.25,0.25);applyZoom();}
 else if(k==='0'){zoom=1;applyZoom();}
};
boot();
</script></body></html>"""


def _ledger():
    return {r['BOX_ID']: r for r in
            json.loads((PKG / 'ledger.json').read_text(encoding='utf-8'))}


def queue_ids(which):
    """The boxes in scope: every one a human EVER dispositioned this way.

    Not the current effective disposition. A geometry_repair carries a role, so
    once a box is repaired its effective disposition is gone and it would drop
    out of its own queue -- leaving no way to look at the repair again, and no
    way for the exporter to tell a repaired box from an ordinary class change.
    Membership is historical; the repair is the resolution, exactly as a
    missing-target flag is resolved by a drawn box rather than erased by one.
    """
    want = 'PARTIAL_BODY_BAD_BOX' if which == 'partial' else 'BALL_WRONG_HUMAN_BOX'
    out = set()
    for d in kb_decisions.read_log(PKG / 'decisions.json'):
        if d.get('HUMAN_FINAL_CLASS') == want:
            out.add(d['BOX_ID'])
    return sorted(out)


def build_state(which):
    led = _ledger()
    by_img = {}
    for r in led.values():
        by_img.setdefault(r['IMAGE'], []).append(r)
    reps = kb_decisions.geometry_repairs(PKG / 'decisions.json')
    balls = kb_decisions.ball_cases(PKG / 'decisions.json')
    hist = {}
    for d in kb_decisions.read_log(PKG / 'decisions.json'):
        if d['mode'] in (kb_decisions.GEOMETRY_MODE, kb_decisions.BALL_MODE):
            hist.setdefault(d['BOX_ID'], []).append(
                {'mode': d['mode'],
                 'value': d.get('HUMAN_FINAL_CLASS') or d.get('action'),
                 'recorded_utc': d.get('recorded_utc')})
    items = []
    for b in queue_ids(which):
        l = led[b]
        rows = by_img.get(l['IMAGE'], [])
        dec = None
        if which == 'partial' and b in reps:
            dec = {'cls': reps[b]['HUMAN_FINAL_CLASS'],
                   'bbox': reps[b]['replacement_bbox_xywh']}
        elif which == 'ball' and b in balls:
            dec = {'action': balls[b]['action'],
                   'bbox': balls[b].get('ball_bbox_xywh')}
        items.append({
            'BOX_ID': b, 'IMAGE': l['IMAGE'], 'bbox': l['bbox_xywh'],
            'original_class': l['eyecu_original_class'],
            'img_w': l['img_w'], 'img_h': l['img_h'],
            'balls_in_image': sum(1 for z in rows
                                  if z['eyecu_original_class'] == 'ball'),
            'existing': [{'BOX_ID': z['BOX_ID'], 'bbox': z['bbox_xywh'],
                          'cls': z['eyecu_original_class']} for z in rows],
            'decision': dec, 'history': hist.get(b, []),
        })
    return {'queue': which, 'items': items,
            'source_log': kb_decisions.log_version(PKG / 'decisions.json')}


def append(rec):
    with LOCK:
        with open(PKG / 'decisions.json', 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec) + '\n')
            fh.flush()


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    QUEUE = 'partial'

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
            return self._send(200, json.dumps(build_state(self.QUEUE)).encode())
        if p.startswith('/img/'):
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
        route = urlparse(self.path).path
        n = int(self.headers.get('Content-Length', 0))
        d = json.loads(self.rfile.read(n) or b'{}')
        box = str(d.get('BOX_ID', ''))
        led = _ledger()
        now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

        if route == '/api/repair':
            if box not in queue_ids('partial'):
                return self._send(400, b'{"error":"not a PARTIAL_BODY_BAD_BOX '
                                       b'annotation"}')
            if d.get('HUMAN_FINAL_CLASS') not in ROLES:
                return self._send(400, b'{"error":"a repaired box must be '
                                       b'classified player, goalkeeper or referee"}')
            l = led[box]
            bbox, err = validate_bbox(d.get('replacement_bbox_xywh'),
                                      l['img_w'], l['img_h'])
            if err:
                return self._send(400, json.dumps({'error': err}).encode())
            append({
                'mode': kb_decisions.GEOMETRY_MODE, 'BOX_ID': box,
                'IMAGE': l['IMAGE'],
                'HUMAN_FINAL_CLASS': d['HUMAN_FINAL_CLASS'],
                'original_bbox_xywh': l['bbox_xywh'],
                'replacement_bbox_xywh': bbox,
                'coordinate_space': 'original image pixels',
                'img_w': l['img_w'], 'img_h': l['img_h'],
                'repairs_disposition': 'PARTIAL_BODY_BAD_BOX',
                'annotation_id_preserved': True,
                'geometry_author': 'human drawn',
                'no_model_proposal_used': True,
                'recorded_utc': now, 'author': 'human reviewer'})
            return self._send(200, json.dumps(
                {'ok': True, 'replacement_bbox_xywh': bbox,
                 'recorded_utc': now}).encode())

        if route == '/api/ball':
            if box not in queue_ids('ball'):
                return self._send(400, b'{"error":"not a BALL_WRONG_HUMAN_BOX '
                                       b'annotation"}')
            act = d.get('action')
            if act not in kb_decisions.BALL_ACTIONS:
                return self._send(400, b'{"error":"action must be one of the three '
                                       b'documented ball outcomes"}')
            l = led[box]
            rec = {'mode': kb_decisions.BALL_MODE, 'BOX_ID': box,
                   'IMAGE': l['IMAGE'], 'action': act,
                   'HUMAN_FINAL_CLASS': None,
                   'original_bbox_xywh': l['bbox_xywh'],
                   'recorded_utc': now, 'author': 'human reviewer',
                   'no_model_proposal_used': True}
            if act == 'DRAW_BALL_BOX':
                bbox, err = validate_bbox(d.get('ball_bbox_xywh'),
                                          l['img_w'], l['img_h'])
                if err:
                    return self._send(400, json.dumps({'error': err}).encode())
                rec['ball_bbox_xywh'] = bbox
                rec['geometry_author'] = 'human drawn'
                rec['annotation_id_preserved'] = True
            elif act == 'RECLASSIFY_TO_BALL':
                # geometry untouched, class becomes ball, id kept
                rec['ball_bbox_xywh'] = l['bbox_xywh']
                rec['geometry_author'] = 'original annotation, unchanged'
                rec['annotation_id_preserved'] = True
            else:
                rec['annotation_id_preserved'] = False
            append(rec)
            return self._send(200, json.dumps(
                {'ok': True, 'ball_bbox_xywh': rec.get('ball_bbox_xywh'),
                 'recorded_utc': now}).encode())
        return self._send(404, b'', 'text/plain')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument('--partial', action='store_true',
                   help='the 7 PARTIAL_BODY_BAD_BOX annotations, redrawn')
    g.add_argument('--ball', action='store_true',
                   help='the 5 BALL_WRONG_HUMAN_BOX cases, decided')
    ap.add_argument('--port', type=int, default=8742)
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args()
    which = 'partial' if args.partial else 'ball'
    H.QUEUE = which
    st = build_state(which)
    if not st['items']:
        print(f'no {which} items outstanding')
        return
    ok, problems = kb_images.preflight([t['IMAGE'] for t in st['items']])
    if not ok:
        print(f'REFUSING TO START: {len(problems)} image(s) cannot be resolved.')
        for im, why in problems:
            print(f'  {im}\n    {why}')
        sys.exit(1)
    done = sum(1 for t in st['items'] if t['decision'])
    print(f'{which} queue: {len(st["items"])} items, {done} decided, '
          f'{len(st["items"]) - done} outstanding')
    print('preflight: every image resolves')
    print('\nHUMAN GEOMETRY ONLY. No model runs and nothing is proposed.')
    url = f'http://127.0.0.1:{args.port}/'
    print(f'\nreview at {url}   Ctrl-C to stop')
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(('127.0.0.1', args.port), H).serve_forever()


if __name__ == '__main__':
    main()
