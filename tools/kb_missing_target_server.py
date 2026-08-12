#!/usr/bin/env python
"""
Draw the boxes for targets a human saw and the dataset never annotated.

SCOPE. This is not another review of the dataset. The queue is exactly the live
MISSING_TARGET_BOX flags -- 48 targets across 39 images at the time of writing --
and nothing else is shown or asked. Every other pass is finished; re-opening them
here would invite a settled question to be answered differently.

WHY IT NEEDS ITS OWN TOOL. Every other review action edits an annotation that
already exists, so it has a BOX_ID to attach to. These do not exist at all. There
is nothing to click, nothing to reclassify, and no geometry until a person draws
it. That asymmetry is the whole reason this file exists.

WHAT IS AUTHORITY. The human-drawn rectangle. No model runs here and no proposal
is offered, because a proposed box a tired reviewer accepts is a model prediction
that became ground truth without anyone deciding it should. If assistance is ever
added it must arrive as a suggestion the human redraws or explicitly confirms,
and it must stay distinguishable in the log forever.

GEOMETRY is stored in ORIGINAL IMAGE COORDINATES. The canvas is scaled to fit the
window and can be zoomed, so viewport pixels are meaningless the moment the
window is resized; every rectangle is converted on the way in and validated
against the image dimensions recorded in the ledger.

THE LOG is append-only, as everywhere else. A redraw is a new
missing_target_resolution event that supersedes the previous one under the usual
precedence rule -- latest recorded_utc wins. Nothing is edited in place, so the
first attempt and the correction both remain visible.

    python tools/kb_missing_target_server.py

Nothing is applied. The corrected dataset is still written only by
kb_apply_review.py --apply, and only when both gates pass.
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

PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
IMGROOT = (REPO / 'EyeCU_external_data' / 'huggingface'
           / 'keremberke_football_object_detection' / 'extract')
LOCK = threading.Lock()

FLAG_MODE = 'missing_target_box'
RETRACT_MODE = 'missing_target_retraction'
RESOLVE_MODE = 'missing_target_resolution'
# A resolution is a role with a box, or an explicit decision to drop the image.
BOXED = {'player': 'boxed_player', 'goalkeeper': 'boxed_goalkeeper',
         'referee': 'boxed_referee'}
EXCLUDE = 'EXCLUDE_IMAGE'
ROLES = ('player', 'goalkeeper', 'referee')

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>missing target boxes</title>
<style>
 :root{--gk:#ffc400;--ref:#ff7a1a;--pl:#3ddc57}
 body{margin:0;background:#0e0e0e;color:#e6e6e6;font:13px system-ui,sans-serif}
 #top{display:flex;gap:14px;align-items:center;padding:7px 12px;background:#000;
      position:sticky;top:0;z-index:9;flex-wrap:wrap;border-bottom:1px solid #222}
 .pill{padding:1px 7px;border-radius:9px;background:#1c1c1c;border:1px solid #333}
 .gk{color:var(--gk)}.ref{color:var(--ref)}.pl{color:var(--pl)}
 #bar{height:4px;background:#222;width:180px;border-radius:2px;overflow:hidden}
 #bar>i{display:block;height:100%;background:#3ddc57;width:0}
 #wrap{position:relative;margin:10px 290px 10px 10px;width:fit-content;
       overflow:auto;max-height:82vh;max-width:calc(100vw - 320px)}
 #stage{position:relative;transform-origin:0 0}
 img{display:block}
 .ex{position:absolute;border:1px solid #4a4a4a;box-sizing:border-box;
     pointer-events:none}
 .ex.ball{border-color:#2f6fb0}
 .new{position:absolute;border:2px dashed #3ddc57;box-sizing:border-box;
      pointer-events:none;background:#3ddc5722}
 .saved{position:absolute;border:2px solid #3ddc57;box-sizing:border-box;
        pointer-events:none}
 .other{position:absolute;border:2px solid #666;box-sizing:border-box;
        pointer-events:none;opacity:.55}
 .tag{position:absolute;font:11px/1.3 monospace;background:#000d;padding:1px 4px;
      transform:translateY(-100%);white-space:nowrap;pointer-events:none}
 #hit{position:absolute;inset:0;cursor:crosshair}
 #panel{position:fixed;right:10px;top:52px;width:266px;background:#141414;
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
 #hint{color:#8a8a8a;font-size:11px}
</style></head><body>
<div id="top">
 <b>missing targets</b>
 <span id="pos" class="pill"></span>
 <span id="img" class="pill" style="font:11px monospace"></span>
 <div id="bar"><i></i></div>
 <span id="prog" class="pill"></span>
 <span class="pill">boxed <b id="cB" class="pl">0</b> ·
   excluded <b id="cX">0</b> · pending <b id="cP" style="color:#ff7a7a">0</b></span>
 <span id="hint">drag to draw · <span class="k">P</span> <span class="k">G</span>
  <span class="k">R</span> save with a class · <span class="k">Z</span> undo draw ·
  <span class="k">X</span> exclude image · <span class="k">N</span>/<span class="k">B</span>
  next/prev target · <span class="k">+</span>/<span class="k">-</span> zoom</span>
</div>
<div id="wrap"><div id="stage"><img id="im"><div id="ov"></div><div id="hit"></div></div></div>
<div id="panel">
 <div id="who"></div>
 <div id="state"></div>
 <button class="big" id="bP">PLAYER <span class="k">P</span></button>
 <button class="big" id="bG">GOALKEEPER <span class="k">G</span></button>
 <button class="big" id="bR">REFEREE <span class="k">R</span></button>
 <button class="big" id="bZ">UNDO DRAW <span class="k">Z</span></button>
 <button class="big" id="bX" style="border-color:#7a3a3a">EXCLUDE THIS IMAGE
  <span class="k">X</span></button>
 <div id="siblings" style="font-size:11px;margin-top:8px"></div>
 <button class="big" id="bN">NEXT TARGET <span class="k">N</span></button>
 <button class="big" id="bB">PREVIOUS <span class="k">B</span></button>
 <div id="hist" style="font-size:11px;color:#888;margin-top:8px"></div>
</div>
<script>
let S=null,i=0,draft=null,drag=null,zoom=1;
const COL={player:'#3ddc57',goalkeeper:'#ffc400',referee:'#ff7a1a'};
async function boot(){
 S=await (await fetch('/api/state')).json();
 i=S.targets.findIndex(t=>!t.resolution); if(i<0)i=0;
 render();
}
function cur(){return S.targets[i];}
function sameImage(){return S.targets.filter(t=>t.IMAGE===cur().IMAGE);}
function render(){
 const t=cur(); if(!t)return;
 document.getElementById('pos').textContent=`target ${i+1}/${S.targets.length}`;
 document.getElementById('img').textContent=t.IMAGE;
 draft=null;
 const im=document.getElementById('im');
 im.onload=()=>{applyZoom();draw();};
 im.src='/img/'+t.IMAGE;
 stats();
}
function applyZoom(){
 const st=document.getElementById('stage');
 st.style.transform='scale('+zoom+')';
 const im=document.getElementById('im');
 st.style.width=im.naturalWidth+'px'; st.style.height=im.naturalHeight+'px';
}
// image px -> displayed px is exactly the zoom factor, because the stage is
// rendered at natural size and scaled as a whole. Nothing here depends on the
// window size, so a resize cannot move a saved box.
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
 // context: every annotation already in this image, never editable here
 t.existing.forEach(b=>add('ex'+(b.cls==='ball'?' ball':''),b.bbox,null,null));
 // boxes already drawn for OTHER flags in the same image stay visible but inert
 sameImage().forEach(o=>{
  if(o.key===t.key||!o.resolution||!o.resolution.bbox_xywh)return;
  add('other',o.resolution.bbox_xywh,COL[o.resolution.role],
      'other target: '+o.resolution.role);
 });
 if(t.resolution&&t.resolution.bbox_xywh)
   add('saved',t.resolution.bbox_xywh,COL[t.resolution.role],
       'SAVED '+t.resolution.role);
 if(draft) add('new',draft,null,'drawn '+draft.map(v=>Math.round(v)).join(','));
 panel();
}
function panel(){
 const t=cur();
 document.getElementById('who').innerHTML=
  `<div style="font-size:11px;color:#8a8a8a">flag</div>
   <div style="font:11px monospace;color:#bbb;word-break:break-all">${t.key}</div>
   <div style="margin-top:4px">originally flagged as
    <b class="${t.flag_role==='goalkeeper'?'gk':t.flag_role==='referee'?'ref':'pl'}">
    ${t.flag_role}</b></div>`;
 const st=document.getElementById('state');
 if(t.resolution&&t.resolution.action==='EXCLUDE_IMAGE')
   st.innerHTML='<div class="ok">IMAGE EXCLUDED &mdash; this target needs no box. '
     +'Redrawing a box here would override that.</div>';
 else if(t.resolution)
   st.innerHTML=`<div class="ok">SAVED <b>${t.resolution.role}</b> at
     [${t.resolution.bbox_xywh.map(v=>Math.round(v)).join(', ')}].
     Drawing again records a correction; the first box stays in history.</div>`;
 else if(draft)
   st.innerHTML='<div class="ok">box drawn &mdash; now choose the class</div>';
 else if(t.flag_role==='uncertain')
   st.innerHTML='<div class="warn">The role was NOT readable when this was '
     +'flagged. Draw the box and choose P/G/R explicitly, or exclude the image. '
     +'It cannot be saved as "uncertain".</div>';
 else st.innerHTML='<div class="warn">no box yet &mdash; drag on the image</div>';
 for(const [id,role] of [['bP','player'],['bG','goalkeeper'],['bR','referee']])
   document.getElementById(id).disabled=!draft;
 document.getElementById('bZ').disabled=!draft;
 const sibs=sameImage();
 document.getElementById('siblings').innerHTML=sibs.length<2?'':
  `<div style="color:#8a8a8a">${sibs.length} targets flagged in this image &mdash;
    each is resolved separately:</div>`
  +sibs.map((o,n)=>`<div style="padding:2px 0;${o.key===cur().key
    ?'background:#1d1d1d':''}">${n+1}. ${o.flag_role}
    <b style="color:${o.resolution?'#8fe08f':'#ff7a7a'}">${o.resolution
      ?(o.resolution.action==='EXCLUDE_IMAGE'?'excluded':'boxed'):'pending'}</b>
    </div>`).join('');
 document.getElementById('hist').innerHTML=!t.history.length?'':
  '<b>history</b>'+t.history.map(h=>`<div>${(h.recorded_utc||'').slice(11,19)}
   ${h.mode.replace('missing_target_','')} ${h.value||''}</div>`).join('');
}
function stats(){
 const b=S.targets.filter(t=>t.resolution&&t.resolution.bbox_xywh).length;
 const x=S.targets.filter(t=>t.resolution&&t.resolution.action==='EXCLUDE_IMAGE').length;
 const p=S.targets.length-b-x;
 document.getElementById('cB').textContent=b;
 document.getElementById('cX').textContent=x;
 document.getElementById('cP').textContent=p;
 document.getElementById('prog').textContent=`${b+x}/${S.targets.length} resolved`;
 document.getElementById('bar').firstElementChild.style.width=
   (100*(b+x)/S.targets.length)+'%';
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
 if(draft[2]<2||draft[3]<2){draft=null;}   // a stray click is not a box
 draw();
};
async function save(role){
 const t=cur();
 if(!draft)return;
 const r=await fetch('/api/resolve',{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({missing_target_id:t.key,IMAGE:t.IMAGE,
   HUMAN_FINAL_CLASS:role,bbox_xywh:draft,flagged_role:t.flag_role})});
 const j=await r.json();
 if(!r.ok){alert(j.error||'refused');return;}
 t.resolution={role:role,bbox_xywh:j.bbox_xywh,action:'BOX_DRAWN'};
 t.history.push({mode:'missing_target_resolution',value:role,
                 recorded_utc:j.recorded_utc});
 draft=null; draw(); stats();
}
async function exclude(){
 const t=cur();
 if(!confirm('Exclude '+t.IMAGE+' from the repaired candidate set?\n\n'
   +'This resolves EVERY missing-target flag in this image. The image and its '
   +'original annotations are not deleted.'))return;
 const why=prompt('Why is this image being excluded?');
 if(!why||!why.trim())return;
 const r=await fetch('/api/resolve',{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({missing_target_id:t.key,IMAGE:t.IMAGE,
   HUMAN_FINAL_CLASS:'EXCLUDE_IMAGE',reason:why.trim()})});
 const j=await r.json();
 if(!r.ok){alert(j.error||'refused');return;}
 (j.applied_to||[]).forEach(k=>{
  const o=S.targets.find(z=>z.key===k);
  if(o){o.resolution={role:null,bbox_xywh:null,action:'EXCLUDE_IMAGE'};
        o.history.push({mode:'missing_target_resolution',value:'EXCLUDE_IMAGE',
                        recorded_utc:j.recorded_utc});}
 });
 draw(); stats();
}
function go(n){i=Math.min(Math.max(0,n),S.targets.length-1);render();}
document.getElementById('bP').onclick=()=>save('player');
document.getElementById('bG').onclick=()=>save('goalkeeper');
document.getElementById('bR').onclick=()=>save('referee');
document.getElementById('bZ').onclick=()=>{draft=null;draw();};
document.getElementById('bX').onclick=exclude;
document.getElementById('bN').onclick=()=>go(i+1);
document.getElementById('bB').onclick=()=>go(i-1);
document.onkeydown=e=>{
 const k=e.key.toLowerCase();
 if(k==='p')save('player'); else if(k==='g')save('goalkeeper');
 else if(k==='r')save('referee');
 else if(k==='z'){draft=null;draw();}
 else if(k==='x'){e.preventDefault();exclude();}
 else if(k==='n'){e.preventDefault();go(i+1);}
 else if(k==='b'){go(i-1);}
 else if(e.key==='+'||e.key==='='){zoom=Math.min(zoom*1.25,8);applyZoom();}
 else if(e.key==='-'){zoom=Math.max(zoom/1.25,0.25);applyZoom();}
 else if(k==='0'){zoom=1;applyZoom();}
};
boot();
</script></body></html>"""


def collect():
    """Live flags, their history and their current resolution. Folded from the log.

    Nothing is read from missing_target_queue.json. That file is a report; it was
    once stale by 51 flags, and a tool that trusted it would have shown an empty
    queue and reported the work finished.
    """
    rows = kb_decisions.read_log(PKG / 'decisions.json')
    flags, retr, hist = {}, set(), {}
    res = {}
    for r in rows:
        b = r['BOX_ID']
        if r['mode'] in (FLAG_MODE, RETRACT_MODE, RESOLVE_MODE):
            hist.setdefault(b, []).append(
                {'mode': r['mode'], 'value': r.get('HUMAN_FINAL_CLASS'),
                 'recorded_utc': r.get('recorded_utc')})
        if r['mode'] == FLAG_MODE:
            flags[b] = r
        elif r['mode'] == RETRACT_MODE:
            retr.add(b)
        elif r['mode'] == RESOLVE_MODE:
            res[b] = r                       # later line supersedes; append-only
    live = {b: f for b, f in flags.items() if b not in retr}
    return live, res, hist


def build_state():
    led = json.loads((PKG / 'ledger.json').read_text(encoding='utf-8'))
    by_img = {}
    for r in led:
        by_img.setdefault(r['IMAGE'], []).append(r)
    live, res, hist = collect()
    targets = []
    for key, f in sorted(live.items(),
                         key=lambda kv: (kv[1].get('IMAGE') or '',
                                         kv[1].get('recorded_utc') or '')):
        img = f.get('IMAGE')
        rows = by_img.get(img, [])
        r = res.get(key)
        resolution = None
        if r:
            v = r.get('HUMAN_FINAL_CLASS')
            if v == EXCLUDE:
                resolution = {'role': None, 'bbox_xywh': None,
                              'action': 'EXCLUDE_IMAGE'}
            else:
                resolution = {'role': r.get('role') or v,
                              'bbox_xywh': r.get('bbox_xywh'),
                              'action': 'BOX_DRAWN'}
        targets.append({
            'key': key, 'IMAGE': img, 'run': f.get('run'),
            'flag_role': f['HUMAN_FINAL_CLASS'],
            'flagged_utc': f.get('recorded_utc'),
            'img_w': rows[0]['img_w'] if rows else None,
            'img_h': rows[0]['img_h'] if rows else None,
            'existing': [{'bbox': z['bbox_xywh'], 'cls': z['eyecu_original_class']}
                         for z in rows],
            'resolution': resolution,
            'history': hist.get(key, []),
        })
    return {'targets': targets,
            'images': len({t['IMAGE'] for t in targets}),
            'source_log': kb_decisions.log_version(PKG / 'decisions.json')}


def validate_bbox(b, w, h):
    """Reject anything that is not a usable rectangle inside the image.

    Clamping silently would turn a mis-drag into a plausible-looking annotation,
    so an out-of-bounds box is refused and the human redraws. Only the sub-pixel
    overshoot from the edge of a scaled canvas is clamped, because that is a
    rendering artefact rather than a mistake.
    """
    if not (isinstance(b, list) and len(b) == 4):
        return None, 'bbox must be [x, y, w, h]'
    try:
        x, y, bw, bh = (float(v) for v in b)
    except (TypeError, ValueError):
        return None, 'bbox values must be numbers'
    if bw <= 0 or bh <= 0:
        return None, 'width and height must be positive'
    if not w or not h:
        return None, 'image dimensions unknown for this target'
    EPS = 1.0
    if x < -EPS or y < -EPS or x + bw > w + EPS or y + bh > h + EPS:
        return None, (f'box [{x:.0f},{y:.0f},{bw:.0f},{bh:.0f}] falls outside '
                      f'the {w}x{h} image')
    x = min(max(x, 0.0), float(w))
    y = min(max(y, 0.0), float(h))
    bw = min(bw, float(w) - x)
    bh = min(bh, float(h) - y)
    if bw < 1 or bh < 1:
        return None, 'box is smaller than a pixel'
    return [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)], None


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'

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
            return self._send(200, json.dumps(build_state()).encode('utf-8'))
        if p.startswith('/img/'):
            split, name = p[len('/img/'):].split('/', 1)
            for c in (IMGROOT / split).rglob(name):
                return self._send(200, c.read_bytes(),
                                  mimetypes.guess_type(name)[0] or 'image/jpeg')
            return self._send(404, b'not found', 'text/plain')
        return self._send(404, b'not found', 'text/plain')

    def do_POST(self):
        if urlparse(self.path).path != '/api/resolve':
            return self._send(404, b'', 'text/plain')
        n = int(self.headers.get('Content-Length', 0))
        d = json.loads(self.rfile.read(n) or b'{}')
        key = str(d.get('missing_target_id', ''))
        live, res, _ = collect()
        if key not in live:
            return self._send(400, b'{"error":"unknown or retracted flag; a '
                                   b'resolution must name one live flag"}')
        val = d.get('HUMAN_FINAL_CLASS')
        img = live[key].get('IMAGE')
        now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())

        if val == EXCLUDE:
            if not str(d.get('reason', '')).strip():
                return self._send(400, b'{"error":"an exclusion needs a reason"}')
            # Documented exclusion rule: excluding an image resolves EVERY live
            # flag in THAT image and nothing else. Written as one event per flag
            # so each obligation is discharged explicitly and the fold stays
            # per-flag -- an image-wide event would make "is this flag resolved"
            # depend on parsing a different entity.
            keys = [k for k, f in live.items() if f.get('IMAGE') == img]
            with LOCK:
                with open(PKG / 'decisions.json', 'a', encoding='utf-8') as fh:
                    for k in keys:
                        fh.write(json.dumps({
                            'mode': RESOLVE_MODE, 'BOX_ID': k,
                            'missing_target_id': k, 'IMAGE': img,
                            'HUMAN_FINAL_CLASS': EXCLUDE,
                            'action': 'EXCLUDE_IMAGE_FROM_CANDIDATE_SET',
                            'reason': d['reason'].strip(),
                            'resolves_flags_in_image': keys,
                            'recorded_utc': now, 'author': 'human reviewer'}) + '\n')
                    fh.flush()
            return self._send(200, json.dumps(
                {'ok': True, 'applied_to': keys, 'recorded_utc': now}).encode())

        if val not in ROLES:
            return self._send(400, b'{"error":"a drawn target must be classified '
                                   b'player, goalkeeper or referee -- not '
                                   b'uncertain"}')
        led = json.loads((PKG / 'ledger.json').read_text(encoding='utf-8'))
        dims = next(((r['img_w'], r['img_h']) for r in led if r['IMAGE'] == img),
                    (None, None))
        bbox, err = validate_bbox(d.get('bbox_xywh'), *dims)
        if err:
            return self._send(400, json.dumps({'error': err}).encode())
        rec = {
            'mode': RESOLVE_MODE, 'BOX_ID': key, 'missing_target_id': key,
            'IMAGE': img, 'run': live[key].get('run'),
            'HUMAN_FINAL_CLASS': BOXED[val], 'role': val,
            'bbox_xywh': bbox,
            'coordinate_space': 'original image pixels',
            'img_w': dims[0], 'img_h': dims[1],
            'flagged_role': live[key]['HUMAN_FINAL_CLASS'],
            'geometry_author': 'human drawn',
            'no_model_proposal_used': True,
            'supersedes': (res[key].get('recorded_utc') if key in res else None),
            'recorded_utc': now, 'author': 'human reviewer',
        }
        with LOCK:
            with open(PKG / 'decisions.json', 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(rec) + '\n')
                fh.flush()
        return self._send(200, json.dumps(
            {'ok': True, 'bbox_xywh': bbox, 'recorded_utc': now}).encode())


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8741)
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args()
    st = build_state()
    done = sum(1 for t in st['targets'] if t['resolution'])
    if not st['targets']:
        print('no live missing-target flags; nothing to do')
        return
    print(f"{len(st['targets'])} live targets across {st['images']} images")
    print(f'already resolved: {done}  pending: {len(st["targets"]) - done}')
    print('\nHUMAN-DRAWN GEOMETRY ONLY. No model runs here and no box is proposed.')
    url = f'http://127.0.0.1:{args.port}/'
    print(f'\ndraw at {url}   Ctrl-C to stop')
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(('127.0.0.1', args.port), H).serve_forever()


if __name__ == '__main__':
    main()
