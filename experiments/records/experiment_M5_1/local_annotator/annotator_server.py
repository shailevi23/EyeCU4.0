#!/usr/bin/env python
"""
M5.1 blinded Como re-annotator -- the 20 frozen com_2-0_sassuolo TEST frames,
re-annotated from scratch to test the M5.1 process-mismatch hypothesis
against the original in-session grid-read GT.

BLINDING IS THE ENTIRE POINT OF THIS TOOL. It intentionally does not import,
read, or serve any of: TEST_DETECTION_ANNOTATIONS.json (the original Como
GT), RAW_PREDICTIONS.json (detector output), DETECTION_METRICS.json (IoU/
metric results), or ANNOTATIONS_DRAFT.json. None of those files are even
opened by this script. The only inputs are the 20 frozen source images
(read-only, unmodified, same files already used for the original GT) and a
fresh, empty save file.

Otherwise the same minimal stdlib-only browser bbox editor as
experiments/records/experiment_M4/local_annotator/annotator_server.py:
draw/move/resize/delete boxes, 4 classes, autosave every edit, next/prev
navigation, no model inference anywhere in this tool.

    python experiments/records/experiment_M5_1/local_annotator/annotator_server.py
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

HERE = Path(__file__).resolve().parent
M5_1 = HERE.parent
REPO = M5_1.parent.parent.parent
IMAGES_ROOT = REPO / 'experiments' / 'records' / 'experiment_M4' / 'candidates' / 'como_2-0_sassuolo'
SAVE_PATH = HERE / 'COMO_BLIND_ANNOTATIONS.json'
LOCK = threading.Lock()

SEQUENCE = 'como_2-0_sassuolo'
FRAME_NUMBERS = [38, 113, 188, 263, 338, 413, 488, 563, 638, 713,
                 789, 864, 939, 1014, 1089, 1164, 1239, 1314, 1389, 1464]
WIDTH, HEIGHT = 640, 360  # known, fixed -- same as recorded for this sequence throughout M4/M5
CLASSES = ['player', 'goalkeeper', 'referee', 'ball']


def load_manifest():
    out = []
    for n in FRAME_NUMBERS:
        fname = f'{SEQUENCE}_{n:06d}.jpg'
        fp = IMAGES_ROOT / fname
        if not fp.exists():
            raise FileNotFoundError(f'expected frozen frame missing: {fp}')
        out.append({'sequence': SEQUENCE, 'frame_number_1based': n,
                    'file': f'experiments/records/experiment_M4/candidates/{SEQUENCE}/{fname}',
                    'img_url': f'/images/{fname}', 'width': WIDTH, 'height': HEIGHT})
    return out


def load_saved():
    if SAVE_PATH.exists():
        return json.loads(SAVE_PATH.read_text(encoding='utf-8'))
    return []


def save_frame(rec):
    with LOCK:
        data = load_saved()
        data = [r for r in data if r['frame_number_1based'] != rec['frame_number_1based']]
        data.append(rec)
        data.sort(key=lambda r: r['frame_number_1based'])
        tmp = SAVE_PATH.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=1), encoding='utf-8')
        tmp.replace(SAVE_PATH)
    return data


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>M5.1 blind Como re-annotation</title>
<style>
 :root{--player:#3ddc57;--goalkeeper:#ffc400;--referee:#ff7a1a;--ball:#4aa3ff}
 body{margin:0;background:#111;color:#ddd;font:13px system-ui,sans-serif}
 #top{display:flex;gap:16px;align-items:center;padding:6px 10px;background:#000;
      position:sticky;top:0;z-index:5;flex-wrap:wrap}
 .k{background:#222;border:1px solid #444;border-radius:3px;padding:1px 5px;font-family:monospace}
 .sw{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:3px}
 select,button{background:#222;color:#ddd;border:1px solid #444;padding:3px 6px}
 #wrap{position:relative;overflow:hidden;margin:8px;height:calc(100vh - 90px);
       background:#000;cursor:crosshair}
 #stage{position:absolute;left:0;top:0;transform-origin:0 0}
 #stage img{display:block;user-select:none;-webkit-user-drag:none}
 .bx{position:absolute;border:2px solid;box-sizing:border-box}
 .bx.sel{border-width:3px;box-shadow:0 0 0 1px #fff}
 .tag{position:absolute;top:-16px;left:-2px;font:11px monospace;padding:0 3px;
      white-space:nowrap;color:#000}
 #rule{color:#f66;font-weight:bold}
 #saved{color:#6f6}
 #blind{color:#0cf;font-weight:bold}
</style></head><body>
<div id="top">
 <b>M5.1 BLIND Como re-annotation</b>
 <span id="blind">no old GT, no predictions, no scores shown</span>
 <span id="pos"></span>
 <span id="label"></span>
 <select id="jump"></select>
 <button id="prev">&larr; Prev</button>
 <button id="next">Next &rarr;</button>
 <span id="saved"></span>
 <span>
  <span class="sw" style="background:var(--player)"></span><span class="k">1</span>player
  <span class="sw" style="background:var(--goalkeeper)"></span><span class="k">2</span>goalkeeper
  <span class="sw" style="background:var(--referee)"></span><span class="k">3</span>referee
  <span class="sw" style="background:var(--ball)"></span><span class="k">4</span>ball
  <span class="k">Del</span>delete <span class="k">wheel</span>zoom <span class="k">right-drag</span>pan
 </span>
 <span id="rule">BALL RULE: ALL_VISIBLE_PHYSICAL_FOOTBALLS -- label every visible physical football. Tight visible-object boxes.</span>
</div>
<div id="wrap"><div id="stage"><img id="im"><div id="ov"></div></div></div>
<script>
const COL={player:'#3ddc57',goalkeeper:'#ffc400',referee:'#ff7a1a',ball:'#4aa3ff'};
let manifest=[], saved={}, idx=0, boxes=[], curClass='player', selId=null;
let scale=1, panX=0, panY=0, drag=null;

function key(f){return f.sequence+'#'+f.frame_number_1based}

async function boot(){
 manifest = await (await fetch('/api/manifest')).json();
 const sv = await (await fetch('/api/annotations')).json();
 for(const r of sv) saved[key(r)] = r;
 const jump = document.getElementById('jump');
 jump.innerHTML = manifest.map((f,i)=>`<option value="${i}">${i+1}. frame ${f.frame_number_1based}${saved[key(f)]?' [saved]':''}</option>`).join('');
 jump.onchange=()=>{idx=+jump.value; render()};
 document.getElementById('prev').onclick=()=>nav(-1);
 document.getElementById('next').onclick=()=>nav(1);
 render();
}

function nav(d){ idx=Math.max(0,Math.min(manifest.length-1, idx+d)); render(); }

function render(){
 const f = manifest[idx];
 document.getElementById('pos').textContent = `${idx+1} / ${manifest.length}`;
 document.getElementById('label').textContent = `${f.sequence}  frame ${f.frame_number_1based}`;
 document.getElementById('jump').value = idx;
 const rec = saved[key(f)];
 boxes = rec ? rec.objects.map(o=>({class:o.class, bbox:o.bbox.slice()})) : [];
 selId = null;
 const im = document.getElementById('im');
 im.src = f.img_url;
 im.onload = ()=>{ fit(f); paint(); };
 updateSavedLabel();
}

function fit(f){
 const wrap = document.getElementById('wrap');
 scale = Math.min((wrap.clientWidth-4)/f.width, (wrap.clientHeight-4)/f.height, 3);
 panX=0; panY=0;
 applyTransform();
}
function applyTransform(){
 document.getElementById('stage').style.transform = `translate(${panX}px,${panY}px) scale(${scale})`;
}

function paint(){
 const ov = document.getElementById('ov');
 ov.innerHTML='';
 boxes.forEach((b,i)=>{
  const [x1,y1,x2,y2] = b.bbox;
  const d = document.createElement('div');
  d.className = 'bx'+(i===selId?' sel':'');
  d.style.left=x1+'px'; d.style.top=y1+'px';
  d.style.width=(x2-x1)+'px'; d.style.height=(y2-y1)+'px';
  d.style.borderColor = COL[b.class];
  const t = document.createElement('div');
  t.className='tag'; t.textContent=b.class; t.style.background=COL[b.class];
  d.appendChild(t);
  ov.appendChild(d);
 });
}

function imgPoint(evt){
 const st = document.getElementById('stage').getBoundingClientRect();
 return [(evt.clientX-st.left)/scale, (evt.clientY-st.top)/scale];
}

function hitTest(x,y){
 for(let i=boxes.length-1;i>=0;i--){
  const [x1,y1,x2,y2]=boxes[i].bbox;
  const near = 6/scale;
  const onEdge = (Math.abs(x-x1)<near||Math.abs(x-x2)<near||Math.abs(y-y1)<near||Math.abs(y-y2)<near)
                 && x>x1-near && x<x2+near && y>y1-near && y<y2+near;
  if(x>=x1&&x<=x2&&y>=y1&&y<=y2) return {i, mode: onEdge?'resize':'move'};
 }
 return null;
}

const wrap = document.getElementById('wrap');
wrap.addEventListener('contextmenu', e=>e.preventDefault());
wrap.addEventListener('wheel', e=>{
 e.preventDefault();
 const wr = wrap.getBoundingClientRect();
 const mx = e.clientX-wr.left, my = e.clientY-wr.top;
 const ix = (mx-panX)/scale, iy = (my-panY)/scale;
 scale = Math.max(0.1, Math.min(8, scale * (e.deltaY<0?1.1:0.9)));
 panX = mx - ix*scale;
 panY = my - iy*scale;
 applyTransform();
}, {passive:false});

wrap.addEventListener('mousedown', e=>{
 if(e.button===2){ drag={mode:'pan', sx:e.clientX, sy:e.clientY, ox:panX, oy:panY}; return; }
 if(e.button!==0) return;
 const [x,y] = imgPoint(e);
 const hit = hitTest(x,y);
 if(hit){ selId = hit.i; paint();
  drag = {mode:hit.mode, i:hit.i, sx:x, sy:y, orig:boxes[hit.i].bbox.slice()};
 } else {
  selId = null;
  drag = {mode:'new', sx:x, sy:y};
 }
});
window.addEventListener('mousemove', e=>{
 if(!drag) return;
 if(drag.mode==='pan'){ panX=drag.ox+(e.clientX-drag.sx); panY=drag.oy+(e.clientY-drag.sy); applyTransform(); return; }
 const [x,y] = imgPoint(e);
 if(drag.mode==='new'){
  const x1=Math.min(drag.sx,x), x2=Math.max(drag.sx,x), y1=Math.min(drag.sy,y), y2=Math.max(drag.sy,y);
  if(boxes.length && boxes[boxes.length-1]._draft){ boxes[boxes.length-1].bbox=[x1,y1,x2,y2]; }
  else boxes.push({class:curClass, bbox:[x1,y1,x2,y2], _draft:true});
  paint();
 } else if(drag.mode==='move'){
  const dx=x-drag.sx, dy=y-drag.sy;
  const [ox1,oy1,ox2,oy2]=drag.orig;
  boxes[drag.i].bbox=[ox1+dx,oy1+dy,ox2+dx,oy2+dy];
  paint();
 } else if(drag.mode==='resize'){
  const [ox1,oy1,ox2,oy2]=drag.orig;
  const nx = Math.abs(x-ox1) < Math.abs(x-ox2) ? [x,ox2] : [ox1,x];
  const ny = Math.abs(y-oy1) < Math.abs(y-oy2) ? [y,oy2] : [oy1,y];
  boxes[drag.i].bbox=[Math.min(nx[0],nx[1]),Math.min(ny[0],ny[1]),Math.max(nx[0],nx[1]),Math.max(ny[0],ny[1])];
  paint();
 }
});
window.addEventListener('mouseup', ()=>{
 if(drag && drag.mode==='new'){
  const b = boxes[boxes.length-1];
  if(b && b._draft){
   delete b._draft;
   if(Math.abs(b.bbox[2]-b.bbox[0])<3 || Math.abs(b.bbox[3]-b.bbox[1])<3) boxes.pop();
   else selId = boxes.length-1;
  }
 }
 const didEdit = drag && drag.mode!=='pan';
 drag=null;
 if(didEdit) saveCurrent();
 paint();
});

window.addEventListener('keydown', e=>{
 if(e.key==='ArrowLeft'){ nav(-1); return; }
 if(e.key==='ArrowRight'){ nav(1); return; }
 if(e.key==='Delete'||e.key==='Backspace'){
  if(selId!==null){ boxes.splice(selId,1); selId=null; paint(); saveCurrent(); }
  return;
 }
 if(['1','2','3','4'].includes(e.key)){
  const cls = CLASSES_JS[+e.key-1];
  if(selId!==null){ boxes[selId].class=cls; paint(); saveCurrent(); }
  else curClass = cls;
 }
});
const CLASSES_JS = ['player','goalkeeper','referee','ball'];

async function saveCurrent(){
 const f = manifest[idx];
 const objects = boxes.filter(b=>!b._draft).map(b=>({class:b.class, bbox:b.bbox.map(v=>Math.round(v))}));
 const rec = {sequence:f.sequence, frame_number_1based:f.frame_number_1based, file:f.file,
              width:f.width, height:f.height, objects, saved:true, notes:''};
 saved[key(f)] = rec;
 updateSavedLabel();
 await fetch('/api/save', {method:'POST', body: JSON.stringify(rec)});
 const jump = document.getElementById('jump');
 jump.options[idx].text = `${idx+1}. frame ${f.frame_number_1based} [saved]`;
}
function updateSavedLabel(){
 const n = Object.keys(saved).length;
 document.getElementById('saved').textContent = `${n} / ${manifest.length} saved`;
}

boot();
</script>
</body></html>"""


class H(BaseHTTPRequestHandler):
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
        if p == '/api/manifest':
            return self._send(200, json.dumps(load_manifest()).encode('utf-8'))
        if p == '/api/annotations':
            return self._send(200, json.dumps(load_saved()).encode('utf-8'))
        if p.startswith('/images/'):
            fname = p[len('/images/'):]
            fp = IMAGES_ROOT / fname
            if fp.exists() and fp.is_file() and fname in {f'{SEQUENCE}_{n:06d}.jpg' for n in FRAME_NUMBERS}:
                return self._send(200, fp.read_bytes(),
                                  mimetypes.guess_type(fname)[0] or 'image/jpeg')
            return self._send(404, b'not found', 'text/plain')
        return self._send(404, b'not found', 'text/plain')

    def do_POST(self):
        if urlparse(self.path).path != '/api/save':
            return self._send(404, b'', 'text/plain')
        n = int(self.headers.get('Content-Length', 0))
        rec = json.loads(self.rfile.read(n) or b'{}')
        if rec.get('sequence') != SEQUENCE or rec.get('frame_number_1based') not in FRAME_NUMBERS:
            return self._send(400, json.dumps({'ok': False, 'error': 'not in the frozen 20-frame queue'}).encode())
        for o in rec.get('objects', []):
            if o.get('class') not in CLASSES:
                return self._send(400, json.dumps({'ok': False, 'error': 'bad class'}).encode())
        rec['saved_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        save_frame(rec)
        return self._send(200, b'{"ok":true}')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8735)
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args()
    m = load_manifest()
    sv = load_saved()
    print(f'{len(m)} Como frames to BLIND-re-annotate ({len(sv)} already saved)')
    print('no old GT, no predictions, no scores are loaded or served by this tool')
    url = f'http://127.0.0.1:{args.port}/'
    print(f'\nannotate at {url}   (Ctrl-C to stop; autosaves to '
          f'{SAVE_PATH.relative_to(REPO)})')
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(('127.0.0.1', args.port), H).serve_forever()


if __name__ == '__main__':
    main()
