#!/usr/bin/env python
"""
M4 local bbox annotator -- for the 40 remaining sealed-TEST frames
(manchester_city_v_liverpool, youth_2) after the CVAT/Roboflow handoff was
superseded by a request to label locally instead.

Scope is intentionally narrow: draw/edit boxes for exactly the 40 frames
listed in HANDOFF_EXTERNAL_ANNOTATION/HANDOFF_MANIFEST.json, nothing else.
No model runs here, no EyeCU prediction of any kind, no proposal boxes --
every box is drawn by hand. Stdlib only (matches tools/kb_review_server.py's
own no-framework approach), single-file HTML/JS served inline.

Saves directly in the ANNOTATIONS_DRAFT.json record schema
({sequence, frame_number_1based, file, objects: [{class, bbox}], notes})
plus width/height/saved, so merge_into_draft.py in this same folder is a
near-trivial concatenation, not a format conversion.

    python experiments/records/experiment_M4/local_annotator/annotator_server.py
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
M4 = HERE.parent
HANDOFF = M4 / 'HANDOFF_EXTERNAL_ANNOTATION'
MANIFEST_PATH = HANDOFF / 'HANDOFF_MANIFEST.json'
IMAGES_ROOT = HANDOFF / 'images'
SAVE_PATH = HERE / 'LOCAL_ANNOTATIONS.json'
LOCK = threading.Lock()

# known from the extraction step earlier in M4 -- avoids an image-decode dependency
DIMS = {'manchester_city_v_liverpool': (640, 360), 'youth_2': (1920, 1080)}

CLASSES = ['player', 'goalkeeper', 'referee', 'ball']


def load_manifest():
    entries = json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))
    out = []
    for e in entries:
        w, h = DIMS[e['sequence']]
        out.append({
            'sequence': e['sequence'],
            'frame_number_1based': e['frame_number_1based'],
            'file': e['source_file_in_repo'],
            'img_url': f"/images/{e['sequence']}/{e['sequence']}_{e['frame_number_1based']:06d}.jpg",
            'width': w, 'height': h,
        })
    out.sort(key=lambda r: (r['sequence'], r['frame_number_1based']))
    return out


def load_saved():
    if SAVE_PATH.exists():
        return json.loads(SAVE_PATH.read_text(encoding='utf-8'))
    return []


def save_frame(rec):
    """Atomic upsert of one frame's record, keyed by (sequence, frame_number_1based)."""
    with LOCK:
        data = load_saved()
        key = (rec['sequence'], rec['frame_number_1based'])
        data = [r for r in data if (r['sequence'], r['frame_number_1based']) != key]
        data.append(rec)
        data.sort(key=lambda r: (r['sequence'], r['frame_number_1based']))
        tmp = SAVE_PATH.with_suffix('.tmp')
        tmp.write_text(json.dumps(data, indent=1), encoding='utf-8')
        tmp.replace(SAVE_PATH)  # atomic on both POSIX and Windows (NTFS)
    return data


PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>M4 local annotator</title>
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
</style></head><body>
<div id="top">
 <b>M4 local annotator</b>
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
 <span id="rule">BALL RULE: ALL_VISIBLE_PHYSICAL_FOOTBALLS -- label every visible physical football</span>
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
 jump.innerHTML = manifest.map((f,i)=>`<option value="${i}">${i+1}. ${f.sequence} ${f.frame_number_1based}${saved[key(f)]?' [saved]':''}</option>`).join('');
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
 const ix = (mx-panX)/scale, iy = (my-panY)/scale;   // image point under cursor
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
 jump.options[idx].text = `${idx+1}. ${f.sequence} ${f.frame_number_1based} [saved]`;
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
            rel = p[len('/images/'):]
            fp = IMAGES_ROOT / rel
            if fp.exists() and fp.is_file():
                return self._send(200, fp.read_bytes(),
                                  mimetypes.guess_type(rel)[0] or 'image/jpeg')
            return self._send(404, b'not found', 'text/plain')
        return self._send(404, b'not found', 'text/plain')

    def do_POST(self):
        if urlparse(self.path).path != '/api/save':
            return self._send(404, b'', 'text/plain')
        n = int(self.headers.get('Content-Length', 0))
        rec = json.loads(self.rfile.read(n) or b'{}')
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
    ap.add_argument('--port', type=int, default=8734)
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args()
    m = load_manifest()
    sv = load_saved()
    print(f'{len(m)} frames to annotate ({sum(1 for r in sv)} already saved)')
    url = f'http://127.0.0.1:{args.port}/'
    print(f'\nannotate at {url}   (Ctrl-C to stop; autosaves to '
          f'{SAVE_PATH.relative_to(M4.parent.parent.parent)})')
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(('127.0.0.1', args.port), H).serve_forever()


if __name__ == '__main__':
    main()
