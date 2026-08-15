#!/usr/bin/env python
"""
The last three boxes: real EyeCU targets whose role is still undetermined.

The U-resolution pass is COMPLETE -- 48 of 48 reviewed. This is not missing work.
Of those 48, three were categorised OCCLUDED_UNCLEAR: a real player, goalkeeper
or referee is in the box, the geometry is fine, and the role could not be read.

They block the gate for one reason. Leaving a real target without a role leaves
it labelled `player`, and if it is actually a goalkeeper or a referee that is a
wrong label in TRAIN -- damaging the two classes EyeCU is weakest on. So each
needs either a role or an explicit decision to drop its image from the candidate
set. Guessing 'player' because player is common is the one thing that must not
happen.

E is a first-class answer here, not a failure. Excluding one image out of 1,232
costs almost nothing; a wrong goalkeeper label costs more.

The UI states both facts at once: 48/48 U cases REVIEWED, and 3 target roles
still UNRESOLVED. Writes go to their own mode, final_target, and nothing else is
touched.
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
IMGROOT = REPO / 'EyeCU_external_data/huggingface/keremberke_football_object_detection/extracted'
MODE = 'final_target'
UNRESOLVED_CATEGORIES = ('AMBIGUOUS_TARGET', 'OCCLUDED_UNCLEAR')
LOCK = threading.Lock()

CHOICES = [
    ('P', 'player', 'the role is readable after all: outfield player', '#3ddc57'),
    ('G', 'goalkeeper', 'the role is readable after all: goalkeeper', '#ffc400'),
    ('R', 'referee', 'the role is readable after all: any on-field official', '#ff7a1a'),
    ('E', 'EXCLUDE_IMAGE', 'I genuinely cannot tell -- drop this IMAGE from the '
                           'future Keremberke training candidate set', '#ff5c5c'),
]

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>final targets</title>
<style>
 body{margin:0;background:#0e0e0e;color:#e6e6e6;font:13px system-ui,sans-serif}
 #top{display:flex;gap:14px;align-items:center;padding:8px 12px;background:#000;
      position:sticky;top:0;z-index:9;flex-wrap:wrap;border-bottom:1px solid #222}
 .pill{padding:2px 8px;border-radius:9px;background:#1c1c1c;border:1px solid #333}
 .ok{color:#3ddc57}.warn{color:#ffc400}
 #wrap{position:relative;margin:10px 320px 10px 10px;width:fit-content}
 img{display:block;max-width:calc(98vw - 320px);max-height:74vh}
 .bx{position:absolute;border:2px solid;box-sizing:border-box}
 .bx.ctx{border-color:#4a4a4a;border-width:1px}
 .tag{position:absolute;font:11px monospace;background:#000d;padding:1px 4px;
      transform:translateY(-100%);white-space:nowrap}
 #zoomwrap{position:fixed;right:8px;bottom:10px;width:300px;background:#141414;
           border:1px solid #2a2a2a;border-radius:6px;padding:6px;z-index:8}
 #zoom{width:100%;image-rendering:pixelated;border:1px solid #333}
 #panel{position:fixed;right:8px;top:52px;width:304px;background:#141414;
        border:1px solid #2a2a2a;border-radius:6px;padding:9px;z-index:8}
 .ch{display:flex;gap:7px;align-items:baseline;padding:5px;border-radius:4px;cursor:pointer}
 .ch:hover{background:#1f1f1f}
 .kk{display:inline-block;min-width:15px;text-align:center;background:#222;
     border:1px solid #3a3a3a;border-radius:3px;font:12px monospace;padding:0 4px}
 .ds{color:#8a8a8a;font-size:11px;display:block;margin-left:28px}
 #info{background:#101010;border:1px solid #242424;border-radius:4px;padding:7px;
       font:11px/1.5 monospace;color:#9a9a9a;margin-bottom:7px}
 button{background:#1e1e1e;color:#ddd;border:1px solid #3a3a3a;border-radius:4px;
        padding:4px 8px;cursor:pointer;width:48%;margin-top:6px}
</style></head><body>
<div id="top">
 <b>final target roles</b>
 <span class="pill ok">48/48 U cases REVIEWED &mdash; this pass is complete</span>
 <span class="pill warn" id="unres"></span>
 <span class="pill" id="pos"></span>
 <span style="color:#8a8a8a">a real target left without a role stays labelled
  <b>player</b> &mdash; wrong, if it is a keeper or an official</span>
</div>
<div id="wrap"><img id="im"><div id="ov"></div></div>
<div id="panel">
 <div id="info"></div>
 <div id="choices"></div>
 <button id="prev">&larr; prev</button>
 <button id="next" style="float:right">next &rarr;</button>
</div>
<div id="zoomwrap"><canvas id="zoom" height="150"></canvas>
 <div style="color:#8a8a8a;font-size:11px;margin-top:3px">box, magnified</div></div>
<script>
let S=null,i=0,dec={};
const CH=CHOICES_JSON;
async function boot(){
 S=await (await fetch('/api/state')).json(); dec=S.decisions;
 document.getElementById('choices').innerHTML=CH.map(c=>
  `<div class="ch" data-v="${c[1]}"><span class="kk" style="color:${c[3]}">${c[0]}</span>
   <span style="color:${c[3]};font-weight:600">${c[1]}</span>
   <span class="ds">${c[2]}</span></div>`).join('');
 document.querySelectorAll('.ch').forEach(e=>e.onclick=()=>decide(e.dataset.v));
 render();
}
function cur(){return S.items[i];}
function render(){
 const it=cur();
 document.getElementById('pos').textContent=`case ${i+1}/${S.items.length}`;
 const left=S.items.filter(x=>!dec[x.BOX_ID]).length;
 document.getElementById('unres').textContent=
   `${left} target role${left===1?'':'s'} still UNRESOLVED`;
 const im=document.getElementById('im'); im.onload=()=>{draw();zoom();}; im.src='/img/'+it.IMAGE;
 info();
}
function draw(){
 const it=cur(),im=document.getElementById('im'),ov=document.getElementById('ov');
 const sx=im.clientWidth/it.img_w, sy=im.clientHeight/it.img_h;
 ov.innerHTML='';
 it.context.forEach(b=>{const e=document.createElement('div');e.className='bx ctx';
  e.style.cssText=`left:${b.bbox[0]*sx}px;top:${b.bbox[1]*sy}px;width:${b.bbox[2]*sx}px;height:${b.bbox[3]*sy}px`;
  ov.appendChild(e);});
 const d=dec[it.BOX_ID], col=d?(CH.find(c=>c[1]===d)||['','','','#fff'])[3]:'#b06cff';
 const e=document.createElement('div');e.className='bx';
 e.style.cssText=`left:${it.bbox[0]*sx}px;top:${it.bbox[1]*sy}px;width:${it.bbox[2]*sx}px;
  height:${it.bbox[3]*sy}px;border-color:${col};box-shadow:0 0 0 2px #000`;
 ov.appendChild(e);
 const t=document.createElement('div');t.className='tag';
 t.style.cssText=`left:${it.bbox[0]*sx}px;top:${it.bbox[1]*sy}px;color:${col}`;
 t.textContent=d?('YOU: '+d):'UNRESOLVED TARGET';
 ov.appendChild(t);
}
function zoom(){
 const it=cur(),im=document.getElementById('im'),c=document.getElementById('zoom');
 const pad=Math.max(it.bbox[2],it.bbox[3])*1.4+10;
 const sx0=it.bbox[0]-pad, sy0=it.bbox[1]-pad;
 const sw=it.bbox[2]+2*pad, sh=it.bbox[3]+2*pad;
 c.width=c.parentElement.clientWidth-14; c.height=c.width*sh/sw;
 const g=c.getContext('2d'); g.imageSmoothingEnabled=false;
 const k=im.naturalWidth/it.img_w;
 g.drawImage(im,sx0*k,sy0*k,sw*k,sh*k,0,0,c.width,c.height);
 g.strokeStyle='#b06cff'; g.lineWidth=2;
 g.strokeRect((it.bbox[0]-sx0)/sw*c.width,(it.bbox[1]-sy0)/sh*c.height,
              it.bbox[2]/sw*c.width,it.bbox[3]/sh*c.height);
}
function info(){
 const it=cur();
 document.getElementById('info').innerHTML=
  `BOX_ID <b>${it.BOX_ID}</b><br>${it.IMAGE}<br>run ${it.run} &middot; `+
  `box ${it.bbox[2].toFixed(0)}x${it.bbox[3].toFixed(0)}px `+
  `(${(100*it.bbox[3]/it.img_h).toFixed(1)}% of frame)<br>`+
  `original class: ${it.ORIGINAL_CLASS}<br>`+
  `first-pass proposal: ${it.ORIGINAL_PROPOSAL||'ambiguous'}<br>`+
  `first-pass decision: <b style="color:#b06cff">U</b><br>`+
  `U-resolution: <b style="color:#66d9d9">${it.U_CATEGORY}</b><br>`+
  `now: <b>${dec[it.BOX_ID]||'unresolved'}</b>`;
}
async function decide(v){
 const it=cur(); dec[it.BOX_ID]=v;
 await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({mode:'final_target',BOX_ID:it.BOX_ID,IMAGE:it.IMAGE,
                       HUMAN_FINAL_CLASS:v})});
 info();draw();
 const nxt=S.items.findIndex(x=>!dec[x.BOX_ID]);
 if(nxt>=0&&nxt!==i){i=nxt;render();} else render();
}
document.getElementById('next').onclick=()=>{i=Math.min(S.items.length-1,i+1);render();};
document.getElementById('prev').onclick=()=>{i=Math.max(0,i-1);render();};
document.onkeydown=e=>{
 const hit=CH.find(c=>c[0]===e.key.toUpperCase());
 if(hit){e.preventDefault();decide(hit[1]);return;}
 if(e.key==='ArrowRight'){i=Math.min(S.items.length-1,i+1);render();}
 if(e.key==='ArrowLeft'){i=Math.max(0,i-1);render();}
};
boot();
</script></body></html>"""


def build_state():
    ledger = {r['BOX_ID']: r for r in
              json.loads((PKG / 'ledger.json').read_text(encoding='utf-8'))}
    by_img = {}
    for r in ledger.values():
        by_img.setdefault(r['IMAGE'], []).append(r)
    q = {r['BOX_ID']: r for r in
         json.loads((PKG / 'u_resolution_queue.json').read_text(encoding='utf-8'))['rows']}
    per = kb_decisions.by_mode(PKG / 'decisions.json')
    ures = {b: v for (m, b), v in per.items() if m == 'u_resolution'}

    items = []
    for box, cat in ures.items():
        if cat not in UNRESOLVED_CATEGORIES:
            continue
        r = ledger[box]
        ids = {box}
        items.append({
            'BOX_ID': box, 'IMAGE': r['IMAGE'], 'run': r['run'],
            'img_w': r['img_w'], 'img_h': r['img_h'], 'bbox': r['bbox_xywh'],
            'ORIGINAL_CLASS': r['ORIGINAL_CLASS'],
            'ORIGINAL_PROPOSAL': q[box]['ORIGINAL_PROPOSAL'],
            'FIRST_HUMAN_DECISION': q[box]['FIRST_HUMAN_DECISION'],
            'U_CATEGORY': cat,
            'context': [{'bbox': b['bbox_xywh']} for b in by_img[r['IMAGE']]
                        if b['BOX_ID'] not in ids
                        and b['eyecu_original_class'] == 'player'],
        })
    items.sort(key=lambda z: z['BOX_ID'])
    dec = {b: v for (m, b), v in per.items() if m == MODE}
    return {'items': items, 'decisions': dec,
            'u_reviewed': len(ures), 'u_total': len(q)}


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
            return self._send(200,
                              PAGE.replace('CHOICES_JSON',
                                           json.dumps(CHOICES)).encode('utf-8'),
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
        if urlparse(self.path).path != '/api/decide':
            return self._send(404, b'', 'text/plain')
        d = json.loads(self.rfile.read(int(self.headers.get('Content-Length', 0))) or b'{}')
        if d.get('mode') != MODE:
            return self._send(400, b'{"error":"this server only writes final_target"}')
        if d.get('HUMAN_FINAL_CLASS') not in {c[1] for c in CHOICES}:
            return self._send(400, b'{"error":"value not in the final-target vocabulary"}')
        d['recorded_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        d['author'] = 'human reviewer'
        with LOCK:
            with open(PKG / 'decisions.json', 'a', encoding='utf-8') as f:
                f.write(json.dumps(d) + '\n')
                f.flush()
        return self._send(200, b'{"ok":true}')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8780)
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--list', action='store_true', help='print the cases and exit')
    args = ap.parse_args()
    st = build_state()
    print(f"U-resolution: {st['u_reviewed']}/{st['u_total']} cases REVIEWED "
          f"-- complete, not missing work")
    print(f"target roles still UNRESOLVED: "
          f"{len([x for x in st['items'] if x['BOX_ID'] not in st['decisions']])}"
          f"/{len(st['items'])}")
    for x in st['items']:
        print(f"   {x['BOX_ID']:<14} {x['U_CATEGORY']:<18} "
              f"{x['bbox'][2]:.0f}x{x['bbox'][3]:.0f}px  {x['IMAGE'][:46]}  "
              f"-> {st['decisions'].get(x['BOX_ID'], 'unresolved')}")
    if args.list:
        return
    url = f'http://127.0.0.1:{args.port}/'
    print(f'\nreview at {url}   Ctrl-C to stop')
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(('127.0.0.1', args.port), H).serve_forever()


if __name__ == '__main__':
    main()
