#!/usr/bin/env python
"""
Dedicated U-resolution server. The 48 boxes the first pass could not settle.

WHY THIS IS A SEPARATE TOOL. The multi-mode server keyed its progress by BOX_ID
across every mode, so the first-pass `candidates: uncertain` answer on each of
these 48 boxes counted as a completed second-pass resolution: the header read
48/48 with zero u_resolution decisions on record, and a reviewer would have seen
a finished pass and skipped it. It also never displayed the six categories, and
its `B` key meant "previous image" while B must mean BALL_WRONG_HUMAN_BOX here.

Fixing that inside the multi-mode server would have meant special-casing the very
thing that was wrong. This tool is scoped to one mode instead: it reads only
u_resolution decisions, writes only u_resolution decisions, and refuses anything
else, so a first-pass answer can neither be counted nor overwritten.

An original U is a QUESTION, not an ANSWER. It is shown as context on every box
and is never treated as progress.
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
PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
IMGROOT = REPO / 'EyeCU_external_data/huggingface/keremberke_football_object_detection/extracted'
MODE = 'u_resolution'
LOCK = threading.Lock()

CHOICES = [
    ('P', 'player', 'resolved: outfield player', '#3ddc57'),
    ('G', 'goalkeeper', 'resolved: goalkeeper', '#ffc400'),
    ('R', 'referee', 'resolved: any on-field official', '#ff7a1a'),
    ('A', 'AMBIGUOUS_TARGET', 'real EyeCU target, role not distinguishable', '#7aa7ff'),
    ('O', 'OCCLUDED_UNCLEAR', 'substantially occluded, role undeterminable', '#66d9d9'),
    ('N', 'NON_TARGET_HUMAN', 'coach / bench / ball person / medical / staff', '#c08adf'),
    ('B', 'BALL_WRONG_HUMAN_BOX', 'this "human" box is actually on the ball', '#ff5c5c'),
    ('F', 'FALSE_POSITIVE', 'no relevant human in the box at all', '#9a9a9a'),
    ('X', 'PARTIAL_BODY_BAD_BOX', 'real person, but the box is a fragment', '#ff9ad5'),
]

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>U resolution</title>
<style>
 body{margin:0;background:#0e0e0e;color:#e6e6e6;font:13px system-ui,sans-serif}
 #top{display:flex;gap:14px;align-items:center;padding:7px 12px;background:#000;
      position:sticky;top:0;z-index:9;flex-wrap:wrap;border-bottom:1px solid #222}
 .pill{padding:1px 7px;border-radius:9px;background:#1c1c1c;border:1px solid #333}
 #bar{height:4px;background:#222;width:160px;border-radius:2px;overflow:hidden}
 #bar>i{display:block;height:100%;background:#3ddc57;width:0}
 #wrap{position:relative;margin:10px 300px 10px 10px;width:fit-content}
 img{display:block;max-width:calc(98vw - 300px);max-height:76vh}
 .bx{position:absolute;border:2px solid;box-sizing:border-box;cursor:pointer}
 .bx.ctx{border-color:#4a4a4a;border-width:1px;cursor:default}
 .tag{position:absolute;font:11px monospace;background:#000d;padding:1px 4px;
      transform:translateY(-100%);white-space:nowrap;pointer-events:none}
 #panel{position:fixed;right:8px;top:50px;width:284px;background:#141414;
        border:1px solid #2a2a2a;border-radius:6px;padding:9px;z-index:8;
        max-height:82vh;overflow:auto}
 .ch{display:flex;gap:7px;align-items:baseline;padding:4px 5px;border-radius:4px;cursor:pointer}
 .ch:hover{background:#1f1f1f}
 .kk{display:inline-block;min-width:15px;text-align:center;background:#222;
     border:1px solid #3a3a3a;border-radius:3px;font:12px monospace;padding:0 4px}
 .nm{font-weight:600}.ds{color:#8a8a8a;font-size:11px;display:block;margin-left:28px}
 h4{margin:8px 0 3px;font-size:11px;color:#8a8a8a;text-transform:uppercase;letter-spacing:.6px}
 button{background:#1e1e1e;color:#ddd;border:1px solid #3a3a3a;border-radius:4px;
        padding:4px 8px;cursor:pointer;width:100%;margin-top:5px}
 #ctxbox{background:#101010;border:1px solid #242424;border-radius:4px;padding:6px;
         font:11px monospace;color:#9a9a9a;margin-bottom:6px}
</style></head><body>
<div id="top">
 <b>U resolution</b>
 <span class="pill" id="pos"></span><span class="pill" id="run"></span>
 <span class="pill" id="prog"></span><div id="bar"><i></i></div>
 <span class="pill" id="remain"></span>
 <span style="color:#8a8a8a">an original <b>U</b> is the QUESTION, not an answer</span>
</div>
<div id="wrap"><img id="im"><div id="ov"></div></div>
<div id="panel">
 <div id="ctxbox"></div>
 <h4>resolved role</h4><div id="roles"></div>
 <h4>u categories</h4><div id="cats"></div>
 <button id="next">NEXT UNRESOLVED &nbsp;<span class="kk">N</span></button>
 <button id="prev">PREVIOUS &nbsp;<span class="kk">&larr;</span></button>
</div>
<script>
let S=null,i=0,sel=null,dec={};
const CH=CHOICES_JSON;
async function boot(){
 S=await (await fetch('/api/state')).json(); dec=S.decisions;
 const mk=(list,el)=>document.getElementById(el).innerHTML=list.map(c=>
  `<div class="ch" data-v="${c[1]}"><span class="kk" style="color:${c[3]}">${c[0]}</span>
   <span class="nm" style="color:${c[3]}">${c[1]}</span><span class="ds">${c[2]}</span></div>`).join('');
 mk(CH.slice(0,3),'roles'); mk(CH.slice(3),'cats');
 document.querySelectorAll('.ch').forEach(e=>e.onclick=()=>decide(e.dataset.v));
 i=S.items.findIndex(x=>x.boxes.some(b=>!dec[b.BOX_ID])); if(i<0)i=0;
 render();
}
function cur(){return S.items[i];}
function render(){
 const it=cur(); if(!it)return;
 document.getElementById('pos').textContent=`image ${i+1}/${S.items.length}`;
 document.getElementById('run').textContent='run '+it.run;
 const im=document.getElementById('im'); im.onload=draw; im.src='/img/'+it.IMAGE;
 sel=(it.boxes.find(b=>!dec[b.BOX_ID])||it.boxes[0]).BOX_ID;
 stats();
}
function draw(){
 const it=cur(),im=document.getElementById('im'),ov=document.getElementById('ov');
 const sx=im.clientWidth/it.img_w, sy=im.clientHeight/it.img_h;
 ov.innerHTML='';
 it.context.forEach(b=>{const e=document.createElement('div');e.className='bx ctx';
  e.style.cssText=`left:${b.bbox[0]*sx}px;top:${b.bbox[1]*sy}px;width:${b.bbox[2]*sx}px;height:${b.bbox[3]*sy}px`;
  ov.appendChild(e);});
 it.boxes.forEach((b,n)=>{
  const d=dec[b.BOX_ID];
  const col=d?(CH.find(c=>c[1]===d)||['','','','#fff'])[3]:'#b06cff';
  const e=document.createElement('div');e.className='bx';
  e.style.cssText=`left:${b.bbox[0]*sx}px;top:${b.bbox[1]*sy}px;width:${b.bbox[2]*sx}px;
   height:${b.bbox[3]*sy}px;border-color:${col};${b.BOX_ID===sel?'box-shadow:0 0 0 2px #fff inset;':''}`;
  e.onclick=()=>{sel=b.BOX_ID;draw();ctx();};
  ov.appendChild(e);
  const t=document.createElement('div');t.className='tag';
  t.style.cssText=`left:${b.bbox[0]*sx}px;top:${b.bbox[1]*sy}px;color:${col}`;
  t.textContent=`${n+1} ${d?('YOU: '+d):'UNRESOLVED'}`;
  ov.appendChild(t);
 });
 ctx();
}
function ctx(){
 const it=cur(),b=it.boxes.find(z=>z.BOX_ID===sel)||it.boxes[0];
 document.getElementById('ctxbox').innerHTML=
  `box ${b.BOX_ID}<br>original class: ${b.ORIGINAL_CLASS}<br>`+
  `first-pass proposal: ${b.ORIGINAL_PROPOSAL||'ambiguous'}<br>`+
  `first-pass decision: <b style="color:#b06cff">U</b> (a question, not an answer)<br>`+
  `now: <b>${dec[b.BOX_ID]||'unresolved'}</b>`;
}
function stats(){
 const done=Object.keys(dec).length;
 document.getElementById('prog').textContent=`${done}/${S.total_boxes} resolved`;
 document.getElementById('remain').textContent=`remaining ${S.total_boxes-done}`;
 document.querySelector('#bar>i').style.width=(100*done/S.total_boxes)+'%';
}
async function decide(v){
 if(!sel)return; dec[sel]=v;
 await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({mode:'u_resolution',BOX_ID:sel,IMAGE:cur().IMAGE,
                       HUMAN_FINAL_CLASS:v})});
 const nxt=cur().boxes.find(b=>!dec[b.BOX_ID]);
 if(nxt){sel=nxt.BOX_ID;draw();stats();} else {stats();draw();}
}
function nextUnresolved(){
 for(let k=i+1;k<S.items.length;k++) if(S.items[k].boxes.some(b=>!dec[b.BOX_ID])){i=k;render();return;}
 i=Math.min(i+1,S.items.length-1);render();
}
document.getElementById('next').onclick=nextUnresolved;
document.getElementById('prev').onclick=()=>{i=Math.max(0,i-1);render();};
document.onkeydown=e=>{
 const k=e.key.toUpperCase();
 const hit=CH.find(c=>c[0]===k);
 if(hit){e.preventDefault();decide(hit[1]);return;}
 if(k==='N'){e.preventDefault();nextUnresolved();}
 else if(e.key==='ArrowLeft'){i=Math.max(0,i-1);render();}
 else if(e.key==='ArrowRight'){i=Math.min(S.items.length-1,i+1);render();}
 else if(/^[1-9]$/.test(e.key)){const b=cur().boxes[+e.key-1];if(b){sel=b.BOX_ID;draw();ctx();}}
};
boot();
</script></body></html>"""


def build_state():
    ledger = {r['BOX_ID']: r for r in
              json.loads((PKG / 'ledger.json').read_text(encoding='utf-8'))}
    by_img = {}
    for r in ledger.values():
        by_img.setdefault(r['IMAGE'], []).append(r)
    q = json.loads((PKG / 'u_resolution_queue.json').read_text(encoding='utf-8'))

    per = {}
    for r in q['rows']:
        per.setdefault(r['IMAGE'], []).append(r)
    items = []
    for img, rows in per.items():
        ids = {r['BOX_ID'] for r in rows}
        base = by_img[img][0]
        items.append({
            'IMAGE': img, 'run': rows[0]['run'],
            'img_w': base['img_w'], 'img_h': base['img_h'],
            'boxes': [{'BOX_ID': r['BOX_ID'], 'bbox': r['bbox_xywh'],
                       'ORIGINAL_CLASS': r['ORIGINAL_CLASS'],
                       'ORIGINAL_PROPOSAL': r['ORIGINAL_PROPOSAL'],
                       'FIRST_HUMAN_DECISION': r['FIRST_HUMAN_DECISION']}
                      for r in rows],
            'context': [{'bbox': b['bbox_xywh']} for b in by_img[img]
                        if b['BOX_ID'] not in ids
                        and b['eyecu_original_class'] == 'player'],
        })

    # ONLY u_resolution rows count as progress. The first-pass 'uncertain' on
    # these same boxes lives under mode='candidates' and is deliberately ignored.
    dec = {}
    p = PKG / 'decisions.json'
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip():
                d = json.loads(line)
                if d.get('mode') == MODE:
                    dec[d['BOX_ID']] = d['HUMAN_FINAL_CLASS']
    return {'items': items, 'decisions': dec,
            'total_boxes': len(q['rows']), 'total_images': len(items)}


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
            page = PAGE.replace('CHOICES_JSON', json.dumps(CHOICES))
            return self._send(200, page.encode('utf-8'), 'text/html; charset=utf-8')
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
        n = int(self.headers.get('Content-Length', 0))
        d = json.loads(self.rfile.read(n) or b'{}')
        if d.get('mode') != MODE:
            return self._send(400, b'{"error":"this server only writes u_resolution"}')
        allowed = {c[1] for c in CHOICES}
        if d.get('HUMAN_FINAL_CLASS') not in allowed:
            return self._send(400, b'{"error":"value not in the U-resolution vocabulary"}')
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
    ap.add_argument('--port', type=int, default=8770)
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args()
    st = build_state()
    print(f"u_resolution: {st['total_images']} images, {st['total_boxes']} boxes")
    print(f"already resolved in mode '{MODE}': {len(st['decisions'])}/{st['total_boxes']}"
          f"  (the first-pass 'U' is NOT counted)")
    url = f'http://127.0.0.1:{args.port}/'
    print(f'\nreview at {url}   Ctrl-C to stop')
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(('127.0.0.1', args.port), H).serve_forever()


if __name__ == '__main__':
    main()
