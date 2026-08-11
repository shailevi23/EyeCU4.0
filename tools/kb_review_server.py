#!/usr/bin/env python
"""
Local review server for the Keremberke class repair.

Runs on localhost with the standard library only -- no framework, no upload, no
network. Serves the images read-only from the immutable extract and appends every
human decision to decisions.json as it is made, so closing the tab loses nothing.

Three modes, because they answer different questions and must not be mixed:

    candidates   the 4,153 proposed goalkeeper/referee/ambiguous boxes
    qa_player    250 stratified LIKELY_PLAYER boxes -- measures triage RECALL
    qa_nocand    every image where nothing was flagged -- catches officials the
                 triage missed entirely

Keys: P player, G goalkeeper, R referee, U uncertain, Tab next box in image,
N/space next image, B previous, A accept every proposal in this image.

'A' is a human action, not automation: it confirms the proposals the reviewer is
looking at, one keypress, and each box is still written to the ledger with the
reviewer as the author. Nothing is ever decided while the image is unseen.
"""

import argparse
import json
import mimetypes
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

REPO = Path(__file__).resolve().parents[1]
PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
IMGROOT = REPO / 'EyeCU_external_data/huggingface/keremberke_football_object_detection/extracted'
LOCK = threading.Lock()

PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><title>Keremberke review</title>
<style>
 body{margin:0;background:#111;color:#ddd;font:13px system-ui,sans-serif}
 #top{display:flex;gap:14px;align-items:center;padding:6px 10px;background:#000;
      position:sticky;top:0;z-index:5;flex-wrap:wrap}
 #wrap{position:relative;margin:8px auto;width:fit-content}
 img{display:block;max-width:98vw;max-height:78vh}
 .bx{position:absolute;border:2px solid;box-sizing:border-box;pointer-events:none}
 .tag{position:absolute;font:11px monospace;background:#000c;padding:0 3px;
      transform:translateY(-100%);white-space:nowrap}
 b{color:#fff}.k{background:#222;border:1px solid #444;border-radius:3px;padding:1px 5px}
 #done{color:#6f6}#pend{color:#fd6}
 select,button{background:#222;color:#ddd;border:1px solid #444;padding:3px 6px}
</style></head><body>
<div id="top">
 <span>mode <select id="mode"></select></span>
 <span id="pos"></span>
 <span id="img"></span>
 <span id="run"></span>
 <span id="cur"></span>
 <span><span class="k">P</span>layer <span class="k">G</span>K <span class="k">R</span>ef
  <span class="k">U</span>ncertain | <span class="k">Tab</span> box
  <span class="k">N</span>ext <span class="k">B</span>ack <span class="k">A</span>ccept-all</span>
 <span id="prog"></span>
</div>
<div id="wrap"><img id="im"><div id="ov"></div></div>
<script>
let S=null,mode='candidates',i=0,sel=0,items=[],dec={};
const COL={player:'#4f4',goalkeeper:'#fc0',referee:'#f80',ambiguous:'#ccc',ball:'#f44'};
async function boot(){
 S=await (await fetch('/api/state')).json();
 const m=document.getElementById('mode');
 m.innerHTML=Object.keys(S.modes).map(k=>`<option value="${k}">${k} (${S.modes[k].length})</option>`).join('');
 m.onchange=()=>{mode=m.value;i=0;sel=0;load()};
 dec=S.decisions; mode=m.value; load();
}
function load(){
 items=S.modes[mode]; if(!items.length)return;
 i=Math.max(0,Math.min(i,items.length-1));
 const it=items[i];
 document.getElementById('pos').textContent=`${i+1}/${items.length}`;
 document.getElementById('img').textContent=it.IMAGE;
 document.getElementById('run').textContent='run '+(it.run||'?');
 const im=document.getElementById('im');
 im.onload=draw; im.src='/img/'+it.IMAGE;
 progress();
}
function draw(){
 const it=items[i],im=document.getElementById('im'),ov=document.getElementById('ov');
 const sx=im.clientWidth/it.img_w, sy=im.clientHeight/it.img_h;
 ov.innerHTML='';
 it.boxes.forEach((b,n)=>{
  const isCand=it.candidate_ids.includes(b.BOX_ID);
  const d=dec[b.BOX_ID];
  const c=d?COL[d]||'#fff':(isCand?COL[(b.PROPOSED_CLASS||'ambiguous')]:'#3a3a3a');
  const e=document.createElement('div'); e.className='bx';
  e.style.cssText=`left:${b.bbox[0]*sx}px;top:${b.bbox[1]*sy}px;width:${b.bbox[2]*sx}px;
   height:${b.bbox[3]*sy}px;border-color:${c};border-width:${isCand&&it.candidate_ids[sel]===b.BOX_ID?4:2}px`;
  ov.appendChild(e);
  if(isCand){const t=document.createElement('div');t.className='tag';
   t.style.cssText=`left:${b.bbox[0]*sx}px;top:${b.bbox[1]*sy}px;color:${c}`;
   t.textContent=(d?('YOU: '+d):('proposed: '+(b.PROPOSED_CLASS||'ambiguous')))+' · '+b.signals;
   ov.appendChild(t);}
 });
 const cid=it.candidate_ids[sel];
 document.getElementById('cur').textContent=cid?('box '+(sel+1)+'/'+it.candidate_ids.length+(dec[cid]?' = '+dec[cid]:'')):'(no candidate)';
}
function progress(){
 const tot=S.modes[mode].reduce((a,x)=>a+x.candidate_ids.length,0);
 const done=S.modes[mode].reduce((a,x)=>a+x.candidate_ids.filter(z=>dec[z]).length,0);
 document.getElementById('prog').innerHTML=`<span id="done">${done}</span>/<span id="pend">${tot}</span> decided`;
}
async function decide(cls){
 const it=items[i],cid=it.candidate_ids[sel]; if(!cid)return;
 dec[cid]=cls;
 await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({mode:mode,BOX_ID:cid,IMAGE:it.IMAGE,HUMAN_FINAL_CLASS:cls})});
 if(sel<it.candidate_ids.length-1){sel++;}else{i++;sel=0;load();return;}
 draw();progress();
}
async function acceptAll(){
 const it=items[i];
 for(const cid of it.candidate_ids){
  const b=it.boxes.find(z=>z.BOX_ID===cid);
  if(!b.PROPOSED_CLASS||dec[cid])continue;
  dec[cid]=b.PROPOSED_CLASS;
  await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},
   body:JSON.stringify({mode:mode,BOX_ID:cid,IMAGE:it.IMAGE,
    HUMAN_FINAL_CLASS:b.PROPOSED_CLASS,note:'accept-all keypress, image was on screen'})});
 }
 i++;sel=0;load();
}
document.onkeydown=e=>{
 const k=e.key.toLowerCase();
 if(k==='p')decide('player'); else if(k==='g')decide('goalkeeper');
 else if(k==='r')decide('referee'); else if(k==='u')decide('uncertain');
 else if(k==='a')acceptAll();
 else if(k==='tab'){e.preventDefault();const it=items[i];sel=(sel+1)%Math.max(1,it.candidate_ids.length);draw();}
 else if(k==='n'||k===' '){e.preventDefault();i++;sel=0;load();}
 else if(k==='b'){i--;sel=0;load();}
};
boot();
</script></body></html>"""


def build_state():
    ledger = {r['BOX_ID']: r for r in
              json.loads((PKG / 'ledger.json').read_text(encoding='utf-8'))}
    by_img = {}
    for r in ledger.values():
        by_img.setdefault(r['IMAGE'], []).append(r)

    def pack(image, cand_ids):
        rows = by_img[image]
        return {'IMAGE': image, 'run': rows[0].get('run'),
                'img_w': rows[0]['img_w'], 'img_h': rows[0]['img_h'],
                'candidate_ids': cand_ids,
                'boxes': [{'BOX_ID': r['BOX_ID'], 'bbox': r['bbox_xywh'],
                           'PROPOSED_CLASS': r['PROPOSED_CLASS'],
                           'signals': r['signals']} for r in rows]}

    queue = json.loads((PKG / 'review_queue.json').read_text(encoding='utf-8'))
    modes = {'candidates': [pack(q['IMAGE'], q['candidate_box_ids']) for q in queue]}

    qp = json.loads((PKG / 'qa_likely_player.json').read_text(encoding='utf-8'))
    per_img = {}
    for r in qp['rows']:
        per_img.setdefault(r['IMAGE'], []).append(r['BOX_ID'])
    modes['qa_player'] = [pack(k, v) for k, v in per_img.items()]

    qn = json.loads((PKG / 'qa_no_candidate_images.json').read_text(encoding='utf-8'))
    modes['qa_nocand'] = [pack(r['IMAGE'], [b['BOX_ID'] for b in
                                            by_img[r['IMAGE']]
                                            if b['eyecu_original_class'] == 'player'])
                          for r in qn['rows'] if r['IMAGE'] in by_img]

    dpath = PKG / 'decisions.json'
    dec = {}
    if dpath.exists():
        for line in dpath.read_text(encoding='utf-8').splitlines():
            if line.strip():
                d = json.loads(line)
                dec[d['BOX_ID']] = d['HUMAN_FINAL_CLASS']
    return {'modes': modes, 'decisions': dec}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype='application/json'):
        self.send_response(code)
        self.send_header('Content-Type', ctype)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        p = unquote(urlparse(self.path).path)
        if p == '/':
            return self._send(200, PAGE.encode('utf-8'), 'text/html; charset=utf-8')
        if p == '/api/state':
            return self._send(200, json.dumps(build_state()).encode('utf-8'))
        if p.startswith('/img/'):
            rel = p[len('/img/'):]
            split, name = rel.split('/', 1)
            for cand in (IMGROOT / split).rglob(name):
                data = cand.read_bytes()
                return self._send(200, data,
                                  mimetypes.guess_type(name)[0] or 'image/jpeg')
            return self._send(404, b'not found', 'text/plain')
        return self._send(404, b'not found', 'text/plain')

    def do_POST(self):
        if urlparse(self.path).path != '/api/decide':
            return self._send(404, b'', 'text/plain')
        n = int(self.headers.get('Content-Length', 0))
        d = json.loads(self.rfile.read(n) or b'{}')
        d['recorded_utc'] = __import__('time').strftime('%Y-%m-%dT%H:%M:%SZ',
                                                        __import__('time').gmtime())
        d['author'] = 'human reviewer'
        with LOCK:
            with open(PKG / 'decisions.json', 'a', encoding='utf-8') as f:
                f.write(json.dumps(d) + '\n')
        return self._send(200, b'{"ok":true}')


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8733)
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args()
    st = build_state()
    for k, v in st['modes'].items():
        print(f'{k:<12} {len(v)} images, '
              f'{sum(len(x["candidate_ids"]) for x in v)} boxes to decide')
    print(f'decisions already recorded: {len(st["decisions"])}')
    url = f'http://127.0.0.1:{args.port}/'
    print(f'\nreview at {url}   (Ctrl-C to stop; decisions append to '
          f'{(PKG / "decisions.json").relative_to(REPO)})')
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(('127.0.0.1', args.port), H).serve_forever()


if __name__ == '__main__':
    main()
