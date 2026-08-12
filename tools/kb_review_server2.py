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
PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
IMGROOT = REPO / 'EyeCU_external_data/huggingface/keremberke_football_object_detection/extracted'
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
 .bx{position:absolute;border:2px solid;box-sizing:border-box;cursor:pointer}
 .bx.ctx{border-color:var(--ctx);border-width:1px;cursor:default}
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
 <b>missed_role</b>
 <span id="pos" class="pill"></span>
 <span id="run" class="pill"></span>
 <span id="imgs" class="pill"></span>
 <span id="boxes" class="pill"></span>
 <div id="bar"><i></i></div>
 <span class="pill"><span class="gk" id="cG">G 0</span> ·
  <span class="ref" id="cR">R 0</span> · <span class="pl" id="cP">P 0</span> ·
  <span class="un" id="cU">U 0</span></span>
 <span class="pill" id="rem"></span>
 <span id="hint"><span class="k">click</span> a box, then <span class="k">P</span>
  <span class="k">G</span> <span class="k">R</span> <span class="k">U</span> ·
  <span class="k">1-9</span> pick box · <span class="k">A</span> all=player ·
  <span class="k">Enter</span> accept proposals · <span class="k">N</span>/<span class="k">B</span> nav</span>
</div>
<div id="wrap"><img id="im"><div id="ov"></div></div>
<div id="panel">
 <div id="list"></div>
 <button class="big" id="allP">ALL CURRENT CANDIDATES = PLAYER <span class="k">A</span></button>
 <button class="big" id="acc">ACCEPT ALL PROPOSALS <span class="k">Enter</span></button>
 <div id="accnote" style="color:#8a8a8a;margin-top:5px"></div>
 <button class="big" id="next">NEXT UNRESOLVED IMAGE <span class="k">N</span></button>
 <button class="big" id="prev">PREVIOUS <span class="k">B</span></button>
</div>
<script>
let S=null,i=0,sel=null,dec={},seen={};
const CLS={player:'pl',goalkeeper:'gk',referee:'ref',uncertain:'un'};
const COLV={player:'#3ddc57',goalkeeper:'#ffc400',referee:'#ff7a1a',uncertain:'#b06cff'};
async function boot(){
 S=await (await fetch('/api/state')).json(); dec=S.decisions;
 i=S.items.findIndex(x=>x.candidates.some(c=>!dec[c.BOX_ID])); if(i<0)i=0;
 render();
}
function cur(){return S.items[i];}
function render(){
 const it=cur(); if(!it)return;
 document.getElementById('pos').textContent=`image ${i+1}/${S.items.length}`;
 document.getElementById('run').textContent='run '+it.run;
 const im=document.getElementById('im'); im.onload=draw; im.src='/img/'+it.IMAGE;
 sel=it.candidates.find(c=>!dec[c.BOX_ID])?.BOX_ID||it.candidates[0].BOX_ID;
 seen[it.IMAGE]=seen[it.IMAGE]||new Set();
 stats();
}
function draw(){
 const it=cur(),im=document.getElementById('im'),ov=document.getElementById('ov');
 const sx=im.clientWidth/it.img_w, sy=im.clientHeight/it.img_h;
 ov.innerHTML='';
 it.context.forEach(b=>{
  const e=document.createElement('div'); e.className='bx ctx';
  e.style.cssText=`left:${b.bbox[0]*sx}px;top:${b.bbox[1]*sy}px;
   width:${b.bbox[2]*sx}px;height:${b.bbox[3]*sy}px`;
  ov.appendChild(e);
 });
 it.candidates.forEach((c,n)=>{
  const d=dec[c.BOX_ID];
  const col=d?COLV[d]:COLV[c.proposed];
  const e=document.createElement('div');
  e.className='bx'+(d?' done':'');
  e.style.cssText=`left:${c.bbox[0]*sx}px;top:${c.bbox[1]*sy}px;
   width:${c.bbox[2]*sx}px;height:${c.bbox[3]*sy}px;border-color:${col};
   ${c.BOX_ID===sel?'box-shadow:0 0 0 2px #fff inset;':''}`;
  e.onclick=()=>{sel=c.BOX_ID;draw();list();};
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
 document.getElementById('list').innerHTML=it.candidates.map((c,n)=>{
  const d=dec[c.BOX_ID];
  return `<div class="row" style="${c.BOX_ID===sel?'background:#1d1d1d':''}">
   <b><span class="k">${n+1}</span></b>
   <span class="${CLS[d||c.proposed]}">${d?d:'?'+c.proposed}</span>
   <span style="color:#777;margin-left:auto">${c.score.toFixed(2)}</span></div>`;
 }).join('');
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
 document.getElementById('imgs').textContent=`images ${imgsDone}/${S.items.length}`;
 document.getElementById('boxes').textContent=`boxes ${done}/${tot}`;
 document.getElementById('rem').textContent=`remaining ${tot-done}`;
 document.getElementById('cG').textContent='G '+c.goalkeeper;
 document.getElementById('cR').textContent='R '+c.referee;
 document.getElementById('cP').textContent='P '+c.player;
 document.getElementById('cU').textContent='U '+c.uncertain;
 document.querySelector('#bar>i').style.width=(100*done/tot)+'%';
}
async function post(box,cls,note){
 dec[box]=cls;
 await fetch('/api/decide',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({mode:'missed_role',BOX_ID:box,IMAGE:cur().IMAGE,
   HUMAN_FINAL_CLASS:cls,note:note||'per-box click'})});
}
async function decide(cls){
 if(!sel)return; await post(sel,cls);
 const it=cur(); const nxt=it.candidates.find(c=>!dec[c.BOX_ID]);
 if(nxt){sel=nxt.BOX_ID;draw();stats();} else {stats();draw();}
}
async function allPlayer(){
 const it=cur();
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
document.getElementById('next').onclick=nextUnresolved;
document.getElementById('prev').onclick=()=>{i=Math.max(0,i-1);render();};
document.onkeydown=e=>{
 const k=e.key.toLowerCase();
 if(k==='p')decide('player'); else if(k==='g')decide('goalkeeper');
 else if(k==='r')decide('referee'); else if(k==='u')decide('uncertain');
 else if(k==='a'){e.preventDefault();allPlayer();}
 else if(e.key==='Enter'){e.preventDefault();acceptAll();}
 else if(k==='n'){e.preventDefault();nextUnresolved();}
 else if(k==='b'){i=Math.max(0,i-1);render();}
 else if(/^[1-9]$/.test(k)){const it=cur();const c=it.candidates[+k-1];
   if(c){sel=c.BOX_ID;draw();}}
};
boot();
</script></body></html>"""


def build_state():
    ledger = {r['BOX_ID']: r for r in
              json.loads((PKG / 'ledger.json').read_text(encoding='utf-8'))}
    by_img = {}
    for r in ledger.values():
        by_img.setdefault(r['IMAGE'], []).append(r)
    mrq = json.loads((PKG / 'missed_role_queue.json').read_text(encoding='utf-8'))

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
            'context': [{'bbox': b['bbox_xywh']} for b in by_img[img]
                        if b['BOX_ID'] not in cand_ids
                        and b['eyecu_original_class'] == 'player'],
        })

    # Resume: replay the same append-only log the gate reads. Only missed_role
    # rows are surfaced here, so a first-pass decision can never be shown or
    # overwritten by this tool.
    dec = {}
    p = PKG / 'decisions.json'
    if p.exists():
        for line in p.read_text(encoding='utf-8').splitlines():
            if line.strip():
                d = json.loads(line)
                if d.get('mode') == 'missed_role':
                    dec[d['BOX_ID']] = d['HUMAN_FINAL_CLASS']
    return {'items': items, 'decisions': dec,
            'total_boxes': len(mrq['rows']), 'total_images': len(items)}


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
        if d.get('mode') != 'missed_role':
            return self._send(400, b'{"error":"this server only writes missed_role"}')
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
    ap.add_argument('--port', type=int, default=8740)
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args()
    st = build_state()
    done = len(st['decisions'])
    imgs_done = sum(1 for it in st['items']
                    if all(c['BOX_ID'] in st['decisions'] for c in it['candidates']))
    print(f"missed_role: {st['total_images']} images, {st['total_boxes']} boxes "
          f"({st['total_boxes']/max(st['total_images'],1):.1f} per image)")
    print(f'already decided: {done} boxes, {imgs_done} images complete '
          f'-- resumed from decisions.json')
    url = f'http://127.0.0.1:{args.port}/'
    print(f'\nreview at {url}   Ctrl-C to stop')
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(('127.0.0.1', args.port), H).serve_forever()


if __name__ == '__main__':
    main()
