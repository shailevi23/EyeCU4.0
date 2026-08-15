#!/usr/bin/env python
"""
Re-open an effective final_target EXCLUDE_IMAGE, with its full cost on screen.

Excluding an image is the honest answer when one target's role genuinely cannot
be read: leaving it labelled `player` puts a wrong label in TRAIN. But the cost
is the whole image, and that cost is invisible at the moment the decision is
made -- the reviewer is looking at one box, not at the ten other annotations
that leave with it.

For the one exclusion currently in force that means losing:

    11 annotations, including 1 valid original ball, and TWO annotations a
    human already corrected to goalkeeper and referee in an earlier pass.

Those two corrections are the kind of finding the whole retrospective sweep
existed to produce. Discarding them to avoid guessing one occluded role may
still be right -- that is the reviewer's call, not this tool's -- but it should
be made while looking at the number.

WHAT THIS TOOL DOES NOT DO. It changes nothing on its own, it proposes nothing,
and KEEP EXCLUDED writes no event at all, because "I looked again and did not
change my mind" must not become a new decision that supersedes the old one.

    python tools/kb_exclusion_revisit_server.py

Answering P/G/R/M appends one decision that supersedes the exclusion by the
ordinary precedence rule -- latest human answer wins -- so the image returns to
the derived export with every other annotation intact. The source is untouched
either way, and the exclusion event stays in history.
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

PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
LOCK = threading.Lock()
# The same vocabulary the other passes use. M is the disposition for a real
# human outside the four classes; it settles the box without giving it a class,
# and like a role it supersedes the exclusion.
ANSWERS = ('player', 'goalkeeper', 'referee', 'NON_TARGET_HUMAN')
MODE = 'final_target'

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>exclusion revisit</title>
<style>
 :root{--gk:#ffc400;--ref:#ff7a1a;--pl:#3ddc57;--un:#b06cff;--ball:#4aa3ff}
 body{margin:0;background:#0e0e0e;color:#e6e6e6;font:13px system-ui,sans-serif}
 #top{display:flex;gap:14px;align-items:center;padding:7px 12px;background:#000;
      position:sticky;top:0;z-index:9;flex-wrap:wrap;border-bottom:1px solid #222}
 .pill{padding:1px 7px;border-radius:9px;background:#1c1c1c;border:1px solid #333}
 #wrap{position:relative;margin:10px 320px 10px 10px;width:fit-content;
       overflow:auto;max-height:84vh;max-width:calc(100vw - 350px)}
 #stage{position:relative;transform-origin:0 0}
 img{display:block}
 .ex{position:absolute;border:1px solid #4a4a4a;box-sizing:border-box;
     pointer-events:none}
 .ex.ball{border-color:var(--ball);border-width:3px}
 .ex.fixed{border-width:3px}
 /* the box the exclusion was made for */
 .target{position:absolute;border:4px solid #b06cff;box-sizing:border-box;
         pointer-events:none;box-shadow:0 0 0 3px #000,0 0 16px 5px #b06cffcc;
         animation:p 1.4s infinite}
 @keyframes p{50%{box-shadow:0 0 0 3px #000,0 0 6px 2px #b06cff66}}
 .tag{position:absolute;font:11px/1.3 monospace;background:#000d;padding:1px 4px;
      transform:translateY(-100%);white-space:nowrap;pointer-events:none}
 #panel{position:fixed;right:10px;top:52px;width:300px;background:#141414;
        border:1px solid #2a2a2a;border-radius:6px;padding:9px;z-index:8;
        max-height:86vh;overflow:auto}
 button{background:#1e1e1e;color:#ddd;border:1px solid #3a3a3a;border-radius:4px;
        padding:3px 8px;cursor:pointer;font:12px system-ui}
 button:hover{background:#2a2a2a}
 button.big{width:100%;margin-top:6px;padding:6px}
 .k{background:#222;border:1px solid #3a3a3a;border-radius:3px;padding:0 5px;
    font:11px monospace}
 .warn{background:#3a1414;border:1px solid #7a3a3a;border-radius:4px;padding:7px;
       margin:6px 0;font-size:11px}
 .ok{background:#132a13;border:1px solid #2f5a2f;border-radius:4px;padding:7px;
     margin:6px 0;font-size:11px}
 .note{color:#8a8a8a;font-size:11px;margin:-2px 0 8px}
 table{border-collapse:collapse;width:100%;font:11px monospace}
 td{padding:1px 3px;border-bottom:1px solid #1e1e1e}
</style></head><body>
<div id="top">
 <b style="color:#b06cff">EXCLUSION REVISIT</b>
 <span id="pos" class="pill"></span>
 <span id="img" class="pill" style="font:11px monospace"></span>
 <span id="state" class="pill"></span>
 <span style="color:#8a8a8a;font-size:11px"><span class="k">P</span>
  <span class="k">G</span> <span class="k">R</span> give it a role ·
  <span class="k">M</span> non-active human · <span class="k">K</span> keep
  excluded · <span class="k">+</span>/<span class="k">-</span> zoom</span>
</div>
<div id="imgerr" style="display:none;margin:10px 320px 10px 10px;background:#3a1414;
     border:1px solid #7a3a3a;border-radius:6px;padding:12px"></div>
<div id="wrap"><div id="stage"><img id="im"><div id="ov"></div></div></div>
<div id="panel"><div id="body"></div></div>
<script>
let S=null,i=0,zoom=1;
const COL={player:'#3ddc57',goalkeeper:'#ffc400',referee:'#ff7a1a',
           ball:'#4aa3ff',NON_TARGET_HUMAN:'#888'};
var boot=async function(){
 S=await (await fetch('/api/state')).json();
 render();
};
function cur(){return S.items[i];}
function render(){
 const t=cur(); if(!t){document.getElementById('body').innerHTML=
   '<b style="color:#8fe08f">no effective image exclusion outstanding</b>';return;}
 document.getElementById('pos').textContent=`item ${i+1}/${S.items.length}`;
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
  err.innerHTML='<b>IMAGE COULD NOT BE LOADED</b><br>'+t.IMAGE+'<br>'+why
   +'<br><br>Do not answer this item.';};
 im.src='/img/'+t.IMAGE;
}
function applyZoom(){
 const st=document.getElementById('stage');
 st.style.transform='scale('+zoom+')';
 const im=document.getElementById('im');
 st.style.width=im.naturalWidth+'px'; st.style.height=im.naturalHeight+'px';
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
   g.textContent=label;ov.appendChild(g);}
 };
 t.others.forEach(o=>{
  const corrected=o.effective&&o.effective!==o.original_class
                  &&o.effective!=='no decision';
  add('ex'+(o.original_class==='ball'?' ball':'')+(corrected?' fixed':''),
      o.bbox,o.original_class==='ball'?COL.ball:(corrected?COL[o.effective]:null),
      o.original_class==='ball'?'BALL GT':(corrected?'human: '+o.effective:null));
 });
 add('target',t.bbox,null,'train:7331 -- the box this exclusion was made for');
 panel();
}
function panel(){
 const t=cur();
 document.getElementById('state').textContent=
   t.effective_excluded?'currently EXCLUDED':'no longer excluded';
 document.getElementById('body').innerHTML=`
  <div style="font-size:11px;color:#8a8a8a">the box this exclusion was made for</div>
  <div style="font:12px monospace;color:#bbb">${t.BOX_ID}</div>
  <div style="font-size:11px;color:#8a8a8a">original class
   <b>${t.original_class}</b> · box
   ${t.bbox.map(v=>Math.round(v)).join(', ')}</div>

  <div style="margin-top:8px;font-size:11px;color:#8a8a8a">how it got here</div>
  <table>${t.history.map(h=>`<tr><td style="color:#777">${(h.recorded_utc||'')
    .slice(5,16).replace('T',' ')}</td><td>${h.mode}</td>
    <td><b>${h.value}</b></td></tr>`).join('')}</table>
  <div class="note">the role could not be read even on the third look, so the
   image was dropped rather than guessing a class.</div>

  <div class="warn"><b>EXCLUDING THIS IMAGE ALSO DISCARDS</b><br>
   ${t.others.length} other annotations in it, including:
   <table style="margin-top:4px">
   ${t.others.filter(o=>o.original_class==='ball').map(o=>
     `<tr><td style="color:#4aa3ff">BALL</td><td>${o.BOX_ID}</td>
      <td>valid original ball GT</td></tr>`).join('')}
   ${t.others.filter(o=>o.effective&&o.effective!==o.original_class
                        &&o.effective!=='no decision').map(o=>
     `<tr><td style="color:${COL[o.effective]||'#ddd'}">${o.effective}</td>
      <td>${o.BOX_ID}</td><td>human-corrected</td></tr>`).join('')}
   </table>
   <div style="margin-top:4px">those corrections are exactly what the
    retrospective sweep existed to find.</div></div>

  <button class="big" id="bP" style="border-color:#3ddc57">P &mdash; PLAYER</button>
  <button class="big" id="bG" style="border-color:#ffc400">G &mdash; GOALKEEPER</button>
  <button class="big" id="bR" style="border-color:#ff7a1a">R &mdash; REFEREE</button>
  <button class="big" id="bM" style="border-color:#777">M &mdash; NON-ACTIVE HUMAN</button>
  <div class="note">any of these supersedes the exclusion. The image and all
   ${t.others.length} other annotations return to the export; the source is not
   touched and the exclusion stays in history.</div>
  <button class="big" id="bK" style="border-color:#7a3a3a">K &mdash; KEEP EXCLUDED</button>
  <div class="note">writes nothing. Looking again and not changing your mind is
   not a new decision.</div>
  <div id="done"></div>`;
 document.getElementById('bP').onclick=()=>answer('player');
 document.getElementById('bG').onclick=()=>answer('goalkeeper');
 document.getElementById('bR').onclick=()=>answer('referee');
 document.getElementById('bM').onclick=()=>answer('NON_TARGET_HUMAN');
 document.getElementById('bK').onclick=keep;
}
async function answer(v){
 const t=cur();
 if(!confirm('Answer '+t.BOX_ID+' as '+v+'?\n\nThis supersedes the image '
   +'exclusion. The image and its '+t.others.length+' other annotations return '
   +'to the derived export.'))return;
 const r=await fetch('/api/answer',{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({BOX_ID:t.BOX_ID,HUMAN_FINAL_CLASS:v})});
 const j=await r.json();
 if(!r.ok){alert(j.error||'refused');return;}
 document.getElementById('done').innerHTML='<div class="ok"><b>RECORDED '+v
  +'</b><br>image excluded: <b>'+(j.effective_excluded?'yes':'no')+'</b><br>'
  +j.restored+' annotations return to the export.<br>'
  +'The EXCLUDE_IMAGE event remains in history.</div>';
 S=await (await fetch('/api/state')).json();
 if(!S.items.length){document.getElementById('body').innerHTML+=
   '<div class="ok">no exclusion outstanding. Re-run the gate and '
   +'<code>kb_export_v2.py --check</code>.</div>';}
}
function keep(){
 document.getElementById('done').innerHTML='<div class="ok"><b>KEPT EXCLUDED</b>'
  +'<br>nothing was written. The exclusion remains effective exactly as it was.'
  +'</div>';
}
document.onkeydown=e=>{
 const k=e.key.toLowerCase();
 if(k==='p')answer('player'); else if(k==='g')answer('goalkeeper');
 else if(k==='r')answer('referee'); else if(k==='m')answer('NON_TARGET_HUMAN');
 else if(k==='k')keep();
 else if(e.key==='+'||e.key==='='){zoom=Math.min(zoom*1.25,8);applyZoom();}
 else if(e.key==='-'){zoom=Math.max(zoom/1.25,0.25);applyZoom();}
 else if(k==='0'){zoom=1;applyZoom();}
};
boot();
</script></body></html>"""


def _ledger():
    return {r['BOX_ID']: r for r in
            json.loads((PKG / 'ledger.json').read_text(encoding='utf-8'))}


def build_state():
    """Every box whose EFFECTIVE disposition currently excludes its image."""
    res = kb_decisions.resolve(PKG / 'decisions.json')
    led = _ledger()
    by_img = {}
    for r in led.values():
        by_img.setdefault(r['IMAGE'], []).append(r)
    items = []
    for b, r in sorted(res.items()):
        if b not in led or r['disposition'] != 'EXCLUDE_IMAGE':
            continue
        l = led[b]
        others = []
        for z in sorted(by_img[l['IMAGE']], key=lambda q: q['BOX_ID']):
            if z['BOX_ID'] == b:
                continue
            zr = res.get(z['BOX_ID'], {})
            others.append({
                'BOX_ID': z['BOX_ID'], 'bbox': z['bbox_xywh'],
                'original_class': z['eyecu_original_class'],
                'effective': (zr.get('final_class') or zr.get('disposition')
                              or 'no decision')})
        items.append({
            'BOX_ID': b, 'IMAGE': l['IMAGE'], 'bbox': l['bbox_xywh'],
            'original_class': l['eyecu_original_class'],
            'img_w': l['img_w'], 'img_h': l['img_h'],
            'effective_excluded': True,
            'history': r['history'], 'others': others})
    return {'items': items,
            'source_log': kb_decisions.log_version(PKG / 'decisions.json')}


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
            return self._send(200, json.dumps(build_state()).encode())
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
        if urlparse(self.path).path != '/api/answer':
            return self._send(404, b'', 'text/plain')
        n = int(self.headers.get('Content-Length', 0))
        d = json.loads(self.rfile.read(n) or b'{}')
        box = str(d.get('BOX_ID', ''))
        live = {t['BOX_ID']: t for t in build_state()['items']}
        if box not in live:
            return self._send(400, b'{"error":"this box is not currently '
                                   b'excluding its image"}')
        v = d.get('HUMAN_FINAL_CLASS')
        if v not in ANSWERS:
            return self._send(400, b'{"error":"answer must be player, goalkeeper, '
                                   b'referee or NON_TARGET_HUMAN"}')
        img = live[box]['IMAGE']
        rec = {'mode': MODE, 'BOX_ID': box, 'IMAGE': img,
               'HUMAN_FINAL_CLASS': v,
               'supersedes': 'EXCLUDE_IMAGE',
               'note': ('exclusion revisited with the full cost of the image on '
                        'screen; this answer supersedes it by ordinary '
                        'precedence and the exclusion event stays in history'),
               'image_returns_to_export': True,
               'annotations_restored': len(live[box]['others']) + 1,
               'recorded_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
               'author': 'human reviewer'}
        with LOCK:
            with open(PKG / 'decisions.json', 'a', encoding='utf-8') as fh:
                fh.write(json.dumps(rec) + '\n')
                fh.flush()
        after = kb_decisions.resolve(PKG / 'decisions.json')[box]
        return self._send(200, json.dumps(
            {'ok': True, 'recorded_utc': rec['recorded_utc'],
             'effective_excluded': after['disposition'] == 'EXCLUDE_IMAGE',
             'restored': rec['annotations_restored']}).encode())


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8743)
    ap.add_argument('--no-browser', action='store_true')
    args = ap.parse_args()
    st = build_state()
    if not st['items']:
        print('no effective image exclusion outstanding; nothing to revisit')
        return
    ok, problems = kb_images.preflight([t['IMAGE'] for t in st['items']])
    if not ok:
        print(f'REFUSING TO START: {len(problems)} image(s) cannot be resolved.')
        for im, why in problems:
            print(f'  {im}\n    {why}')
        sys.exit(1)
    for t in st['items']:
        ball = [o for o in t['others'] if o['original_class'] == 'ball']
        fixed = [o for o in t['others']
                 if o['effective'] not in ('no decision', o['original_class'])]
        print(f'{t["BOX_ID"]}  {t["IMAGE"]}')
        print(f'  excluding it discards {len(t["others"]) + 1} annotations: '
              f'{len(ball)} original ball GT, {len(fixed)} human-corrected')
        for o in fixed:
            print(f'    {o["BOX_ID"]:<12} {o["original_class"]} -> {o["effective"]}')
    print('\nNothing is changed by opening this. KEEP EXCLUDED writes no event.')
    url = f'http://127.0.0.1:{args.port}/'
    print(f'\nrevisit at {url}   Ctrl-C to stop')
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(('127.0.0.1', args.port), H).serve_forever()


if __name__ == '__main__':
    main()
