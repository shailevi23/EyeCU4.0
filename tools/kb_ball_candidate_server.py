#!/usr/bin/env python
"""
Answer the model's ball proposals: Y / N / U, one candidate at a time.

    IS THIS A REAL MISSING FOOTBALL?

The detector proposed 488 unmatched boxes across the unresolved PP images. Each
is a question, not an annotation. A human YES records a HUMAN_APPROVED_ADDITION
carrying the model's geometry as *approved* geometry; a NO records that the
proposal was wrong; a UNSURE resolves nothing and is reported separately.

WHAT THIS TOOL CANNOT TELL YOU. Completing this queue says the proposals were
adjudicated. It does not say the images are clean, because a football the
detector never proposed cannot appear here. That is measured separately by the
residual QA, and the distinction is the same one the role pass learned the hard
way: 4,153 of 4,153 candidates answered, and a QA of the REJECTED population
still found 6.40% missed.

NO ACTIVE/NON-ACTIVE QUESTION. The ontology is settled -- every real physical
football is football -- so the reviewer is never asked which kind it is.

    python tools/kb_ball_candidate_server.py

Append-only. No annotation is modified and no proposal becomes GT without a Y.
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
import kb_ball_candidates as CAND                                 # noqa: E402
import kb_ball_qa_sample as kb_sample                             # noqa: E402
import kb_decisions                                               # noqa: E402
import kb_images                                                  # noqa: E402

PKG = REPO / 'experiments' / 'external_sources' / 'keremberke_review'
DECISIONS = PKG / 'decisions.json'
LOCK = threading.Lock()

CAND_MODE = 'ball_candidate_review'
YES, NO, UNSURE = 'YES', 'NO', 'UNSURE'
VERDICTS = (YES, NO, UNSURE)

PAGE = r"""<!doctype html><html><head><meta charset="utf-8">
<title>ball candidate review</title>
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
 #ov{position:absolute;left:0;top:0;width:100%;height:100%;pointer-events:none}
 .cand{position:absolute;border:3px solid #ffd23d;box-sizing:border-box;
       box-shadow:0 0 0 2px #000,0 0 14px #ffd23d}
 .sib{position:absolute;border:1px dashed #7a6a2a;box-sizing:border-box}
 .ballgt{position:absolute;border:2px solid #4aa3ff;box-sizing:border-box;
         box-shadow:0 0 0 1px #000}
 .ctx{position:absolute;border:1px solid #333;box-sizing:border-box}
 .tag{position:absolute;font:11px/1.3 monospace;background:#000d;padding:1px 4px;
      transform:translateY(-100%);white-space:nowrap}
 #panel{position:fixed;right:10px;top:52px;width:276px;background:#141414;
        border:1px solid #2a2a2a;border-radius:6px;padding:9px;z-index:8;
        max-height:84vh;overflow:auto}
 button{background:#1e1e1e;color:#ddd;border:1px solid #3a3a3a;border-radius:4px;
        padding:3px 8px;cursor:pointer;font:12px system-ui}
 button:hover{background:#2a2a2a}
 button.big{width:100%;margin-top:6px;padding:7px;text-align:left}
 .k{background:#222;border:1px solid #3a3a3a;border-radius:3px;padding:0 6px;
    font:12px monospace;margin-right:6px;min-width:16px;display:inline-block;
    text-align:center}
 .ok{background:#132a13;border:1px solid #2f5a2f;border-radius:4px;padding:6px;
     margin:6px 0;font-size:11px}
 .no{background:#3a1414;border:1px solid #7a3a3a;border-radius:4px;padding:6px;
     margin:6px 0;font-size:11px}
 .uns{background:#2f2a10;border:1px solid #6a5a20;border-radius:4px;padding:6px;
      margin:6px 0;font-size:11px}
 .note{color:#8a8a8a;font-size:11px;margin:-2px 0 8px}
 .qbox{font-size:13px;font-weight:600;margin:2px 0 8px}
 #tiles{display:flex;gap:3px;flex-wrap:wrap;margin:5px 0}
 #tiles button{padding:2px 7px;font:11px monospace}
 #tiles button.on{background:#20402a;border-color:#3ddc57;color:#cfc}
 #legend{font-size:11px;color:#888;margin:6px 0}
 #legend i{display:inline-block;width:10px;height:10px;margin-right:4px;
           vertical-align:middle}
</style></head><body>
<div id="top">
 <b>BALL CANDIDATE REVIEW</b>
 <span id="pos" class="pill"></span>
 <span id="rem" class="pill"></span>
 <span id="img" class="pill" style="font:11px monospace"></span>
 <div id="bar"><i></i></div>
 <span id="zoomp" class="pill"></span>
 <span id="tally" class="pill" style="font-size:11px"></span>
 <span class="pill" style="font:10px monospace;color:#777">build __BUILD__</span>
</div>
<div id="imgerr" style="display:none;margin:10px 300px 10px 10px;background:#3a1414;
     border:1px solid #7a3a3a;border-radius:6px;padding:12px;max-width:640px"></div>
<div id="wrap"><div id="stage"><img id="im"><div id="ov"></div></div></div>
<div id="panel">
 <div class="qbox">IS THIS A REAL MISSING FOOTBALL?</div>
 <div id="legend">
  <div><i style="background:#ffd23d"></i>the model proposal being judged</div>
  <div><i style="background:#7a6a2a"></i>other proposals in this image</div>
  <div><i style="background:#4aa3ff"></i>existing football annotations</div>
 </div>
 <div id="state"></div>
 <button class="big" id="aY"><span class="k">Y</span>YES &mdash; real missing football</button>
 <button class="big" id="aN"><span class="k">N</span>NO &mdash; not a football</button>
 <button class="big" id="aU"><span class="k">U</span>UNSURE</button>
 <div class="note">a YES approves this geometry as a human-approved addition.
  Nothing is written to the dataset by this tool.</div>
 <div style="border-top:1px solid #2a2a2a;margin:9px 0 6px"></div>
 <div style="font-size:11px;color:#8a8a8a">view</div>
 <div id="tiles"></div>
 <button class="big" id="bNx"><span class="k">M</span>NEXT</button>
 <button class="big" id="bB"><span class="k">B</span>PREVIOUS</button>
 <button class="big" id="bJ"><span class="k">J</span>NEXT UNANSWERED</button>
 <div id="meta" style="font-size:11px;color:#777;margin-top:8px"></div>
 <div id="hist" style="font-size:11px;color:#888;margin-top:8px"></div>
</div>
<script>
let S=null,i=0,zoom=1,tile=-1;
const TILE_ZOOM=4,TCOLS=4,TROWS=3,OVERLAP=0.12;
var boot=async function(){
 S=await (await fetch('/api/state')).json();
 i=S.items.findIndex(x=>!x.verdict); if(i<0)i=0;
 render();
};
function cur(){return S.items[i];}
function remaining(){return S.items.filter(x=>!x.verdict).length;}
function render(){
 const t=cur(); if(!t)return;
 zoom=1; tile=-1;
 document.getElementById('pos').textContent=`candidate ${i+1}/${S.items.length}`;
 document.getElementById('rem').textContent=`${remaining()} remaining`;
 document.getElementById('img').textContent=t.IMAGE;
 const im=document.getElementById('im');
 const err=document.getElementById('imgerr');
 err.style.display='none'; im.style.display='block';
 im.onload=()=>{applyZoom();zoomTo();draw();};
 im.onerror=async()=>{im.style.display='none';
  let why='the server did not return an image';
  try{const r=await fetch('/img/'+t.IMAGE);const j=await r.json();
      why=j.error||why;}catch(e){}
  err.style.display='block';
  err.innerHTML='<b>IMAGE COULD NOT BE LOADED</b><br><span style="font:11px '
   +'monospace">'+t.IMAGE+'</span><br>'+why+'<br><br>Do not answer this one.';};
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
function zoomTo(){
 // a 6 px proposal is unreadable at fit, so open ON the candidate
 const t=cur(),wrap=document.getElementById('wrap');
 zoom=TILE_ZOOM; applyZoom(); tile=-2;
 const b=t.bbox_xywh;
 wrap.scrollLeft=Math.max(0,(b[0]+b[2]/2)*zoom-wrap.clientWidth/2);
 wrap.scrollTop =Math.max(0,(b[1]+b[3]/2)*zoom-wrap.clientHeight/2);
 tilebar();
}
function tilebar(){
 const bar=document.getElementById('tiles');
 bar.innerHTML='';
 const mk=(label,idx)=>{
  const b=document.createElement('button');
  b.textContent=label; if(idx===tile)b.className='on';
  b.onclick=()=>gotoTile(idx); bar.appendChild(b);
 };
 mk('FIT',-1); mk('CANDIDATE',-2);
 for(let k=0;k<TCOLS*TROWS;k++)mk(String(k+1),k);
}
function gotoTile(idx){
 const t=cur(),im=document.getElementById('im');
 const W=im.naturalWidth||t.img_w,H=im.naturalHeight||t.img_h;
 const wrap=document.getElementById('wrap');
 if(idx===-2){zoomTo();return;}
 if(idx<0){tile=-1;zoom=1;applyZoom();tilebar();return;}
 tile=idx; zoom=TILE_ZOOM; applyZoom();
 const cw=W/TCOLS,ch=H/TROWS;
 wrap.scrollLeft=Math.max(0,((idx%TCOLS)*cw-cw*OVERLAP)*zoom);
 wrap.scrollTop =Math.max(0,(Math.floor(idx/TCOLS)*ch-ch*OVERLAP)*zoom);
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
 t.ball_gt.forEach(b=>add('ballgt',b.bbox,'existing football GT','#4aa3ff'));
 t.siblings.forEach(b=>add('sib',b.bbox_xywh,
   'other proposal'+(b.verdict?' → '+b.verdict:''),'#7a6a2a'));
 add('cand',t.bbox_xywh,'JUDGE THIS  conf '+t.conf,'#ffd23d');
 panel();
}
function panel(){
 const t=cur(),st=document.getElementById('state');
 if(t.verdict===S.YES)
  st.innerHTML='<div class="ok">answered <b>YES</b> &mdash; recorded as a '
   +'human-approved addition. Answering again appends a new event.</div>';
 else if(t.verdict===S.NO)
  st.innerHTML='<div class="no">answered <b>NO</b> &mdash; the proposal was '
   +'rejected. Nothing is added.</div>';
 else if(t.verdict===S.UNSURE)
  st.innerHTML='<div class="uns">answered <b>UNSURE</b> &mdash; reported '
   +'separately, never counted as either.</div>';
 else st.innerHTML='<div class="note">not answered yet. The view opens zoomed '
   +'on the proposal; press FIT to see the whole frame.</div>';
 document.getElementById('meta').innerHTML=
  `conf <b>${t.conf}</b> · box
   <span style="font:11px monospace">${t.bbox_xywh.map(v=>Math.round(v)).join(', ')}</span>
   (${Math.round(t.bbox_xywh[2])}x${Math.round(t.bbox_xywh[3])} px)
   <div>run <b>${t.run}</b> · existing football GT in image
    <b>${t.ball_gt.length}</b> · proposals here <b>${t.siblings.length+1}</b></div>
   <div style="color:#666">model proposal &mdash; not an annotation</div>`;
 document.getElementById('hist').innerHTML=!t.history.length?'':
  '<b>history</b>'+t.history.map(h=>`<div>${(h.recorded_utc||'').slice(11,19)}
   ${h.verdict}</div>`).join('');
}
function stats(){
 const d=S.items.filter(x=>x.verdict).length;
 document.getElementById('bar').firstElementChild.style.width=
   (100*d/S.items.length)+'%';
 const y=S.items.filter(x=>x.verdict===S.YES).length;
 const n=S.items.filter(x=>x.verdict===S.NO).length;
 const u=S.items.filter(x=>x.verdict===S.UNSURE).length;
 document.getElementById('tally').textContent=`Y ${y} · N ${n} · U ${u}`;
 document.getElementById('rem').textContent=`${remaining()} remaining`;
}
async function answer(v){
 const t=cur();
 const r=await fetch('/api/verdict',{method:'POST',
  headers:{'Content-Type':'application/json'},
  body:JSON.stringify({candidate_id:t.candidate_id,verdict:v})});
 const j=await r.json();
 if(!r.ok){alert(j.error||'refused');return;}
 t.verdict=v;
 t.history.push({verdict:v,recorded_utc:j.recorded_utc});
 S.items.forEach(o=>o.siblings.forEach(s=>{
   if(s.candidate_id===t.candidate_id)s.verdict=v;}));
 draw(); stats(); next();
}
function next(){
 for(let k=i+1;k<S.items.length;k++){if(!S.items[k].verdict){i=k;render();return;}}
 for(let k=0;k<S.items.length;k++){if(!S.items[k].verdict){i=k;render();return;}}
 render();
}
function step(d){const n=i+d; if(n>=0&&n<S.items.length){i=n;render();}}
document.getElementById('aY').onclick=()=>answer(S.YES);
document.getElementById('aN').onclick=()=>answer(S.NO);
document.getElementById('aU').onclick=()=>answer(S.UNSURE);
document.getElementById('bNx').onclick=()=>step(1);
document.getElementById('bB').onclick=()=>step(-1);
document.getElementById('bJ').onclick=next;
document.onkeydown=e=>{
 const k=e.key.toLowerCase();
 if(k==='y')answer(S.YES);
 else if(k==='n')answer(S.NO);
 else if(k==='u')answer(S.UNSURE);
 else if(k==='m'||e.key==='ArrowRight'){e.preventDefault();step(1);}
 else if(k==='b'||e.key==='ArrowLeft'){e.preventDefault();step(-1);}
 else if(k==='j'){e.preventDefault();next();}
 else if(k==='t'){gotoTile(tile+1>=TCOLS*TROWS?-1:tile+1);}
 else if(k==='z'){zoomTo();}
 else if(e.key==='+'||e.key==='='){zoom=Math.min(zoom*1.25,12);applyZoom();}
 else if(e.key==='-'){zoom=Math.max(zoom/1.25,0.25);applyZoom();}
 else if(k==='0'){zoom=1;tile=-1;applyZoom();tilebar();}
};
boot();
</script></body></html>"""


def verdicts(decisions: Path = DECISIONS):
    """Effective verdict per candidate. Latest human answer wins."""
    per = {}
    for d in kb_decisions.read_log(decisions):
        if d.get('mode') == CAND_MODE:
            per.setdefault(d['candidate_id'], []).append(d)
    out = {}
    for cid, evs in per.items():
        evs = sorted(evs, key=lambda d: (d.get('recorded_utc') or '', d['_line']))
        out[cid] = {
            'verdict': evs[-1].get('verdict'),
            'recorded_utc': evs[-1].get('recorded_utc'),
            'history': [{'verdict': e.get('verdict'),
                         'recorded_utc': e.get('recorded_utc')} for e in evs],
        }
    return out


def approved_additions(decisions: Path = DECISIONS):
    """Candidates a human said YES to. The only ones that may become GT."""
    v = verdicts(decisions)
    q = json.loads(CAND.CANDIDATES.read_text(encoding='utf-8'))
    return [c for c in q['candidates']
            if v.get(c['candidate_id'], {}).get('verdict') == YES]


def build_state(show_context=False, decisions: Path = DECISIONS):
    q = json.loads(CAND.CANDIDATES.read_text(encoding='utf-8'))
    live = kb_sample.population_fingerprint()['population_sha256']
    if live != q['population']['population_sha256']:
        raise SystemExit('REFUSING: the export changed since these candidates '
                         'were generated. Regenerate rather than reviewing a '
                         'stale queue.')
    v = verdicts(decisions)
    gts = CAND.existing_ball_gt(
        [{'IMAGE': im, 'split': im.split('/')[0],
          'coco_image_id': cid} for im, cid in _coco_ids(q).items()])
    by_img = {}
    for c in q['candidates']:
        by_img.setdefault(c['IMAGE'], []).append(c)
    ctx = _context(q)
    items = []
    for c in q['candidates']:
        sibs = [{'candidate_id': s['candidate_id'], 'bbox_xywh': s['bbox_xywh'],
                 'verdict': v.get(s['candidate_id'], {}).get('verdict')}
                for s in by_img[c['IMAGE']]
                if s['candidate_id'] != c['candidate_id']]
        rec = v.get(c['candidate_id'], {})
        items.append({**c,
                      'ball_gt': [{'bbox': g['bbox_xywh']}
                                  for g in gts.get(c['IMAGE'], [])],
                      'context': ctx.get(c['IMAGE'], []),
                      'siblings': sibs,
                      'verdict': rec.get('verdict'),
                      'history': rec.get('history', [])})
    return {'items': items, 'show_context': show_context,
            'YES': YES, 'NO': NO, 'UNSURE': UNSURE,
            'source_log': kb_decisions.log_version(decisions)}


def _coco_ids(q):
    qq = json.loads(
        (PKG / 'BALL_PP_SWEEP_QUEUE.json').read_text(encoding='utf-8'))
    want = {c['IMAGE'] for c in q['candidates']}
    return {r['IMAGE']: r['coco_image_id'] for r in qq['images']
            if r['IMAGE'] in want}


def _context(q):
    qq = json.loads(
        (PKG / 'BALL_PP_SWEEP_QUEUE.json').read_text(encoding='utf-8'))
    rows = {r['IMAGE']: r for r in qq['images']}
    want = {c['IMAGE'] for c in q['candidates']}
    cache, out = {}, {}
    for im in want:
        r = rows[im]
        split = r['split']
        if split not in cache:
            doc = json.loads((kb_sample.EXPORT /
                              f'{split}_annotations.coco.json')
                             .read_text(encoding='utf-8'))
            bc = kb_sample._ball_category(doc['categories'])
            per = {}
            for a in doc['annotations']:
                if a['category_id'] != bc:
                    per.setdefault(a['image_id'], []).append(a)
            cache[split] = per
        out[im] = [{'bbox': a['bbox']}
                   for a in cache[split].get(r['coco_image_id'], [])]
    return out


def append(rec):
    with LOCK:
        with open(DECISIONS, 'a', encoding='utf-8') as fh:
            fh.write(json.dumps(rec) + '\n')
            fh.flush()


def build_id_info():
    sha = hashlib.sha256(PAGE.encode('utf-8')).hexdigest()
    return {'build': sha[:12], 'page_sha256': sha,
            'tool': 'kb_ball_candidate_server'}


class H(BaseHTTPRequestHandler):
    protocol_version = 'HTTP/1.1'
    SHOW_CONTEXT = False
    CANDS = {}

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
                              PAGE.replace('__BUILD__',
                                           build_id_info()['build'])
                              .encode('utf-8'), 'text/html; charset=utf-8')
        if p == '/api/build':
            return self._send(200, json.dumps(build_id_info()).encode())
        if p == '/api/state':
            return self._send(200, json.dumps(
                build_state(self.SHOW_CONTEXT)).encode())
        if p.startswith('/img/'):
            want = p[len('/img/'):]
            if want not in {c['IMAGE'] for c in self.CANDS.values()}:
                return self._send(403, json.dumps(
                    {'error': 'no candidate in this image', 'IMAGE': want}
                ).encode())
            try:
                body, ctype = kb_images.read(want)
            except kb_images.ImageError as e:
                return self._send(404, json.dumps(
                    {'error': str(e), 'IMAGE': want}).encode())
            return self._send(200, body, ctype)
        return self._send(404, b'not found', 'text/plain')

    def do_POST(self):
        if urlparse(self.path).path != '/api/verdict':
            return self._send(404, b'', 'text/plain')
        n = int(self.headers.get('Content-Length', 0))
        d = json.loads(self.rfile.read(n) or b'{}')
        cid = str(d.get('candidate_id', ''))
        if cid not in self.CANDS:
            return self._send(400, json.dumps(
                {'error': 'not a generated candidate', 'candidate_id': cid}
            ).encode())
        v = d.get('verdict')
        if v not in VERDICTS:
            return self._send(400, json.dumps(
                {'error': f'verdict must be one of {VERDICTS}'}).encode())
        c = self.CANDS[cid]
        now = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        rec = {
            'mode': CAND_MODE,
            'BOX_ID': cid, 'candidate_id': cid,
            'IMAGE': c['IMAGE'], 'split': c['split'],
            'verdict': v,
            'HUMAN_FINAL_CLASS': None,
            'proposed_bbox_xywh': c['bbox_xywh'],
            'model_conf': c['conf'],
            'proposal_source': 'best_A_960.pt ball @ conf 0.03',
            'ontology': 'ALL_VISIBLE_PHYSICAL_FOOTBALLS',
            'no_annotation_modified': True,
            'recorded_utc': now, 'author': 'human reviewer',
        }
        if v == YES:
            # the ONLY path that can later become an annotation, and it is
            # explicit: the geometry is the model's, the approval is a human's
            rec['evidence_class'] = 'HUMAN_APPROVED_ADDITION'
            rec['approved_bbox_xywh'] = c['bbox_xywh']
            rec['geometry_author'] = 'model proposal, human approved'
        else:
            rec['evidence_class'] = ('HUMAN_REJECTED_PROPOSAL' if v == NO
                                     else 'UNRESOLVED_PROPOSAL')
        append(rec)
        return self._send(200, json.dumps(
            {'ok': True, 'verdict': v, 'recorded_utc': now}).encode())


def main():
    sys.stdout.reconfigure(encoding='utf-8')
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--port', type=int, default=8747)
    ap.add_argument('--no-browser', action='store_true')
    ap.add_argument('--show-context', action='store_true')
    args = ap.parse_args()

    if not CAND.CANDIDATES.is_file():
        print(f'no candidate queue. Run:\n'
              f'  python tools/kb_ball_candidates.py --generate')
        sys.exit(1)
    q = json.loads(CAND.CANDIDATES.read_text(encoding='utf-8'))
    H.SHOW_CONTEXT = args.show_context
    H.CANDS = {c['candidate_id']: c for c in q['candidates']}

    ok, problems = kb_images.preflight(
        sorted({c['IMAGE'] for c in q['candidates']}))
    if not ok:
        print(f'REFUSING TO START: {len(problems)} image(s) unresolvable.')
        for im, why in problems[:10]:
            print(f'  {im}\n    {why}')
        sys.exit(1)

    v = verdicts()
    done = sum(1 for c in H.CANDS if c in v)
    y = sum(1 for c in H.CANDS if v.get(c, {}).get('verdict') == YES)
    print(f'BALL CANDIDATE REVIEW -- {len(H.CANDS)} model proposals across '
          f'{len({c["IMAGE"] for c in q["candidates"]})} images')
    print(f'{done} answered ({y} approved), {len(H.CANDS) - done} outstanding')
    print(f'\nproposals only. A YES records a HUMAN_APPROVED_ADDITION; nothing '
          f'is written to the dataset.')
    print('Completing this queue says the PROPOSALS were adjudicated. It does '
          'not certify the images:\na football the detector never proposed '
          'cannot appear here -- that is what the residual QA measures.')
    print('\nKeys: Y yes  N no  U unsure  M/-> next  B/<- prev  J next '
          'unanswered  Z zoom to candidate  T tiles')
    url = f'http://127.0.0.1:{args.port}/'
    print(f'\nreview at {url}   Ctrl-C to stop')
    if not args.no_browser:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    ThreadingHTTPServer(('127.0.0.1', args.port), H).serve_forever()


if __name__ == '__main__':
    main()
