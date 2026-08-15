// Drive the REAL served page through the uncertain-revisit queue.
// Fixture: 7 effective-uncertain boxes across different images, plus one box
// that WAS uncertain and has since been resolved -- it must never appear.
// Every POST is captured, never sent.
const fs = require('fs'), vm = require('vm');

const page = fs.readFileSync(process.argv[2], 'utf8');
const state = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const posts = [];

// 8 images, each with one candidate and one context box.
const N = 8;
state.items = [];
state.decisions = {};
state.manual = {};
state.manual_kind = {};
for (let n = 0; n < N; n++) {
  const img = `train/u_fixture_${n}.jpg`;
  state.items.push({
    IMAGE: img, run: 'fixture', img_w: 1000, img_h: 1000,
    candidates: [{ BOX_ID: `fx:cand${n}`, bbox: [10, 10, 50, 50],
                   proposed: 'player', score: 1.0 }],
    context: [{ BOX_ID: `fx:ctx${n}`, bbox: [500, 500, 50, 50],
                already: null, already_mode: null }],
  });
  state.decisions[`fx:cand${n}`] = n < 6 ? 'uncertain' : 'player';
}
// six candidates uncertain + one context box uncertain = 7 open items
state.manual['fx:ctx0'] = 'uncertain';
// and one that used to be uncertain but is resolved now -- must not appear
state.manual['fx:ctx1'] = 'referee';
state.u_open = [];
for (let n = 0; n < 6; n++)
  state.u_open.push({ BOX_ID: `fx:cand${n}`, IMAGE: `train/u_fixture_${n}.jpg` });
state.u_open.push({ BOX_ID: 'fx:ctx0', IMAGE: 'train/u_fixture_0.jpg' });
state.u_open.push({ BOX_ID: 'fx:ctx1', IMAGE: 'train/u_fixture_1.jpg' });  // stale
state.total_boxes = 6684;
state.total_images = state.items.length;

function mkEl(id) {
  const e = {
    id, style: {}, children: [], _html: '', _text: '',
    onclick: null, onload: null, disabled: false, className: '',
    appendChild(c) { this.children.push(c); return c; },
    getBoundingClientRect() { return { left: 0, top: 0, width: 1000, height: 1000 }; },
  };
  Object.defineProperty(e, 'innerHTML',
    { get() { return this._html; }, set(v) { this._html = v; this.children = []; } });
  Object.defineProperty(e, 'textContent',
    { get() { return this._text; }, set(v) { this._text = String(v); } });
  return e;
}
const els = {};
const document = {
  getElementById: id => (els[id] = els[id] || mkEl(id)),
  querySelector: () => (els.__q = els.__q || mkEl('q')),
  createElement: t => mkEl(t),
  onkeydown: null,
};
els['im'] = mkEl('im'); els['im'].clientWidth = 1000; els['im'].clientHeight = 1000;
const ctx = {
  document, window: {}, console, location: { search: '' },
  Image: function () { return mkEl('img'); },
  Date, Math, JSON, Object, Array, String, Number, RegExp, Promise,
  setTimeout, prompt: () => 'sandbox reason', confirm: () => true, alert: () => {},
  fetch: async (url, opt) => {
    if (url === '/api/state') return { json: async () => state, ok: true };
    posts.push(JSON.parse(opt.body));
    return { json: async () => ({ ok: true }), ok: true };
  },
};
ctx.globalThis = ctx;
vm.createContext(ctx);

const i0 = page.indexOf('<script');
const js = page.slice(page.indexOf('>', i0) + 1, page.lastIndexOf('</script>'));
const probe = `
globalThis.__get=()=>({i,sel,selKind,uMode,uAt,queue:uQueue().map(u=>u.BOX_ID)});
globalThis.__enter=()=>uEnter();
`;
vm.runInContext(js + probe, ctx, { filename: 'page.js' });
const G = () => ctx.__get();
const wait = () => new Promise(r => setTimeout(r, 40));
const press = k => document.onkeydown({ key: k, preventDefault() {} });
const hdr = () => ({
  mode: els['mode'].textContent, pos: els['pos'].textContent,
  imgs: els['imgs'].textContent, boxes: els['boxes'].textContent,
  rem: els['rem'].textContent,
});

(async () => {
  await wait(); await wait();
  const out = {};
  out.full_header = hdr();
  out.button = els['uJump'].textContent;

  ctx.__enter();
  await wait();
  out.queue = G().queue;
  out.revisit_header = hdr();
  out.first = { sel: G().sel, kind: G().selKind, image: state.items[G().i].IMAGE,
                at: G().uAt };
  out.panel = (els['urow'].innerHTML || '').replace(/\s+/g, ' ').slice(0, 110);

  // the selected box is the one carrying the highlight class
  els['im'].onload();
  out.highlight = els['ov'].children
    .filter(c => (c.className || '').includes('utarget')).length;

  // N / B stay inside the queue
  const seen = [G().sel];
  for (let n = 0; n < 8; n++) { press('n'); await wait(); seen.push(G().sel); }
  out.n_walk = seen;
  out.n_stayed_in_queue = seen.every(s => out.queue.includes(s));
  press('b'); await wait();
  out.after_b = G().sel;

  // answer one and watch the queue shrink
  ctx.__enter(); await wait();
  const target = G().sel;
  press('g'); await wait();
  out.answered = { box: target, post: posts[posts.length - 1] };
  out.after_answer = { queue: G().queue.length, header: hdr().pos,
                       advanced_to: G().sel };

  // answer the rest
  for (let n = 0; n < 10 && G().queue.length; n++) { press('m'); await wait(); }
  out.final = { queue: G().queue.length, banner_shown: els['ubanner'].style.display,
                banner: (els['ubanner'].innerHTML || '')
                  .replace(/\s+/g, ' ').slice(0, 90) };
  out.posts = posts.map(p => ({ box: p.BOX_ID, mode: p.mode,
                                v: p.HUMAN_FINAL_CLASS }));
  out.boxes_written = [...new Set(posts.map(p => p.BOX_ID))];

  // return to the full review
  els['uBack'].onclick();
  await wait();
  out.back_header = hdr();
  out.back_mode = G().uMode;
  console.log(JSON.stringify(out, null, 1));
})();
