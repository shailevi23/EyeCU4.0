// Drive the REAL served page: one large box containing one small box, click
// inside both, and watch which BOX_ID the clicks resolve to. Nothing is sent.
const fs = require('fs'), vm = require('vm');

const page = fs.readFileSync(process.argv[2], 'utf8');
const state = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const posts = [];

// One synthetic image: a large context box with a small context box fully
// inside it, a candidate partially overlapping the large one, and a lone box.
const IMG = 'train/__overlap_fixture__.jpg';
state.items = [{
  IMAGE: IMG, run: 'fixture', img_w: 1000, img_h: 1000,
  candidates: [
    { BOX_ID: 'fx:cand_big', bbox: [100, 100, 400, 400], proposed: 'player', score: 1.0 },
    { BOX_ID: 'fx:cand_far', bbox: [800, 800, 100, 100], proposed: 'player', score: 0.5 },
  ],
  context: [
    { BOX_ID: 'fx:ctx_huge', bbox: [50, 50, 700, 700], already: null, already_mode: null },
    { BOX_ID: 'fx:ctx_small', bbox: [200, 200, 40, 40], already: null, already_mode: null },
    { BOX_ID: 'fx:ctx_done', bbox: [190, 190, 60, 60], already: 'referee', already_mode: 'candidates' },
  ],
}].concat(state.items.slice(0, 2));
state.decisions = {}; state.manual = {}; state.manual_kind = {};

function mkEl(id) {
  const e = {
    id, style: {}, children: [], _html: '', _text: '',
    onclick: null, onload: null, disabled: false,
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
  querySelector: () => mkEl('q'),
  createElement: t => mkEl(t),
  onkeydown: null,
};
// the image element reports its natural size, so image px == client px
els['im'] = mkEl('im'); els['im'].clientWidth = 1000; els['im'].clientHeight = 1000;

const ctx = {
  document, window: {}, console, location: { search: '' },
  Image: function () { return mkEl('img'); },
  Date, Math, JSON, Object, Array, String, Number, RegExp, Promise,
  setTimeout, prompt: () => 'sandbox reason',
  fetch: async (url, opt) => {
    if (url === '/api/state') return { json: async () => state, ok: true };
    posts.push(JSON.parse(opt.body));
    return { json: async () => ({ ok: true, manual_kind: 'DISPOSITION_SET' }), ok: true };
  },
};
ctx.globalThis = ctx;
vm.createContext(ctx);

const i0 = page.indexOf('<script');
const js = page.slice(page.indexOf('>', i0) + 1, page.lastIndexOf('</script>'));
const probe = `
globalThis.__get=()=>({i,sel,selKind,qMode,ovlp:ovlp.map(b=>b.BOX_ID),ovlpAt});
globalThis.__setI=n=>{i=n;};
`;
vm.runInContext(js + probe, ctx, { filename: 'page.js' });
const G = () => ctx.__get();
const wait = () => new Promise(r => setTimeout(r, 60));
const click = (x, y) => els['ov'].onclick({ clientX: x, clientY: y });
const press = (key, shift) =>
  document.onkeydown({ key, shiftKey: !!shift, preventDefault() {} });

(async () => {
  await wait(); await wait();
  ctx.__setI(0);
  els['im'].onload();                     // force a draw of the fixture image
  const out = {};

  // (220,220) is inside all four: ctx_small, ctx_done, cand_big, ctx_huge
  out.group = (click(220, 220), G().ovlp);
  out.cycle = [G().sel];
  for (let n = 0; n < 4; n++) { click(220, 220); out.cycle.push(G().sel); }
  // the exact case from the report: one large box containing one small box.
  // (120,120) is inside cand_big, which is wholly inside ctx_huge.
  click(0, 0);
  const two = [];
  click(120, 120); two.push(G().sel);
  click(120, 120); two.push(G().sel);
  click(120, 120); two.push(G().sel);
  out.small_inside_large = { group: G().ovlp, sequence: two };

  // a fresh click on the small box, then read the selection kind
  click(0, 0); click(220, 220);
  out.first_selection = { sel: G().sel, kind: G().selKind, at: G().ovlpAt };

  // M applies to the CURRENTLY selected box only
  press('m'); await wait();
  out.m_post = posts[posts.length - 1];

  // P/G/R/U behave the same on the same selection
  out.roles = {};
  for (const [k, v] of [['p', 'player'], ['g', 'goalkeeper'],
                        ['r', 'referee'], ['u', 'uncertain']]) {
    click(0, 0); click(220, 220);
    press(k); await wait();
    const p = posts[posts.length - 1];
    out.roles[v] = { box: p.BOX_ID, mode: p.mode, value: p.HUMAN_FINAL_CLASS };
  }

  // Tab / Shift+Tab cycle without moving the mouse
  click(0, 0); click(220, 220);
  const t = [G().sel];
  press('Tab'); t.push(G().sel);
  press('Tab'); t.push(G().sel);
  press('Tab', true); t.push(G().sel);
  out.tab = t;

  // a point inside exactly one box selects it with no overlap UI
  click(0, 0); click(850, 850);
  out.single = { sel: G().sel, group: G().ovlp, panel: els['ovbox'].style.display };

  // partial overlap: (120,120) is in ctx_huge and cand_big but not the small ones
  click(0, 0); click(120, 120);
  out.partial = { group: G().ovlp, sel: G().sel };

  // 1-9 still works and clears the cycle
  click(220, 220);
  press('2');
  out.numeric = { sel: G().sel, kind: G().selKind, group: G().ovlp };

  // Q mode must not select anything
  click(0, 0);
  press('q');
  const before = G().sel;
  click(220, 220);
  out.qmode = { selected_changed: G().sel !== before, qMode: G().qMode,
                group: G().ovlp };
  press('Escape');

  // overlap panel text
  click(0, 0); click(220, 220);
  out.panel = { display: els['ovbox'].style.display,
                html: (els['ovbox'].innerHTML || '').replace(/\s+/g, ' ').slice(0, 60),
                buttons: els['ovbox'].children.length };
  out.selinfo = (els['selinfo'].innerHTML || '').replace(/\s+/g, ' ').slice(0, 90);

  out.posts_total = posts.length;
  out.post_boxes = [...new Set(posts.map(p => p.BOX_ID))];
  console.log(JSON.stringify(out, null, 1));
})();
