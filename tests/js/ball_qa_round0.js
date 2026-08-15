// Drive the REAL Round-0 page script under a minimal DOM.
// The behaviours that matter here cannot be checked by reading Python: that one
// image with three drawn balls posts THREE objects but stays ONE positive, that
// a ~4 px box survives being drawn at 4x zoom, and that tile navigation changes
// nothing but the viewport. So the real page is executed and real events fired.
const fs = require('fs'), vm = require('vm');

const page = fs.readFileSync(process.argv[2], 'utf8');
const state = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const posts = [];

function mkEl(id) {
  const e = {
    id, style: {}, dataset: {}, children: [], className: '',
    _html: '', _text: '', onclick: null, onload: null, onerror: null,
    disabled: false, scrollLeft: 0, scrollTop: 0,
    naturalWidth: 1280, naturalHeight: 720,
    appendChild(c) { this.children.push(c); return c; },
    getBoundingClientRect() { return { left: 0, top: 0 }; },
  };
  Object.defineProperty(e, 'innerHTML', {
    get() { return this._html; },
    set(v) { this._html = v; this.children = []; },
  });
  Object.defineProperty(e, 'textContent', {
    get() { return this._text; },
    set(v) { this._text = String(v); },
  });
  Object.defineProperty(e, 'firstElementChild', {
    get() { return this.children[0] || mkEl('first'); },
  });
  return e;
}
const els = {};
const document = {
  getElementById: id => (els[id] = els[id] || mkEl(id)),
  querySelector: () => mkEl('q'),
  createElement: t => mkEl(t),
  onkeydown: null,
};
const ctx = {
  document, window: {}, console, location: { search: '' },
  Date, Math, JSON, Object, Array, String, Number, RegExp, Promise,
  setTimeout, alert: m => { ctx.__alert = m; },
  fetch: async (url, opt) => {
    if (url === '/api/state') return { json: async () => state, ok: true };
    if (!opt) return { json: async () => ({}), ok: true };
    const body = JSON.parse(opt.body);
    posts.push(body);
    return {
      ok: true,
      json: async () => ({
        ok: true,
        missing: (body.missing_balls_xywh || []).map(b => ({ bbox_xywh: b })),
        recorded_utc: '2026-08-14T00:00:00Z',
      }),
    };
  },
};
ctx.globalThis = ctx;
vm.createContext(ctx);

const i = page.indexOf('<script');
const js = page.slice(page.indexOf('>', i) + 1, page.lastIndexOf('</script>'));
const probe = `
globalThis.__get=()=>({i,mode,drawn:drawn.slice(),zoom,tile,
                       seenCount:Object.keys(seen[cur().IMAGE]||{}).length});
globalThis.__cur=()=>cur();
`;
vm.runInContext(js + probe, ctx, { filename: 'ballqa.js' });
const G = () => ctx.__get();
const press = k => document.onkeydown({ key: k, preventDefault() {} });
const wait = () => new Promise(r => setTimeout(r, 50));

// a real drag on the hit layer, in SCREEN pixels; the page divides by zoom
const dragBox = (x, y, w, h) => {
  const hit = els['hit'], z = G().zoom;
  hit.onmousedown({ clientX: x * z, clientY: y * z });
  hit.onmousemove({ clientX: (x + w) * z, clientY: (y + h) * z });
  hit.onmouseup({ clientX: (x + w) * z, clientY: (y + h) * z });
};

(async () => {
  await wait(); await wait();
  const out = {};

  // 1. NO MISSING BALL posts a negative answer and advances
  const first = ctx.__cur().IMAGE;
  press('1'); await wait();
  out.negative = { post: posts[posts.length - 1], advanced: G().i === 1 };

  // 2. MISSING BALL enters draw mode WITHOUT posting anything yet
  const nBefore = posts.length;
  press('2'); await wait();
  out.enters_draw_mode = { mode: G().mode, posted_prematurely: posts.length > nBefore };

  // 3. three balls drawn on ONE image -> three objects, one positive image
  dragBox(100, 100, 9, 9);
  dragBox(400, 300, 12, 11);
  dragBox(800, 500, 4, 4);           // tiny, the case that matters
  out.drawn_count = G().drawn.length;
  press('Enter'); await wait();
  const p = posts[posts.length - 1];
  out.multi_object = {
    answer: p.answer,
    objects_posted: p.missing_balls_xywh.length,
    boxes: p.missing_balls_xywh.map(b => b.map(v => Math.round(v * 100) / 100)),
    one_image_one_answer: p.answer === 'MISSING_BALL' && !Array.isArray(p.answer),
  };

  // 4. a ~4 px box drawn at 4x zoom must round-trip to ~4 px in IMAGE pixels
  press('2'); await wait();
  ctx.__get(); // ensure state
  press('t'); await wait();                 // tile 1 -> zoom 4x
  const zoomAtDraw = G().zoom;
  dragBox(640, 360, 4.2, 3.8);
  const tiny = G().drawn[G().drawn.length - 1];
  out.tiny_at_zoom = {
    zoom: zoomAtDraw,
    drawn_image_px: tiny.map(v => Math.round(v * 100) / 100),
    width_ok: Math.abs(tiny[2] - 4.2) < 0.01,
    height_ok: Math.abs(tiny[3] - 3.8) < 0.01,
  };

  // 5. undo removes only the last
  const before = G().drawn.length;
  press('z'); await wait();
  out.undo = { before, after: G().drawn.length };

  // 6. tile navigation must not touch the drawing or the answer
  const drawnBefore = JSON.stringify(G().drawn);
  const curBefore = ctx.__cur().IMAGE;
  press('t'); press('t'); await wait();
  out.tiles_are_inert = {
    drawn_unchanged: JSON.stringify(G().drawn) === drawnBefore,
    image_unchanged: ctx.__cur().IMAGE === curBefore,
    tiles_visited: G().seenCount,
    zoom: G().zoom,
  };
  press('0'); await wait();
  out.fit_returns = { zoom: G().zoom, tile: G().tile };

  // 7. UNSURE posts and stays unresolved
  press('escape'); await wait();
  press('3'); await wait();
  out.unsure = { post: posts[posts.length - 1] };

  // 8. navigation stays inside the sample
  let guard = 0;
  while (guard++ < 400) press('n');
  out.navigation = {
    index: G().i, sample_size: state.items.length,
    stayed_in_sample: G().i >= 0 && G().i < state.items.length,
  };
  for (let k = 0; k < 400; k++) press('b');
  out.navigation.after_many_back = G().i;

  out.all_posted_images_in_sample = posts.every(
    q => state.items.some(it => it.IMAGE === q.IMAGE));

  console.log(JSON.stringify(out, null, 1));
})();
