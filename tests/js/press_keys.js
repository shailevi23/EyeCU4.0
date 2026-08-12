// Execute the REAL served page script under a minimal DOM and press keys.
// The bug was that the M branch did not exist in the handler; only actually
// dispatching the key can prove it now does. Every POST is captured, never sent.
const fs = require('fs'), vm = require('vm');

const page = fs.readFileSync(process.argv[2], 'utf8');
const state = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const posts = [];

function mkEl(id) {
  const e = {
    id, style: {}, dataset: {}, children: [],
    _html: '', onclick: null, onload: null, disabled: false,
    appendChild(c) { this.children.push(c); return c; },
    getContext() { return null; },
  };
  Object.defineProperty(e, 'innerHTML', {
    get() { return this._html; },
    set(v) { this._html = v; this.children = []; },
  });
  Object.defineProperty(e, 'textContent', {
    get() { return this._text; },
    set(v) { this._text = String(v); },
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
  document, window: {}, console,
  Image: function () { return mkEl('img'); },
  Date, Math, JSON, Object, Array, String, Number, RegExp, Promise,
  setTimeout, prompt: () => 'sandbox reason',
  fetch: async (url, opt) => {
    if (url === '/api/state') {
      return { json: async () => state, ok: true };
    }
    posts.push(JSON.parse(opt.body));
    return { json: async () => ({ ok: true, manual_kind: 'DISPOSITION_SET' }), ok: true };
  },
};
ctx.globalThis = ctx;
vm.createContext(ctx);

const i = page.indexOf('<script');
const js = page.slice(page.indexOf('>', i) + 1, page.lastIndexOf('</script>'));
// top-level `let` is not a property of the vm context, so expose a probe
const probe = `
globalThis.__get=()=>({i,sel,selKind,qMode});
globalThis.__set=o=>{if('sel' in o)sel=o.sel;if('selKind' in o)selKind=o.selKind;};
`;
vm.runInContext(js + probe, ctx, { filename: 'page.js' });
const G = () => ctx.__get();
const SET = o => ctx.__set(o);

const press = k => document.onkeydown({ key: k, preventDefault() {} });
const wait = () => new Promise(r => setTimeout(r, 60));

(async () => {
  await wait(); await wait();               // let boot() resolve
  const out = { posts: [], counters: {} };
  const counter = id => els[id] && els[id].textContent;
  out.counters.before = counter('cM');

  // 1. M on the required candidate the UI selected itself
  const before = G().sel;
  const imgAtStart = G().i;
  press('m'); await wait();
  out.posts.push({ step: 'M on required candidate', selected_before: before,
                   post: posts[posts.length - 1] });
  out.counters.after_candidate = counter('cM');

  // 2. M on a clicked black context box
  const item = state.items[G().i];
  const ctxBox = item.context.length ? item.context[0].BOX_ID : null;
  if (ctxBox) {
    SET({ sel: ctxBox, selKind: 'ctx' });
    press('m'); await wait();
    out.posts.push({ step: 'M on context box', selected_before: ctxBox,
                     post: posts[posts.length - 1] });
    out.counters.after_context = counter('cM');
  }

  // 3. the visible button, same effect as the key
  const n0 = posts.length;
  SET({ sel: before, selKind: 'cand' });
  els['nonact'].onclick(); await wait();
  out.button = { wired: typeof els['nonact'].onclick === 'function',
                 posted: posts.length > n0,
                 post: posts[posts.length - 1] };

  // 4. M must not advance to the next image
  out.image_index_unchanged = G().i === imgAtStart;

  // 5. every other key still works
  const keymap = {};
  for (const k of ['p', 'g', 'r', 'u']) {
    SET({ sel: before, selKind: 'cand' });
    const n = posts.length; press(k); await wait();
    keymap[k] = posts.length > n ? posts[posts.length - 1].HUMAN_FINAL_CLASS : null;
  }
  const iBefore = G().i;
  press('n'); await wait();
  keymap['n = next image'] = G().i !== iBefore ? 'advanced' : 'no further unresolved';
  press('b'); await wait();
  keymap['b = previous'] = 'i=' + G().i;
  out.other_keys = keymap;

  // 6. M during the Q flag prompt must cancel, never post a bad value
  press('q'); await wait();
  const nq = posts.length;
  press('m'); await wait();
  out.m_during_q = { posted_anything: posts.length > nq, qMode_after: G().qMode,
                     banner: els['qbanner'].style.display };

  console.log(JSON.stringify(out, null, 1));
})();
