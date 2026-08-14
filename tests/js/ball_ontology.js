// Drive the REAL ontology page. The keyboard map is the whole risk here: the
// brief warned that 'n' reads as both NEXT and NON-ACTIVE, and a wrong binding
// would silently mislabel real findings rather than fail loudly. So every key
// is pressed against the real handler and what it posted is recorded.
const fs = require('fs'), vm = require('vm');

const page = fs.readFileSync(process.argv[2], 'utf8');
const state = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const posts = [];

function mkEl(id) {
  const e = {
    id, style: {}, dataset: {}, children: [], className: '',
    _html: '', _text: '', onclick: null, onload: null, onerror: null,
    disabled: false, scrollLeft: 0, scrollTop: 0,
    clientWidth: 900, clientHeight: 600,
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
      json: async () => ({ ok: true, HUMAN_BALL_ROLE: body.HUMAN_BALL_ROLE,
                           recorded_utc: '2026-08-14T00:00:00Z' }),
    };
  },
};
ctx.globalThis = ctx;
vm.createContext(ctx);

const i = page.indexOf('<script');
const js = page.slice(page.indexOf('>', i) + 1, page.lastIndexOf('</script>'));
const probe = `
globalThis.__get=()=>({i,zoom,tile});
globalThis.__cur=()=>cur();
globalThis.__set=o=>{if('i' in o)i=o.i;};
globalThis.__overlay=()=>document.getElementById('ov').children.length;
`;
vm.runInContext(js + probe, ctx, { filename: 'onto.js' });
const G = () => ctx.__get();
const press = k => document.onkeydown({ key: k, preventDefault() {} });
const wait = () => new Promise(r => setTimeout(r, 50));

(async () => {
  await wait(); await wait();
  const out = { keys: {} };
  const ACTIVE = state.ACTIVE, NON = state.NON_ACTIVE, UNS = state.UNSURE;

  // 1. every classification key, checked by what it actually posted
  const tryKey = async (key) => {
    ctx.__set({ i: 0 });
    const before = posts.length;
    press(key); await wait();
    return posts.length > before
      ? posts[posts.length - 1].HUMAN_BALL_ROLE : null;
  };
  out.keys.a = await tryKey('a');
  out.keys.x = await tryKey('x');
  out.keys.u = await tryKey('u');

  // 2. 'n' must post NOTHING. It reads as both NEXT and NON-ACTIVE.
  out.keys.n = await tryKey('n');

  // 3. navigation keys must not classify
  ctx.__set({ i: 3 });
  const nBefore = posts.length;
  press('j'); await wait();
  const afterJ = G().i;
  press('b'); await wait();
  press('k'); await wait();
  out.navigation = {
    posted_anything: posts.length > nBefore,
    moved_on_j: afterJ !== 3,
    index_now: G().i,
  };

  // 4. the object under review is a specific box, and siblings stay separate
  const withSibs = state.items.findIndex(o => o.siblings.length > 0);
  ctx.__set({ i: withSibs });
  press('0'); await wait();
  const cur = ctx.__cur();
  out.multi_object_image = {
    object_id: cur.object_id,
    siblings: cur.siblings.length,
    sibling_ids_differ: cur.siblings.every(s => s.object_id !== cur.object_id),
    overlay_boxes: ctx.__overlay(),
  };

  // classifying one object in a multi-object image must post only that one
  const n2 = posts.length;
  press('x'); await wait();
  out.multi_object_image.posted = posts.slice(n2).map(p => p.object_id);

  // 5. two objects in the SAME image are independently classifiable
  const sibId = cur.siblings[0].object_id;
  const sibIdx = state.items.findIndex(o => o.object_id === sibId);
  ctx.__set({ i: sibIdx });
  const n3 = posts.length;
  press('a'); await wait();
  out.independent = {
    same_image: state.items[sibIdx].IMAGE === cur.IMAGE,
    posted_id: posts[posts.length - 1].object_id,
    distinct_from_first: posts[posts.length - 1].object_id !== cur.object_id,
    roles: [posts[n2].HUMAN_BALL_ROLE, posts[n3].HUMAN_BALL_ROLE],
  };

  // 6. zoom and the BALL jump
  ctx.__set({ i: 0 });
  press('0'); await wait();
  const z0 = G().zoom;
  press('c'); await wait();                      // centre on the object
  out.zoom = { fit: z0, on_object: G().zoom, tile: G().tile };
  press('0'); await wait();
  out.zoom.back_to_fit = G().zoom;

  // 7. every post names an object that exists in the queue
  out.all_posted_in_queue = posts.every(
    p => state.items.some(o => o.object_id === p.object_id));
  out.roles_posted = [...new Set(posts.map(p => p.HUMAN_BALL_ROLE))].sort();

  console.log(JSON.stringify(out, null, 1));
})();
