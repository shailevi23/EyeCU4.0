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
    if (url === '/api/flag') {
      return {
        ok: true,
        json: async () => ({ ok: true,
                             flag_type: body.retract ? null : body.flag_type,
                             counts: { false: 1, bad_box: 0 },
                             recorded_utc: '2026-08-14T00:00:00Z' }),
      };
    }
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
globalThis.__get=()=>({i,zoom,tile,selGT,cycle});
globalThis.__cur=()=>cur();
globalThis.__set=o=>{if('i' in o)i=o.i;if('selGT' in o)selGT=o.selGT;};
globalThis.__overlay=()=>document.getElementById('ov').children.length;
globalThis.__click=(x,y)=>pick({clientX:x*zoom,clientY:y*zoom});
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
  press('z'); await wait();                      // centre on the object
  out.zoom = { fit: z0, on_object: G().zoom, tile: G().tile };
  press('0'); await wait();
  out.zoom.back_to_fit = G().zoom;

  // 7. every ontology post names an object that exists in the queue
  const onto = posts.filter(p => p.object_id);
  out.all_posted_in_queue = onto.every(
    p => state.items.some(o => o.object_id === p.object_id));
  out.roles_posted = [...new Set(onto.map(p => p.HUMAN_BALL_ROLE))].sort();

  // ---------------------------------------------------------- GT FLAGS
  const withGT = state.items.findIndex(o => o.ball_gt.length > 0);
  ctx.__set({ i: withGT, selGT: null });
  press('0'); await wait();
  const item = ctx.__cur();
  const gt = item.ball_gt[0];

  // 8. F/V/C do nothing with no GT selected, and never touch the ontology
  const n8 = posts.length;
  press('f'); await wait();
  press('v'); await wait();
  press('c'); await wait();
  out.flag_without_selection = {
    posted: posts.length - n8,
    alerted: !!ctx.__alert,
  };

  // 9. clicking selects the GT by geometry
  const cx = gt.bbox[0] + gt.bbox[2] / 2, cy = gt.bbox[1] + gt.bbox[3] / 2;
  ctx.__click(cx, cy); await wait();
  out.selection = { selGT: ctx.__get().selGT, expected: gt.BOX_ID,
                    matched: ctx.__get().selGT === gt.BOX_ID };

  // 10. F flags the SELECTED GT, not the magenta Round-0 object
  const n10 = posts.length;
  press('f'); await wait();
  const fp = posts[posts.length - 1];
  out.flag_false = {
    posted: posts.length - n10,
    BOX_ID: fp.BOX_ID, flag_type: fp.flag_type,
    targets_existing_gt: fp.BOX_ID === gt.BOX_ID,
    is_not_the_round0_object: fp.BOX_ID !== item.object_id
                              && !String(fp.BOX_ID).startsWith('BALLOBJ:'),
    carries_no_object_id: fp.object_id === undefined,
  };

  // 11. flagging must NOT advance the queue or answer the ontology object
  out.flag_is_inert = {
    index_unchanged: ctx.__get().i === withGT,
    role_still: ctx.__cur().role,
    ontology_posts_added: posts.filter(p => p.object_id).length - onto.length,
  };

  // 12. V then C
  press('v'); await wait();
  out.flag_bad_box = { flag_type: posts[posts.length - 1].flag_type };
  press('c'); await wait();
  out.flag_clear = { retract: posts[posts.length - 1].retract === true,
                     BOX_ID: posts[posts.length - 1].BOX_ID };

  // 13. repeated identical flags stay unambiguous: same target, same type
  const n13 = posts.length;
  press('f'); await wait();
  press('f'); await wait();
  const dup = posts.slice(n13);
  out.duplicate_flag = {
    count: dup.length,
    same_target: dup.every(p => p.BOX_ID === gt.BOX_ID),
    same_type: new Set(dup.map(p => p.flag_type)).size === 1,
  };

  // 14. overlapping / multiple GT: deterministic, smallest first, cycles
  const multi = state.items.findIndex(o => o.ball_gt.length > 1);
  if (multi >= 0) {
    ctx.__set({ i: multi, selGT: null });
    press('0'); await wait();
    const boxes = ctx.__cur().ball_gt;
    const areas = boxes.map(b => b.bbox[2] * b.bbox[3]);
    // click a point inside the largest box; the smallest containing box wins
    const big = boxes[areas.indexOf(Math.max(...areas))];
    const px = big.bbox[0] + big.bbox[2] / 2, py = big.bbox[1] + big.bbox[3] / 2;
    ctx.__click(px, py); await wait();
    const first = ctx.__get().selGT;
    ctx.__click(px, py); await wait();
    const second = ctx.__get().selGT;
    ctx.__set({ selGT: null });
    ctx.__click(px, py); await wait();
    out.overlap = {
      n_gt: boxes.length,
      first, second,
      deterministic: ctx.__get().selGT === first,
      cycles: boxes.filter(b =>
        px >= b.bbox[0] && px <= b.bbox[0] + b.bbox[2] &&
        py >= b.bbox[1] && py <= b.bbox[1] + b.bbox[3]).length > 1
        ? second !== first : second === first,
    };
  }

  // 15. clicking empty space clears the selection
  ctx.__set({ i: withGT });
  ctx.__click(gt.bbox[0] + gt.bbox[2] / 2, gt.bbox[1] + gt.bbox[3] / 2);
  await wait();
  const had = ctx.__get().selGT;
  ctx.__click(5, 715); await wait();
  out.click_empty_clears = { had: !!had, now: ctx.__get().selGT };

  out.flag_posts_all_target_existing_gt = posts
    .filter(p => p.BOX_ID !== undefined)
    .every(p => state.items.some(o =>
      o.ball_gt.some(g => g.BOX_ID === p.BOX_ID)));

  console.log(JSON.stringify(out, null, 1));
})();
