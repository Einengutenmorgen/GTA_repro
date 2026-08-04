/* GTA Viewer -- client. Vanilla, no build step, no CDN. */
'use strict';

// =========================================================================
// tiny DOM helpers
// =========================================================================
const $ = (sel, root = document) => root.querySelector(sel);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === 'class') n.className = v;
    else if (k === 'html') n.innerHTML = v;
    else if (k.startsWith('on')) n.addEventListener(k.slice(2), v);
    else if (k === 'style' && typeof v === 'object') Object.assign(n.style, v);
    else n.setAttribute(k, v === true ? '' : v);
  }
  for (const kid of kids.flat(9)) {
    if (kid == null || kid === false) continue;
    n.appendChild(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return n;
};
const esc = s => String(s ?? '').replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const fmtBytes = n => n < 1024 ? n + ' B' : n < 1048576 ? (n / 1024).toFixed(0) + ' KB' : (n / 1048576).toFixed(1) + ' MB';
const norm = s => String(s ?? '').toLowerCase().replace(/[‐-―]/g, '-')
  .replace(/[^a-z0-9]+/g, ' ').trim();
const pct = x => (100 * (x || 0)).toFixed(0) + '%';

const SERIES = ['--series-1', '--series-2', '--series-3', '--series-4',
                '--series-5', '--series-6', '--series-7', '--series-8'];
const seriesVar = i => `var(${SERIES[i % SERIES.length]})`;

// stable color for a category name (identity follows the entity, not its rank)
const colorRegistry = new Map();
function catColor(name) {
  const k = norm(name);
  if (!colorRegistry.has(k)) colorRegistry.set(k, seriesVar(colorRegistry.size));
  return colorRegistry.get(k);
}

const ROLE_COLOR = { Protagonist: 'var(--series-3)', Antagonist: 'var(--series-8)', Innocent: 'var(--series-1)' };

// =========================================================================
// state
// =========================================================================
const S = {
  index: null,
  runId: null,
  manifest: null,
  tab: null,
  cache: new Map(),          // `${runId}::${file}` -> data
  semeval: { root: null, lang: null, article: null, data: null },
  ui: {},                    // per-tab scratch (filters, selections)
};

async function api(path, params) {
  const u = new URL(path, location.origin);
  for (const [k, v] of Object.entries(params || {})) u.searchParams.set(k, v);
  const r = await fetch(u);
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return r.json();
}

async function artifact(name) {
  const key = `${S.runId}::${name}`;
  if (S.cache.has(key)) return S.cache.get(key);
  const res = await api('/api/artifact', { run: S.runId, name });
  const val = res.kind === 'json' ? res.data : (res.text ?? res);
  S.cache.set(key, val);
  return val;
}
const has = f => (S.manifest?.files || []).includes(f);
const firstOf = (...names) => names.find(has) || null;

/**
 * Normalize the several code-row shapes this repo has emitted over time into
 * one: {open_code, text_passage, chunk_id, source_id, in_vivo}.
 *  - current : {open_code, text_passage, chunk_id: "<source>_c0007"}
 *  - legacy  : {chunk_id: 1, codes: "a, b, c"}   (one row per chunk, no passages)
 */
function normalizeCodes(raw) {
  const out = [];
  for (const r of Array.isArray(raw) ? raw : []) {
    if (!r || typeof r !== 'object' || r.__status__) continue;
    const chunk_id = r.chunk_id != null ? String(r.chunk_id) : '(no chunk)';
    const source_id = r.source_id != null ? String(r.source_id)
      : chunk_id.replace(/_c\d+$/, '');
    const label = r.open_code ?? r.code ?? r.label ?? r.name;
    const passage = r.text_passage ?? r.passage ?? r.quote ?? r.evidence ?? '';
    if (label != null) {
      out.push({ ...r, open_code: String(label), text_passage: String(passage ?? ''),
        chunk_id, source_id });
      continue;
    }
    // legacy: a single comma-joined string of labels, no passages
    if (typeof r.codes === 'string') {
      for (const piece of r.codes.split(/\s*[,;]\s*/).filter(Boolean)) {
        out.push({ open_code: piece, text_passage: '', chunk_id, source_id,
          __legacy__: true });
      }
      continue;
    }
    if (Array.isArray(r.codes)) {
      for (const piece of r.codes) {
        if (typeof piece === 'string') out.push({ open_code: piece, text_passage: '', chunk_id, source_id, __legacy__: true });
        else if (piece && typeof piece === 'object') out.push({
          ...piece, open_code: String(piece.open_code ?? piece.code ?? ''),
          text_passage: String(piece.text_passage ?? ''), chunk_id, source_id });
      }
    }
  }
  return out;
}

async function loadCodes() {
  const f = firstOf('output_initial_codes.json', 'output_open_codes.json');
  if (!f) return { file: null, codes: [], failed: [] };
  const raw = (await artifact(f)) || [];
  const failed = (Array.isArray(raw) ? raw : []).filter(c => c && c.__status__);
  return { file: f, codes: normalizeCodes(raw), failed };
}

// =========================================================================
// passage location -- find an LLM-quoted passage inside the source text
// =========================================================================
function buildNormMap(text) {
  // collapse whitespace + lowercase, keeping a map back to original indices
  let out = '', map = [], prevSpace = true;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (/\s/.test(c)) { if (!prevSpace) { out += ' '; map.push(i); prevSpace = true; } continue; }
    prevSpace = false;
    out += c.toLowerCase().replace(/[‘’]/, "'").replace(/[“”]/, '"')
            .replace(/[‐-―]/, '-');
    map.push(i);
  }
  return { norm: out, map };
}

/** Locate `passage` in `text`. Returns [{start,end,exact}] (end exclusive). */
function locatePassage(text, passage) {
  if (!text || !passage) return [];
  const { norm: nt, map } = buildNormMap(text);
  const toOrig = (a, b) => ({ start: map[a], end: (map[b - 1] ?? map[a]) + 1 });

  const tryFind = (needle) => {
    if (needle.length < 8) return null;
    const i = nt.indexOf(needle);
    if (i < 0) return null;
    return toOrig(i, i + needle.length);
  };

  const np = buildNormMap(passage).norm.trim();
  let hit = tryFind(np);
  if (hit) return [{ ...hit, exact: true }];

  // ellipsis-joined quotes: locate each fragment on its own
  const frags = np.split(/\s*(?:\.\.\.|…)\s*/).map(s => s.trim()).filter(s => s.length >= 10);
  if (frags.length > 1) {
    const spans = frags.map(tryFind).filter(Boolean).map(s => ({ ...s, exact: false }));
    if (spans.length) return spans;
  }

  // longest anchor: shrink from both ends by whole words until something lands
  const words = np.split(' ');
  for (let len = words.length; len >= 4; len--) {
    for (let st = 0; st + len <= words.length; st++) {
      const cand = words.slice(st, st + len).join(' ');
      if (cand.length < 14) continue;
      const h = tryFind(cand);
      if (h) return [{ ...h, exact: false }];
    }
  }
  return [];
}

/** Render `text` with non-overlapping highlight spans. */
function renderAnnotated(text, anns, onClick) {
  // anns: [{start,end,color,label,payload,exact}] -- resolve overlaps by keeping
  // the earliest-starting, then longest; later overlappers get truncated.
  const sorted = anns.slice().sort((a, b) => a.start - b.start || b.end - a.end);
  const kept = [];
  let cursor = 0;
  for (const a of sorted) {
    const start = Math.max(a.start, cursor);
    if (start >= a.end) { kept.push({ ...a, clipped: true, start: a.start, end: a.start }); continue; }
    kept.push({ ...a, start, end: a.end });
    cursor = a.end;
  }
  const frag = document.createDocumentFragment();
  let pos = 0;
  for (const a of kept) {
    if (a.end <= a.start) continue;
    if (a.start > pos) frag.appendChild(document.createTextNode(text.slice(pos, a.start)));
    const m = el('mark', {
      class: 'hl' + (a.exact === false ? ' approx' : ''),
      style: { '--hl': a.color },
      title: a.title || a.label || '',
      'data-key': a.key || '',
      onclick: () => onClick && onClick(a),
    }, text.slice(a.start, a.end));
    if (a.sup) m.appendChild(el('span', { class: 'tagsup' }, a.sup));
    frag.appendChild(m);
    pos = a.end;
  }
  if (pos < text.length) frag.appendChild(document.createTextNode(text.slice(pos)));
  return frag;
}

// =========================================================================
// markdown (small subset -- enough for output_final_theory.md)
// =========================================================================
function md(src) {
  const inline = s => esc(s)
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/(^|\W)\*([^*]+)\*(?=\W|$)/g, '$1<em>$2</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>');
  const out = [];
  let list = null;
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  for (const raw of String(src || '').split('\n')) {
    const line = raw.replace(/\s+$/, '');
    let m;
    if (!line.trim()) { closeList(); continue; }
    if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) { closeList(); out.push('<hr>'); continue; }
    if ((m = line.match(/^(#{1,6})\s+(.*)$/))) {
      closeList();
      const lv = Math.min(m[1].length, 3);
      out.push(`<h${lv}>${inline(m[2])}</h${lv}>`); continue;
    }
    if ((m = line.match(/^\s*[-*+]\s+(.*)$/))) {
      if (list !== 'ul') { closeList(); out.push('<ul>'); list = 'ul'; }
      out.push(`<li>${inline(m[1])}</li>`); continue;
    }
    if ((m = line.match(/^\s*\d+[.)]\s+(.*)$/))) {
      if (list !== 'ol') { closeList(); out.push('<ol>'); list = 'ol'; }
      out.push(`<li>${inline(m[1])}</li>`); continue;
    }
    closeList();
    out.push(`<p>${inline(line)}</p>`);
  }
  closeList();
  return out.join('\n');
}

// =========================================================================
// shared widgets
// =========================================================================
const tile = (v, k, cls) => el('div', { class: 'tile ' + (cls || '') },
  el('div', { class: 'v' }, v), el('div', { class: 'k' }, k));

const tiles = (...items) => el('div', { class: 'tiles' }, items.filter(Boolean));

const tag = (type, text) => el('span', { class: 'tag t-' + String(type).replace(/\W+/g, '_') },
  el('span', { class: 'g' }), text ?? type);

function barRows(rows, opts = {}) {
  const max = Math.max(1, ...rows.map(r => r.value));
  return el('div', {}, rows.map(r => el('div', { class: 'barrow ' + (r.cls || ''), title: r.label },
    el('div', { class: 'bl' }, r.label),
    el('div', { class: 'bt' }, el('div', {
      class: 'bf',
      style: { width: (100 * r.value / max) + '%', background: r.color || (opts.color || 'var(--accent)') },
    })),
    el('div', { class: 'bv' }, opts.fmt ? opts.fmt(r.value) : r.value),
  )));
}

function kv(pairs) {
  const d = el('dl', { class: 'kv' });
  for (const [k, v] of pairs) {
    if (v == null || v === '') continue;
    d.appendChild(el('dt', {}, k));
    d.appendChild(el('dd', {}, v instanceof Node ? v : String(v)));
  }
  return d;
}

const jsonBlock = obj => el('pre', { class: 'json' }, JSON.stringify(obj, null, 2));

function rawDetails(obj, label = 'raw JSON') {
  return el('details', { class: 'raw' }, el('summary', {}, label), jsonBlock(obj));
}

// =========================================================================
// inspector
// =========================================================================
function inspect(title, sub, bodyNode) {
  $('#insp-title').textContent = title;
  $('#insp-sub').textContent = sub || '';
  const b = $('#insp-body'); b.innerHTML = '';
  b.appendChild(bodyNode);
  $('#app').classList.add('with-inspector');
  b.scrollTop = 0;
}
function closeInspector() { $('#app').classList.remove('with-inspector'); }

// =========================================================================
// rail
// =========================================================================
function renderRail() {
  const host = $('#rail-scroll'); host.innerHTML = '';
  const byCollection = new Map();
  for (const r of S.index.runs) {
    if (!byCollection.has(r.collection)) byCollection.set(r.collection, []);
    byCollection.get(r.collection).push(r);
  }
  for (const [coll, runs] of byCollection) {
    const g = el('div', { class: 'rail-group' }, el('h2', {}, coll));
    for (const r of runs) {
      const st = r.stats || {};
      g.appendChild(el('button', {
        class: 'run-item' + (S.runId === r.id ? ' active' : ''),
        onclick: () => selectRun(r.id),
      },
        el('div', { class: 'rname' }, r.name),
        el('div', { class: 'rmeta' },
          el('span', { class: 'pill arm-' + r.arm }, el('span', { class: 'dot' }), r.arm),
          r.timestamp && el('span', {}, r.timestamp.slice(0, 10)),
          st.codes != null && el('span', {}, st.codes + ' codes'),
          st.categories != null && el('span', {}, st.categories + ' cats'),
        ),
      ));
    }
    host.appendChild(g);
  }

  // SemEval corpora
  for (const sv of S.index.semeval || []) {
    const g = el('div', { class: 'rail-group' }, el('h2', {}, 'SemEval corpus'));
    g.appendChild(el('button', {
      class: 'run-item' + (S.runId === '@semeval:' + sv.root ? ' active' : ''),
      onclick: () => selectSemeval(sv.root),
    },
      el('div', { class: 'rname' }, sv.root),
      el('div', { class: 'rmeta' },
        el('span', { class: 'pill arm-corpus' }, el('span', { class: 'dot' }), 'corpus'),
        el('span', {}, sv.langs.length + ' languages'),
        el('span', {}, sv.langs.reduce((a, l) => a + l.n_articles, 0) + ' articles'),
      ),
    ));
    host.appendChild(g);
  }
}

// =========================================================================
// tab machinery
// =========================================================================
function setTabs(defs) {
  const nav = $('#tabs'); nav.innerHTML = '';
  if (!defs.some(d => d.id === S.tab)) S.tab = defs[0]?.id;
  for (const d of defs) {
    nav.appendChild(el('button', {
      class: 'tab' + (S.tab === d.id ? ' active' : ''),
      onclick: () => { S.tab = d.id; closeInspector(); renderTabs(); },
    }, d.label, d.count != null && el('span', { class: 'count' }, d.count)));
  }
  S.tabDefs = defs;
}

let renderGen = 0;
async function renderTabs() {
  const gen = ++renderGen;          // discard any render this one supersedes
  setTabs(S.tabDefs);
  writeHash();
  const view = $('#view');
  view.innerHTML = '';
  view.appendChild(el('div', { class: 'empty-state' }, 'loading…'));
  const def = S.tabDefs.find(d => d.id === S.tab);
  if (!def) { view.innerHTML = ''; return; }
  try {
    const node = await def.render();
    if (gen !== renderGen) return;
    view.innerHTML = '';
    view.appendChild(node);
    view.scrollTop = 0;
  } catch (e) {
    if (gen !== renderGen) return;
    view.innerHTML = '';
    view.appendChild(el('div', { class: 'empty-state' }, 'Failed to render: ' + e.message));
    console.error(e);
  }
}

// =========================================================================
// RUN selection
// =========================================================================
async function selectRun(id) {
  S.runId = id; S.semeval.root = null; S.ui = {};
  closeInspector();
  renderRail();
  $('#run-title').textContent = 'loading…';
  S.manifest = await api('/api/run', { id });
  const m = S.manifest;
  $('#run-title').textContent = m.name;
  $('#run-path').textContent = m.id;

  const defs = [
    { id: 'pipeline', label: 'Pipeline', render: viewPipeline },
    { id: 'codes', label: 'Codes & text', count: m.stats.codes, render: viewCodes },
  ];
  if (m.arm === 'seeded') {
    defs.push({ id: 'taxonomy', label: 'Taxonomy match', count: m.stats.assignments, render: viewTaxonomy });
  } else {
    defs.push({ id: 'lineage', label: 'Lineage', count: m.stats.categories, render: viewLineage });
  }
  if (has('output_charmaz_change_tree.json')) {
    defs.push({ id: 'changes', label: 'Change tree', count: m.stats.iterations, render: viewChanges });
  }
  if (has('output_slot_traces.json')) {
    defs.push({ id: 'ladder', label: 'Slot ladder', render: viewLadder });
  }
  if (has('output_initial_memos.json') || has('output_advanced_memos_seed.json')) {
    defs.push({ id: 'memos', label: 'Memos', render: viewMemos });
  }
  if (has('output_final_theory.md')) defs.push({ id: 'theory', label: 'Theory', render: viewTheory });
  defs.push({ id: 'artifacts', label: 'Artifacts', count: m.artifacts.length, render: viewArtifacts });

  S.tabDefs = defs;
  if (S.pendingTab && defs.some(d => d.id === S.pendingTab)) { S.tab = S.pendingTab; }
  S.pendingTab = null;
  if (!defs.some(d => d.id === S.tab)) S.tab = 'pipeline';
  await renderTabs();
}

async function selectSemeval(root) {
  S.runId = '@semeval:' + root;
  S.semeval = { root, lang: null, article: null, data: null };
  S.manifest = null; S.ui = {};
  closeInspector(); renderRail();
  const sv = S.index.semeval.find(s => s.root === root);
  $('#run-title').textContent = 'SemEval corpus — ' + root;
  $('#run-path').textContent = sv.langs.map(l => `${l.lang} (${l.n_articles})`).join('  ·  ');
  S.semeval.lang = sv.langs[0]?.lang;
  S.tabDefs = [
    { id: 'corpus', label: 'Annotated articles', count: sv.langs.reduce((a, l) => a + l.n_articles, 0), render: viewCorpus },
    { id: 'gold', label: 'Gold overview', render: viewGoldOverview },
  ];
  S.tab = (S.pendingTab && S.tabDefs.some(d => d.id === S.pendingTab)) ? S.pendingTab : 'corpus';
  S.pendingTab = null;
  await renderTabs();
}

// =========================================================================
// VIEW: pipeline spine
// =========================================================================
async function viewPipeline() {
  const m = S.manifest;
  const root = el('div', {});

  root.appendChild(tiles(
    tile(m.stages.length, 'stages'),
    m.stats.chunks != null && tile(m.stats.chunks, 'chunks'),
    m.stats.sources != null && tile(m.stats.sources, 'sources'),
    m.stats.codes != null && tile(m.stats.codes, 'codes'),
    m.stats.categories != null && tile(m.stats.categories, 'categories'),
    m.stats.iterations != null && tile(m.stats.iterations, 'loop iterations'),
    m.stats.failed_chunks ? tile(m.stats.failed_chunks, 'failed chunks', 'bad') : null,
    m.stats.edge_cases != null ? tile(m.stats.edge_cases, 'edge cases', 'warn') : null,
  ));

  const spine = el('div', { class: 'spine' });
  for (const st of m.stages) {
    const open = S.ui['open_' + st.id] ?? (st.kind === 'loop');
    const body = el('div', {});

    const head = el('button', { class: 'stage-head', onclick: () => {
      S.ui['open_' + st.id] = !open; renderTabs();
    } },
      el('span', { class: 'phase-no' }, String(st.phase).padStart(2, '0')),
      el('div', { class: 'h-main' },
        el('h3', {}, st.title),
        el('div', { class: 'sub' }, st.subtitle),
        el('div', { class: 'stage-stats' }, Object.entries(st.stats || {}).map(([k, v]) =>
          el('span', { class: 'stat-chip' + (/(failed|empty|not reached)/i.test(String(v) + k) ? ' warn' : '') },
            el('b', {}, String(v)), ' ' + k))),
      ),
      el('span', { class: 'chev' }, open ? '▾' : '▸'),
    );
    body.appendChild(head);
    if (st.note) body.appendChild(el('div', { class: 'stage-note' }, st.note));
    if (st.artifacts?.length) {
      body.appendChild(el('div', { class: 'artifact-row' }, st.artifacts.map(a =>
        el('button', {
          class: 'artifact-chip', title: a.desc,
          onclick: (ev) => { ev.stopPropagation(); openArtifact(a.file); },
        }, a.label, el('span', { class: 'sz' }, fmtBytes(a.size))))));
    }
    if (open && st.kind === 'loop') body.appendChild(el('div', { class: 'stage-body' }, loopGrid(st)));
    else if (open && st.detail) body.appendChild(el('div', { class: 'stage-body' }, stageJump(st)));

    spine.appendChild(el('div', { class: 'stage-row' },
      el('div', { class: 'rail-col' }, el('div', {
        class: 'node ' + st.status + (st.kind === 'loop' ? ' loop' : ''),
      })),
      el('div', { class: 'stage is-' + st.kind }, body),
    ));
  }
  root.appendChild(spine);
  return root;
}

function stageJump(st) {
  const jumps = {
    chunks: ['codes', 'Open the chunk browser with codes highlighted in the text'],
    codes: ['codes', 'Open the code table and text highlighting'],
    axial: ['lineage', 'Open the lineage view for these categories'],
    changetree: ['changes', 'Open the change tree'],
    ladder: ['ladder', 'Open the full slot-escalation ladder'],
    memos: ['memos', 'Open the memo reader'],
    assignments: ['taxonomy', 'Open the taxonomy match table'],
    signals: ['taxonomy', 'Open category signals'],
    theory: ['theory', 'Read the theoretical account'],
    gold: ['artifacts', 'Open the gold score artifact'],
  };
  const j = jumps[st.detail];
  if (!j) return el('div', { class: 'muted' }, 'No dedicated view.');
  return el('button', { class: 'icon-btn', onclick: () => { S.tab = j[0]; renderTabs(); } }, j[1] + ' →');
}

function loopGrid(st) {
  const cols = st.loop.columns || [];
  if (!cols.length) return el('div', { class: 'muted' }, 'The loop produced no iterations.');
  const rows = cols[0].substeps.map(s => s.title);
  // trailing 1fr spacer keeps the iteration columns packed to the left
  const grid = el('div', {
    class: 'loop-grid',
    style: { gridTemplateColumns: `184px repeat(${cols.length}, minmax(220px, 300px)) 1fr` },
  });
  grid.appendChild(el('div', { class: 'loop-corner' }));
  for (const c of cols) {
    grid.appendChild(el('div', { class: 'loop-colhead' },
      el('div', { class: 'ct' }, c.label),
      el('div', { class: 'cs' },
        c.decision ? c.decision.replace('_', ' ') :
        (c.escalated === false ? 'no escalation' : (c.escalated ? 'escalated' : ''))),
      c.change_counts && Object.keys(c.change_counts).length
        ? el('div', { style: { marginTop: '6px', display: 'flex', gap: '4px', flexWrap: 'wrap' } },
            Object.entries(c.change_counts).map(([t, n]) => tag(t, n + ' ' + t)))
        : null,
    ));
  }
  grid.appendChild(el('div', {}));            // spacer under the corner row
  rows.forEach((rowTitle, ri) => {
    grid.appendChild(el('div', { class: 'loop-rowlabel' }, rowTitle));
    for (const c of cols) {
      const s = c.substeps[ri];
      if (!s) { grid.appendChild(el('div', { class: 'loop-cell empty' }, '—')); continue; }
      grid.appendChild(el('div', {
        class: 'loop-cell' + (s.status === 'warn' ? ' warn' : ''),
        onclick: () => inspect(s.title, c.label,
          el('div', {}, s.note ? el('p', { class: 'muted' }, s.note) : null,
            renderLoopPayload(s.payload ?? s))),
      },
        el('div', { class: 'lt' }, s.stat ?? s.title),
        el('div', { class: 'ls' }, s.note || (s.payload?.rung ? 'rung ' + s.payload.rung : '')),
      ));
    }
    grid.appendChild(el('div', {}));          // trailing spacer
  });
  return el('div', { class: 'loop-wrap' }, grid);
}

function renderLoopPayload(p) {
  const box = el('div', {});
  if (!p || typeof p !== 'object') return jsonBlock(p);

  if (Array.isArray(p.slice_sources)) {
    box.appendChild(el('div', { class: 'section-title' }, 'Sampled sources'));
    box.appendChild(el('div', {}, p.slice_sources.map(s =>
      el('div', { class: 'lineage-code' }, s))));
  }
  if (Array.isArray(p.changes_made)) {
    box.appendChild(el('div', { class: 'section-title' }, 'Declared changes'));
    if (!p.changes_made.length) box.appendChild(el('div', { class: 'muted' }, 'No categories changed.'));
    for (const c of p.changes_made) {
      box.appendChild(el('div', { class: 'card' },
        el('div', { style: { marginBottom: '6px' } }, tag(c.type)),
        el('h3', {}, c.category || c.name || '—'),
        c.from && el('div', { class: 'cs' }, 'from: ' + (Array.isArray(c.from) ? c.from.join(' + ') : c.from)),
        c.into && el('div', { class: 'cs' }, 'into: ' + (Array.isArray(c.into) ? c.into.join(' + ') : c.into)),
        el('p', { class: 'muted', style: { margin: '4px 0 0', fontSize: '12.5px' } }, c.rationale || ''),
      ));
    }
  }
  if (Array.isArray(p.new_initial_codes)) {
    box.appendChild(el('div', { class: 'section-title' }, `New initial codes (${p.new_initial_codes.length})`));
    for (const c of p.new_initial_codes) {
      box.appendChild(el('div', { class: 'card' },
        el('h3', {}, c.open_code, c.in_vivo ? el('span', { class: 'pill', style: { marginLeft: '8px' } }, 'in vivo') : null),
        el('blockquote', { class: 'passage' }, c.text_passage || '')));
    }
  }
  if (Array.isArray(p.resulting_categories)) {
    box.appendChild(el('div', { class: 'section-title' }, `Resulting categories (${p.resulting_categories.length})`));
    for (const c of p.resulting_categories) {
      box.appendChild(el('div', { class: 'card' },
        el('h3', {}, el('span', { style: { color: catColor(c.axial_category || c.category) } }, '● '),
          c.axial_category || c.category),
        el('div', { class: 'cs' }, (c.supporting_open_codes || []).length + ' supporting codes'),
        el('p', { class: 'muted', style: { fontSize: '12.5px', margin: 0 } }, c.reasoning || '')));
    }
  }
  if (p.applicability_raw != null || p.decision) {
    box.appendChild(el('div', { class: 'section-title' }, 'Applicability check'));
    box.appendChild(kv([
      ['decision', p.decision], ['units fit', p.n_fit], ['unabsorbed', p.n_unabsorbed_units],
    ]));
    if (p.applicability_raw) box.appendChild(rawDetails(safeParse(p.applicability_raw), 'assignment payload'));
  }
  if (p.advanced_memo) {
    box.appendChild(el('div', { class: 'section-title' }, 'Advanced memo'));
    box.appendChild(memoList((p.advanced_memo.memos) || []));
  }
  if (p.rung) {
    box.appendChild(kv(Object.entries(p).map(([k, v]) =>
      [k, typeof v === 'object' ? JSON.stringify(v) : v])));
  }
  if (!box.childNodes.length) box.appendChild(jsonBlock(p));
  return box;
}

const safeParse = s => { try { return JSON.parse(s); } catch { return s; } };

// =========================================================================
// VIEW: codes & text (annotation highlighting)
// =========================================================================
async function viewCodes() {
  const { file: codeFile, codes: valid, failed } = await loadCodes();
  if (!codeFile) return el('div', { class: 'empty-state' }, 'This run has no code artifact.');
  const chunkIndex = has('chunk_index.json') ? (await artifact('chunk_index.json')) : null;
  const legacy = valid.some(c => c.__legacy__);

  // group codes by chunk
  const byChunk = new Map();
  for (const c of valid) {
    const k = c.chunk_id || '(no chunk)';
    if (!byChunk.has(k)) byChunk.set(k, []);
    byChunk.get(k).push(c);
  }

  // category membership for coloring, when categories exist
  const catFile = firstOf('output_axial_codes_resolved.json', 'output_focused_codes_final.json',
                          'output_axial_codes.json', 'output_focused_codes.json');
  const cats = catFile ? (await artifact(catFile)) || [] : [];
  const codeToCat = new Map();
  for (const c of Array.isArray(cats) ? cats : []) {
    for (const oc of c.supporting_open_codes || []) codeToCat.set(norm(oc), c.axial_category);
  }

  const q = S.ui.codeQ || '';
  const chunkIds = [...byChunk.keys()].sort((a, b) =>
    a.localeCompare(b, undefined, { numeric: true }));
  const filtered = q
    ? chunkIds.filter(id => id.toLowerCase().includes(q.toLowerCase()) ||
        byChunk.get(id).some(c => (c.open_code || '').toLowerCase().includes(q.toLowerCase()) ||
                                  (c.text_passage || '').toLowerCase().includes(q.toLowerCase())))
    : chunkIds;
  if (!S.ui.chunk || !byChunk.has(S.ui.chunk)) S.ui.chunk = filtered[0] || chunkIds[0];

  const root = el('div', {});
  root.appendChild(tiles(
    tile(valid.length, 'codes'),
    tile(byChunk.size, 'chunks with codes'),
    chunkIndex && tile(Object.keys(chunkIndex).length - byChunk.size, 'chunks with none', 'warn'),
    tile(new Set(valid.map(c => norm(c.open_code))).size, 'distinct labels'),
    failed.length ? tile(failed.length, 'failed chunks', 'bad') : null,
    tile(codeToCat.size ? [...new Set(valid.map(c => norm(c.open_code)))].filter(k => !codeToCat.has(k)).length : '—',
      'labels unaggregated', 'warn'),
  ));

  root.appendChild(el('div', { class: 'toolbar' },
    el('input', {
      type: 'search', placeholder: 'filter chunks, codes, passages…', value: q,
      oninput: e => { S.ui.codeQ = e.target.value; clearTimeout(S.ui._t);
        S.ui._t = setTimeout(() => { S.ui.chunk = null; renderTabs(); }, 220); },
    }),
    el('span', { class: 'muted', style: { fontSize: '12px' } },
      `${filtered.length} / ${chunkIds.length} chunks`),
  ));

  const picker = el('div', { class: 'picker' },
    el('div', { class: 'ph' }, 'chunks'),
    el('div', { class: 'pl' }, filtered.map(id => el('button', {
      class: 'pi' + (S.ui.chunk === id ? ' on' : ''),
      onclick: () => { S.ui.chunk = id; renderTabs(); },
    },
      el('div', { class: 'pt' }, chunkIndex?.[id]?.question?.slice(0, 90) || id),
      el('div', { class: 'ps' }, `${byChunk.get(id).length} codes · ${String(id).split('_').pop()}`),
    ))),
  );

  if (legacy) {
    root.appendChild(el('div', { class: 'stage-note', style: { marginLeft: 0, marginRight: 0 } },
      'Legacy artifact format: this run stored a comma-joined code list per chunk with no ' +
      'evidence passages and no chunk index, so nothing can be highlighted back into the text. ' +
      'Labels are shown as-is.'));
  }

  root.appendChild(el('div', { class: 'split' }, picker,
    chunkPanel(S.ui.chunk, byChunk.get(S.ui.chunk) || [], chunkIndex?.[S.ui.chunk], codeToCat)));
  return root;
}

function chunkPanel(chunkId, codes, chunk, codeToCat) {
  const wrap = el('div', {});
  if (!chunkId) return el('div', { class: 'empty-state' }, 'No chunk selected.');

  const text = chunk?.text ?? null;
  wrap.appendChild(el('div', { class: 'card' },
    el('h3', {}, chunk?.question ? 'Q' + (chunk.q_index ?? '') + ' · ' + chunk.question.split('\n')[0] : chunkId),
    el('div', { class: 'cs' }, [chunk?.source_id, chunk?.unit, chunkId].filter(Boolean).join('  ·  ')),
  ));

  // legend by category
  const used = new Map();
  for (const c of codes) {
    const cat = codeToCat.get(norm(c.open_code));
    if (cat) used.set(cat, catColor(cat));
  }
  if (used.size) {
    wrap.appendChild(el('div', { class: 'legend' }, [...used].map(([name, col]) =>
      el('span', { class: 'li' }, el('span', { class: 'sw', style: { background: col } }), name))));
  }

  if (text) {
    const anns = [];
    codes.forEach((c, i) => {
      const cat = codeToCat.get(norm(c.open_code));
      const col = cat ? catColor(cat) : 'var(--ink-3)';
      for (const sp of locatePassage(text, c.text_passage)) {
        anns.push({ ...sp, color: col, sup: String(i + 1), title: c.open_code,
          payload: c, cat, key: 'c' + i });
      }
    });
    const unlocated = codes.filter(c => !locatePassage(text, c.text_passage).length);
    const doc = el('div', { class: 'doc' });
    doc.appendChild(renderAnnotated(text, anns, a => inspectCode(a.payload, a.cat)));
    wrap.appendChild(doc);
    if (unlocated.length) {
      wrap.appendChild(el('div', { class: 'stage-note', style: { margin: '10px 0 0' } },
        `${unlocated.length} of ${codes.length} passage(s) could not be located verbatim in the chunk — ` +
        `the model paraphrased or merged them. They are listed below without highlights.`));
    }
  } else {
    wrap.appendChild(el('div', { class: 'stage-note', style: { margin: '0 0 12px' } },
      'No chunk text available (this run has no chunk_index.json) — showing quoted passages only.'));
  }

  wrap.appendChild(el('div', { class: 'section-title' }, `Codes on this chunk (${codes.length})`));
  codes.forEach((c, i) => {
    const cat = codeToCat.get(norm(c.open_code));
    wrap.appendChild(el('div', { class: 'card', style: { cursor: 'pointer' },
      onclick: () => inspectCode(c, cat) },
      el('h3', {}, el('span', { class: 'muted', style: { fontFamily: 'var(--mono)', fontSize: '11px' } }, i + 1 + '  '),
        c.open_code,
        c.in_vivo ? el('span', { class: 'pill', style: { marginLeft: '8px' } }, 'in vivo') : null),
      cat ? el('div', { class: 'cs' }, el('span', { style: { color: catColor(cat) } }, '● '), cat)
          : el('div', { class: 'cs' }, el('span', { style: { color: 'var(--critical)' } }, '● '), 'not aggregated into any category'),
      c.text_passage ? el('blockquote', { class: 'passage' }, c.text_passage) : null,
    ));
  });
  return wrap;
}

function inspectCode(c, cat) {
  inspect(c.open_code || 'code', c.chunk_id || '', el('div', {},
    kv([
      ['category', cat ? el('span', { style: { color: catColor(cat) } }, cat)
                       : el('span', { style: { color: 'var(--critical)' } }, 'none — orphaned code')],
      ['chunk', c.chunk_id], ['source', c.source_id], ['in vivo', c.in_vivo ? 'yes' : null],
    ]),
    el('div', { class: 'section-title' }, 'Quoted passage'),
    el('blockquote', { class: 'passage' }, c.text_passage || ''),
    rawDetails(c),
  ));
}

// =========================================================================
// VIEW: lineage (aggregation / merge / split / orphan)
// =========================================================================
function indexCategories(cats) {
  const map = new Map();               // norm(code) -> [catName]
  for (const c of cats || []) {
    const name = c.axial_category || c.category || '(unnamed)';
    for (const oc of c.supporting_open_codes || []) {
      const k = norm(oc);
      if (!map.has(k)) map.set(k, []);
      map.get(k).push(name);
    }
  }
  return map;
}

async function viewLineage() {
  const { file: codeFile, codes } = await loadCodes();
  const codeKeys = new Set(codes.map(c => norm(c.open_code)));
  const codeByKey = new Map(codes.map(c => [norm(c.open_code), c]));

  const finalFile = firstOf('output_axial_codes_resolved.json', 'output_focused_codes_final.json',
                            'output_axial_codes.json', 'output_focused_codes.json');
  const seedFile = has('output_focused_codes.json') && finalFile !== 'output_focused_codes.json'
    ? 'output_focused_codes.json'
    : (has('output_axial_codes.json') && finalFile !== 'output_axial_codes.json' ? 'output_axial_codes.json' : null);

  const finalCats = finalFile ? (await artifact(finalFile)) || [] : [];
  const seedCats = seedFile ? (await artifact(seedFile)) || [] : null;
  if (!Array.isArray(finalCats) || !finalCats.length) {
    return el('div', { class: 'empty-state' }, 'No category artifact in this run.');
  }

  const finalIdx = indexCategories(finalCats);
  const seedIdx = seedCats ? indexCategories(seedCats) : null;

  // --- orphan / phantom accounting -------------------------------------
  const aggregated = new Set([...finalIdx.keys()].filter(k => codeKeys.has(k)));
  const orphanCodes = codes.filter(c => !finalIdx.has(norm(c.open_code)));
  const phantom = [...finalIdx.keys()].filter(k => !codeKeys.has(k));
  const multi = [...finalIdx.entries()].filter(([, v]) => v.length > 1);

  // --- seed → final flow -------------------------------------------------
  let flows = null;
  if (seedIdx) {
    const seedNames = [...new Set(seedCats.map(c => c.axial_category || c.category))];
    const finalNames = [...new Set(finalCats.map(c => c.axial_category || c.category))];
    const matrix = new Map();   // seed -> Map(final -> count)
    for (const [k, seedList] of seedIdx) {
      const finalList = finalIdx.get(k);
      if (!finalList) continue;
      for (const s of seedList) for (const f of finalList) {
        if (!matrix.has(s)) matrix.set(s, new Map());
        matrix.get(s).set(f, (matrix.get(s).get(f) || 0) + 1);
      }
    }
    const inbound = new Map();
    for (const [s, m] of matrix) for (const [f, n] of m) {
      if (!inbound.has(f)) inbound.set(f, new Map());
      inbound.get(f).set(s, n);
    }
    flows = { seedNames, finalNames, matrix, inbound };
  }

  const root = el('div', {});
  root.appendChild(tiles(
    tile(finalCats.length, 'categories'),
    tile(aggregated.size, 'codes aggregated'),
    tile(orphanCodes.length, 'orphaned codes', orphanCodes.length ? 'warn' : ''),
    tile(multi.length, 'codes in 2+ categories', multi.length ? 'warn' : ''),
    tile(phantom.length, 'phantom references', phantom.length ? 'bad' : ''),
  ));

  const mode = S.ui.lineageMode || 'categories';
  root.appendChild(el('div', { class: 'toolbar' },
    el('div', { class: 'seg' }, [
      ['categories', 'Categories'], ['orphans', `Orphans (${orphanCodes.length})`],
      ['overlap', `Overlaps (${multi.length + phantom.length})`],
      flows && ['flow', 'Seed → final flow'],
    ].filter(Boolean).map(([id, label]) => el('button', {
      class: mode === id ? 'on' : '',
      onclick: () => { S.ui.lineageMode = id; renderTabs(); },
    }, label))),
    el('span', { class: 'muted', style: { fontSize: '12px' } },
      seedFile ? `${seedFile} → ${finalFile}` : finalFile),
  ));

  if (mode === 'categories') {
    root.appendChild(el('div', { class: 'section-title' }, 'Category load'));
    root.appendChild(barRows(finalCats.map(c => ({
      label: c.axial_category || c.category,
      value: (c.supporting_open_codes || []).length,
      color: catColor(c.axial_category || c.category),
    })).sort((a, b) => b.value - a.value)));

    root.appendChild(el('div', { class: 'section-title' }, 'Categories & their codes'));
    for (const c of finalCats) {
      const name = c.axial_category || c.category;
      const supp = c.supporting_open_codes || [];
      const inbound = flows?.inbound.get(name);
      const sources = inbound ? [...inbound.entries()].sort((a, b) => b[1] - a[1]) : [];
      const isMerge = sources.length > 1;
      const slots = ['condition', 'action_interaction', 'consequence']
        .filter(s => s in c);
      const emptySlots = slots.filter(s => !String(c[s] || '').trim());
      root.appendChild(el('div', { class: 'card' },
        el('h3', {}, el('span', { style: { color: catColor(name) } }, '● '), name,
          isMerge ? el('span', { style: { marginLeft: '8px' } }, tag('merged', `merge of ${sources.length}`)) : null,
          emptySlots.length ? el('span', { style: { marginLeft: '8px' } }, tag('orphan', `${emptySlots.length} empty slot`)) : null),
        el('div', { class: 'cs' }, `${supp.length} supporting codes`),
        c.reasoning && el('p', { class: 'muted', style: { fontSize: '12.5px', marginTop: 0 } }, c.reasoning),
        sources.length ? el('div', { class: 'flowline', style: { margin: '8px 0' } },
          sources.map(([s, n], i) => [
            i ? el('span', { class: 'farrow' }, '+') : null,
            el('span', { class: 'fnode', style: { color: catColor(s) } }, `${s} (${n})`),
          ]), el('span', { class: 'farrow' }, '→'),
          el('span', { class: 'fnode', style: { color: catColor(name) } }, name)) : null,
        slots.length ? kv(slots.map(s => [s.replace('_', '/'),
          String(c[s] || '').trim() || el('span', { style: { color: 'var(--critical)' } }, '— empty —')])) : null,
        el('details', { class: 'raw', open: supp.length <= 12 },
          el('summary', {}, `supporting codes (${supp.length})`),
          el('div', {}, supp.map(oc => {
            const k = norm(oc), real = codeByKey.get(k);
            return el('div', {
              class: 'lineage-code' + (real ? '' : ' orphaned'),
              title: real ? real.text_passage : 'no open code with this label — phantom reference',
              onclick: () => real ? inspectCode(real, name)
                : inspect(oc, 'phantom reference', el('div', { class: 'stage-note' },
                    'This label appears in the category’s supporting list but matches no code in ' +
                    codeFile + '. The synthesis step invented or reworded it.')),
            }, oc);
          }))),
      ));
    }
  }

  if (mode === 'orphans') {
    root.appendChild(el('div', { class: 'stage-note', style: { marginLeft: 0, marginRight: 0 } },
      'Codes produced by open/initial coding that no category lists as supporting evidence. ' +
      'They were dropped at the aggregation step.'));
    const byChunkSource = new Map();
    for (const c of orphanCodes) {
      const s = c.source_id || (c.chunk_id || '').replace(/_c\d+$/, '') || '(unknown)';
      if (!byChunkSource.has(s)) byChunkSource.set(s, []);
      byChunkSource.get(s).push(c);
    }
    root.appendChild(el('div', { class: 'section-title' }, 'Orphans by source'));
    root.appendChild(barRows([...byChunkSource.entries()]
      .map(([k, v]) => ({ label: k, value: v.length, color: 'var(--critical)' }))
      .sort((a, b) => b.value - a.value)));
    root.appendChild(el('div', { class: 'section-title' }, `All orphaned codes (${orphanCodes.length})`));
    root.appendChild(codeTable(orphanCodes));
  }

  if (mode === 'overlap') {
    root.appendChild(el('div', { class: 'section-title' }, `Codes claimed by more than one category (${multi.length})`));
    if (!multi.length) root.appendChild(el('div', { class: 'muted' }, 'None — the aggregation is a clean partition.'));
    for (const [k, cats] of multi) {
      const real = codeByKey.get(k);
      root.appendChild(el('div', { class: 'card' },
        el('h3', {}, real?.open_code || k),
        el('div', { class: 'flowline', style: { marginTop: '6px' } },
          cats.map(c => el('span', { class: 'fnode', style: { color: catColor(c) } }, c))),
        real && el('blockquote', { class: 'passage' }, real.text_passage)));
    }
    root.appendChild(el('div', { class: 'section-title' }, `Phantom references (${phantom.length})`));
    root.appendChild(el('div', { class: 'stage-note', style: { marginLeft: 0, marginRight: 0 } },
      'Labels listed as supporting evidence that match no code in the coding output — ' +
      'the aggregation step reworded or invented them, so the trace back to raw text is broken.'));
    root.appendChild(el('div', {}, phantom.map(k => {
      const owners = finalIdx.get(k);
      return el('div', { class: 'lineage-code orphaned' }, k, ' ',
        el('span', { class: 'muted' }, '→ ' + owners.join(', ')));
    })));
  }

  if (mode === 'flow' && flows) {
    root.appendChild(el('div', { class: 'stage-note', style: { marginLeft: 0, marginRight: 0 } },
      'Inferred from shared supporting codes: a final category fed by several seed categories is a merge; ' +
      'a seed category whose codes scatter across several final categories is a split.'));
    root.appendChild(el('div', { class: 'section-title' }, 'Seed categories → final categories'));
    for (const s of flows.seedNames) {
      const targets = [...(flows.matrix.get(s) || new Map()).entries()].sort((a, b) => b[1] - a[1]);
      const kind = targets.length === 0 ? 'dropped' : targets.length === 1
        ? (norm(targets[0][0]) === norm(s) ? 'carried' : 'renamed') : 'split';
      root.appendChild(el('div', { class: 'card' },
        el('h3', {}, el('span', { style: { color: catColor(s) } }, '● '), s, ' ',
          el('span', { style: { marginLeft: '6px' } }, tag(kind === 'carried' ? 'clean' : kind, kind))),
        el('div', { class: 'flowline', style: { marginTop: '8px' } },
          el('span', { class: 'fnode', style: { color: catColor(s) } }, s),
          el('span', { class: 'farrow' }, '→'),
          targets.length ? targets.map(([t, n]) =>
            el('span', { class: 'fnode', style: { color: catColor(t) } }, `${t} (${n})`))
            : el('span', { class: 'fnode', style: { color: 'var(--critical)' } }, 'nothing')),
      ));
    }
    const newOnes = flows.finalNames.filter(f => !flows.inbound.has(f));
    if (newOnes.length) {
      root.appendChild(el('div', { class: 'section-title' }, `Final categories with no seed ancestor (${newOnes.length})`));
      root.appendChild(el('div', {}, newOnes.map(f =>
        el('div', { class: 'card' }, el('h3', {}, tag('added', 'added'), ' ',
          el('span', { style: { color: catColor(f) } }, f))))));
    }
  }
  return root;
}

function codeTable(codes) {
  return el('table', { class: 'grid' },
    el('thead', {}, el('tr', {},
      el('th', {}, 'code'), el('th', {}, 'passage'), el('th', {}, 'chunk'))),
    el('tbody', {}, codes.map(c => el('tr', { class: 'clickable', onclick: () => inspectCode(c, null) },
      el('td', {}, c.open_code),
      el('td', { class: 'muted' }, (c.text_passage || '').slice(0, 160)),
      el('td', { class: 'mono' }, c.chunk_id || ''),
    ))));
}

// =========================================================================
// VIEW: change tree (Charmaz)
// =========================================================================
async function viewChanges() {
  const tree = await artifact('output_charmaz_change_tree.json');
  const its = tree.change_tree || [];
  const root = el('div', {});
  root.appendChild(tiles(
    tile(tree.n_iterations, 'iterations'),
    tile(tree.saturation_reached ? 'yes' : 'no', 'saturation reached',
      tree.saturation_reached ? 'good' : 'warn'),
    tile(String(tree.stop_reason || '—').replace(/_/g, ' '), 'stop reason'),
    tile(pct(tree.saturation_threshold), 'threshold'),
    tile(tree.slice_size, 'slice size'),
  ));

  root.appendChild(el('div', { class: 'section-title' }, 'Absorbed fraction per iteration'));
  root.appendChild(barRows(its.map(i => ({
    label: 'Iteration ' + i.iteration_n,
    value: Math.round(100 * (i.absorbed_fraction || 0)),
    color: (i.absorbed_fraction || 0) >= (i.threshold || 1) ? 'var(--good)' : 'var(--warning)',
  })), { fmt: v => v + '%' }));
  root.appendChild(el('div', { class: 'muted', style: { fontSize: '11.5px', marginTop: '4px' } },
    `Saturation threshold ${pct(tree.saturation_threshold)} — bars below it mean the slice brought ` +
    `material the existing categories could not absorb.`));

  const allChanges = [];
  for (const i of its) for (const c of i.changes_made || []) allChanges.push({ ...c, iteration: i.iteration_n });
  const byType = {};
  for (const c of allChanges) byType[c.type] = (byType[c.type] || 0) + 1;

  root.appendChild(el('div', { class: 'section-title' }, 'Category operations across the run'));
  if (!allChanges.length) root.appendChild(el('div', { class: 'muted' }, 'No category changed during the loop.'));
  else {
    root.appendChild(el('div', { class: 'legend' }, Object.entries(byType)
      .map(([t, n]) => tag(t, `${n} ${t}`))));
  }

  for (const it of its) {
    const changes = it.changes_made || [];
    root.appendChild(el('div', { class: 'card' },
      el('h3', {}, `Iteration ${it.iteration_n}`, ' ',
        tag(it.decision === 'saturated' ? 'clean' : 'straddler', it.decision?.replace('_', ' '))),
      el('div', { class: 'cs' },
        `${(it.slice_sources || []).length} source(s) · ${it.n_units} units · ` +
        `${it.n_fit} absorbed (${pct(it.absorbed_fraction)}) · ${it.n_unabsorbed_units} unabsorbed · ` +
        `${(it.new_initial_codes || []).length} new codes`),
      el('div', { class: 'flowline', style: { margin: '4px 0 10px' } },
        (it.slice_sources || []).map(s => el('span', { class: 'fnode' }, s))),
      changes.length
        ? el('div', {}, changes.map(c => el('div', {
            class: 'card', style: { background: 'var(--surface-2)', marginBottom: '8px' },
          },
          el('div', {}, tag(c.type), ' ',
            el('b', { style: { color: catColor(c.category) } }, c.category)),
          el('p', { class: 'muted', style: { fontSize: '12.5px', margin: '6px 0 0' } }, c.rationale || ''),
        )))
        : el('div', { class: 'muted' }, 'No category changes in this iteration.'),
      el('details', { class: 'raw' }, el('summary', {}, 'new initial codes from this slice'),
        el('div', {}, (it.new_initial_codes || []).map(c => el('div', { class: 'card',
          style: { background: 'var(--surface-2)' } },
          el('h3', {}, c.open_code, c.in_vivo ? el('span', { class: 'pill', style: { marginLeft: '8px' } }, 'in vivo') : null),
          el('blockquote', { class: 'passage' }, c.text_passage || ''))))),
      el('details', { class: 'raw' }, el('summary', {}, 'resulting categories'),
        el('div', {}, (it.resulting_categories || []).map(c => el('div', { class: 'lineage-code' },
          el('span', { style: { color: catColor(c.axial_category) } }, '● '), c.axial_category,
          ' ', el('span', { class: 'muted' }, `(${(c.supporting_open_codes || []).length})`))))),
    ));
  }
  return root;
}

// =========================================================================
// VIEW: slot ladder (Straussian)
// =========================================================================
async function viewLadder() {
  const traces = await artifact('output_slot_traces.json');
  const resolved = has('output_axial_codes_resolved.json') ? await artifact('output_axial_codes_resolved.json') : null;
  const before = has('output_axial_codes.json') ? await artifact('output_axial_codes.json') : null;
  const root = el('div', {});

  const escalated = traces.filter(t => (t.trace || []).some(r => r.rung && r.rung !== 'none'));
  const SLOTS = ['condition', 'action_interaction', 'consequence'];
  const countEmpty = arr => (arr || []).reduce((a, c) =>
    a + SLOTS.filter(s => !String(c[s] || '').trim()).length, 0);

  root.appendChild(tiles(
    tile(traces.length, 'categories'),
    tile(escalated.length, 'escalated', escalated.length ? 'warn' : 'good'),
    before && tile(countEmpty(before), 'empty slots before'),
    resolved && tile(countEmpty(resolved), 'empty slots after',
      countEmpty(resolved) ? 'warn' : 'good'),
  ));

  if (!escalated.length) {
    root.appendChild(el('div', { class: 'stage-note', style: { marginLeft: 0, marginRight: 0 } },
      'Every category came out of axial coding with all three paradigm slots filled, so the ' +
      'escalation ladder short-circuited at rung "none" for all of them. The per-category rows ' +
      'below record that decision.'));
  }

  root.appendChild(el('div', { class: 'section-title' }, 'Per-category escalation'));
  for (const t of traces) {
    const rungs = t.trace || [];
    const cat = (resolved || before || []).find(c => c.axial_category === t.axial_category);
    root.appendChild(el('div', { class: 'card' },
      el('h3', {}, el('span', { style: { color: catColor(t.axial_category) } }, '● '), t.axial_category),
      el('div', { class: 'flowline', style: { marginTop: '8px' } },
        rungs.map((r, i) => [
          i ? el('span', { class: 'farrow' }, '→') : null,
          el('span', {
            class: 'fnode',
            style: { color: r.rung === 'none' ? 'var(--ink-3)' : r.rung.startsWith('terminal') ? 'var(--critical)' : 'var(--series-4)', cursor: 'pointer' },
            onclick: () => inspect(r.rung, t.axial_category, jsonBlock(r)),
          }, r.rung),
        ])),
      rungs.map(r => r.note ? el('div', { class: 'cs', style: { marginTop: '6px' } }, `${r.rung}: ${r.note}`) : null),
      cat ? kv(SLOTS.map(s => [s.replace('_', '/'), String(cat[s] || '').trim()
        || el('span', { style: { color: 'var(--critical)' } }, '— still empty —')])) : null,
    ));
  }
  return root;
}

// =========================================================================
// VIEW: memos
// =========================================================================
function memoList(memos) {
  return el('div', {}, (memos || []).map(m => el('div', { class: 'card' },
    el('h3', {}, m.focus_code || m.category || m.title || 'memo'),
    m.tentative_category && el('div', { class: 'cs' }, '→ ' + m.tentative_category),
    m.properties?.length ? el('div', { class: 'legend', style: { margin: '4px 0 8px' } },
      m.properties.map(p => el('span', { class: 'pill' }, p))) : null,
    el('p', { class: 'muted', style: { fontSize: '12.5px', margin: '0 0 6px' } },
      m.reasoning || m.memo || m.text || ''),
    m.thin_areas?.length ? el('div', {},
      el('div', { class: 'section-title', style: { margin: '10px 0 6px' } }, 'Thin / ambiguous areas'),
      el('div', {}, m.thin_areas.map(t => el('div', { class: 'lineage-code orphaned' }, t)))) : null,
    rawDetails(m),
  )));
}

async function viewMemos() {
  const root = el('div', {});
  const sets = [
    ['output_initial_memos.json', 'Initial memos (step 4)'],
    ['output_advanced_memos_seed.json', 'Advanced memos — seed pass (step 6)'],
    ['output_advanced_memos_final.json', 'Advanced memos — final (post-loop)'],
  ].filter(([f]) => has(f));
  for (const [f, label] of sets) {
    const data = await artifact(f);
    const memos = data?.memos || [];
    root.appendChild(el('div', { class: 'section-title' }, `${label} — ${memos.length}`));
    if (data?.__raw__) {
      root.appendChild(el('div', { class: 'stage-note', style: { marginLeft: 0, marginRight: 0 } },
        'This memo file failed JSON parsing at write time; showing the raw model output.'));
      root.appendChild(el('pre', { class: 'json' }, data.__raw__));
    }
    root.appendChild(memoList(memos));
  }
  return root;
}

// =========================================================================
// VIEW: taxonomy (seeded arm)
// =========================================================================
async function viewTaxonomy() {
  const rows = (await artifact('output_taxonomy_assignments.json')) || [];
  const signals = has('output_taxonomy_category_signals.json')
    ? await artifact('output_taxonomy_category_signals.json') : null;
  const root = el('div', {});

  const types = {};
  for (const r of rows) for (const t of (r.edge_case_types || '').split('|')) if (t) types[t] = (types[t] || 0) + 1;
  const agree = rows.filter(r => r.agree === true).length;

  root.appendChild(tiles(
    tile(rows.length, 'assignments'),
    tile(rows.filter(r => r.edge_case_type === 'clean').length, 'clean', 'good'),
    tile(types.orphan || 0, 'orphans', 'bad'),
    tile(types.straddler || 0, 'straddlers', 'warn'),
    tile(types.matcher_disagreement || 0, 'disagreements', 'warn'),
    tile(rows.length ? Math.round(100 * agree / rows.length) + '%' : '—', 'matcher agreement'),
  ));

  // category signals
  const packs = signals?.per_domain ? Object.entries(signals.per_domain) : (signals ? [['', signals]] : []);
  for (const [dom, sig] of packs) {
    const per = sig.per_category || {};
    const empty = new Set(sig.empty_categories || []);
    const over = new Set(sig.over_populated || []);
    root.appendChild(el('div', { class: 'section-title' },
      `Taxonomy leaf load${dom ? ' — ' + dom : ''} · ${sig.n_leaves} leaves · ${empty.size} empty · ${over.size} over-populated`));
    root.appendChild(el('div', { class: 'legend' },
      tag('empty', 'empty leaf (missed category)'),
      tag('over_populated', 'over-populated (over-split candidate)')));
    root.appendChild(barRows(Object.entries(per).map(([cid, v]) => ({
      label: v.name || cid,
      value: v.n_codes,
      cls: empty.has(cid) ? 'is-empty' : over.has(cid) ? 'is-over' : '',
      color: empty.has(cid) ? 'var(--critical)' : over.has(cid) ? 'var(--warning)' : 'var(--accent)',
    })).sort((a, b) => b.value - a.value)));
  }

  // edge case table
  const filt = S.ui.edgeFilter || 'all';
  const shown = filt === 'all' ? rows
    : rows.filter(r => (r.edge_case_types || '').split('|').includes(filt) ||
                       (filt === 'clean' && r.edge_case_type === 'clean'));
  root.appendChild(el('div', { class: 'section-title' }, 'Assignments'));
  root.appendChild(el('div', { class: 'toolbar' },
    el('div', { class: 'seg' }, [['all', `All (${rows.length})`], ['clean', 'Clean'],
      ...Object.entries(types).map(([t, n]) => [t, `${t} (${n})`])].map(([id, label]) =>
      el('button', { class: filt === id ? 'on' : '', onclick: () => { S.ui.edgeFilter = id; renderTabs(); } }, label)))));

  root.appendChild(el('table', { class: 'grid' },
    el('thead', {}, el('tr', {}, el('th', {}, 'open code'), el('th', {}, 'embedding leaf'),
      el('th', {}, 'score'), el('th', {}, 'margin'), el('th', {}, 'LLM leaf'), el('th', {}, 'edge type'))),
    el('tbody', {}, shown.slice(0, 600).map(r => el('tr', {
      class: 'clickable', onclick: () => inspect(r.open_code, r.chunk_id || '', el('div', {},
        kv([['embedding', r.emb_category], ['emb score', r.emb_score], ['emb margin', r.emb_margin],
            ['LLM', r.llm_category], ['LLM confidence', r.llm_confidence],
            ['agree', String(r.agree)], ['edge types', r.edge_case_types]]),
        el('div', { class: 'section-title' }, 'Evidence passage'),
        el('blockquote', { class: 'passage' }, r.text_passage || ''), rawDetails(r))),
    },
      el('td', {}, r.open_code),
      el('td', { class: 'mono' }, r.emb_category),
      el('td', { class: 'num' }, (r.emb_score ?? 0).toFixed(3)),
      el('td', { class: 'num' }, (r.emb_margin ?? 0).toFixed(3)),
      el('td', { class: 'mono' }, r.llm_category ?? '—'),
      el('td', {}, tag(r.edge_case_type || 'clean')),
    )))));
  if (shown.length > 600) root.appendChild(el('div', { class: 'muted', style: { marginTop: '8px' } },
    `showing first 600 of ${shown.length}`));
  return root;
}

// =========================================================================
// VIEW: theory / artifacts
// =========================================================================
async function viewTheory() {
  const txt = await artifact('output_final_theory.md');
  return el('div', { class: 'doc md', style: { whiteSpace: 'normal' }, html: md(txt) });
}

async function viewArtifacts() {
  const root = el('div', {});
  const sel = S.ui.artifact || (S.ui.artifact = S.manifest.artifacts[0]?.file);
  root.appendChild(el('div', { class: 'toolbar' }, S.manifest.artifacts.map(a =>
    el('button', { class: 'artifact-chip', onclick: () => { S.ui.artifact = a.file; renderTabs(); } },
      a.label, el('span', { class: 'sz' }, fmtBytes(a.size))))));
  if (!sel) return (root.appendChild(el('div', { class: 'empty-state' }, 'Pick an artifact above.')), root);

  const meta = S.manifest.artifacts.find(a => a.file === sel);
  root.appendChild(el('div', { class: 'card' }, el('h3', {}, meta.label),
    el('div', { class: 'cs' }, `${sel} · ${fmtBytes(meta.size)}`),
    el('p', { class: 'muted', style: { margin: 0, fontSize: '12.5px' } }, meta.desc)));

  const data = await artifact(sel);
  if (sel.endsWith('.md')) root.appendChild(el('div', { class: 'doc md', style: { whiteSpace: 'normal' }, html: md(data) }));
  else if (sel.endsWith('.pkl')) root.appendChild(el('div', { class: 'muted' }, 'Binary cache — not inspectable.'));
  else if (sel.endsWith('.csv')) root.appendChild(el('pre', { class: 'json' }, String(data).slice(0, 200000)));
  else root.appendChild(el('pre', { class: 'json' }, JSON.stringify(data, null, 2).slice(0, 400000)));
  return root;
}

function openArtifact(file) {
  S.ui.artifact = file; S.tab = 'artifacts'; renderTabs();
}

// =========================================================================
// VIEW: SemEval corpus
// =========================================================================
async function viewCorpus() {
  const sv = S.index.semeval.find(s => s.root === S.semeval.root);
  const lang = sv.langs.find(l => l.lang === S.semeval.lang) || sv.langs[0];
  S.semeval.lang = lang.lang;

  const q = (S.ui.artQ || '').toLowerCase();
  const arts = lang.articles.filter(a => !q || a.id.toLowerCase().includes(q));
  if (!S.semeval.article || !arts.some(a => a.id === S.semeval.article)) {
    S.semeval.article = arts[0]?.id;
    S.semeval.data = null;
  }

  const root = el('div', {});
  root.appendChild(tiles(
    tile(lang.n_articles, 'articles'),
    tile(lang.n_spans, 'gold entity spans'),
    tile(lang.articles.filter(a => a.n_spans > 0).length, 'with entity labels'),
    tile(lang.articles.filter(a => a.n_narratives > 0).length, 'with narrative labels'),
  ));

  root.appendChild(el('div', { class: 'toolbar' },
    el('div', { class: 'seg' }, sv.langs.map(l => el('button', {
      class: l.lang === lang.lang ? 'on' : '',
      onclick: () => { S.semeval.lang = l.lang; S.semeval.article = null; S.semeval.data = null; renderTabs(); },
    }, `${l.lang} (${l.n_articles})`))),
    el('input', {
      type: 'search', placeholder: 'filter article id…', value: S.ui.artQ || '',
      oninput: e => { S.ui.artQ = e.target.value; clearTimeout(S.ui._t2);
        S.ui._t2 = setTimeout(() => { S.semeval.article = null; renderTabs(); }, 220); },
    }),
  ));

  const picker = el('div', { class: 'picker' },
    el('div', { class: 'ph' }, `${lang.lang} · ${arts.length} articles`),
    el('div', { class: 'pl' }, arts.map(a => el('button', {
      class: 'pi' + (S.semeval.article === a.id ? ' on' : ''),
      onclick: () => { S.semeval.article = a.id; S.semeval.data = null; renderTabs(); },
    },
      el('div', { class: 'pt' }, a.id.replace(/\.txt$/, '')),
      el('div', { class: 'ps' }, `${a.domain} · ${a.n_spans} spans · ${a.n_narratives} narratives`),
    ))),
  );

  const panel = el('div', {});
  root.appendChild(el('div', { class: 'split' }, picker, panel));

  if (!S.semeval.article) { panel.appendChild(el('div', { class: 'empty-state' }, 'No article.')); return root; }
  if (!S.semeval.data || S.semeval.data.id !== S.semeval.article) {
    S.semeval.data = await api('/api/semeval/article',
      { root: S.semeval.root, lang: lang.lang, id: S.semeval.article });
  }
  const d = S.semeval.data;

  const roles = [...new Set(d.spans.map(s => s.main_role))];
  panel.appendChild(el('div', { class: 'card' },
    el('h3', {}, d.id.replace(/\.txt$/, '')),
    el('div', { class: 'cs' }, `${d.text.length} chars · ${d.spans.length} entity spans · ` +
      `${(d.narratives.narratives || []).length} narratives`)));

  if (d.narratives?.narratives?.length) {
    panel.appendChild(el('div', { class: 'section-title' }, 'Gold narratives (subtask 2)'));
    panel.appendChild(el('div', { class: 'legend' },
      d.narratives.narratives.map(n => el('span', { class: 'pill' }, n))));
    if (d.narratives.subnarratives?.length) {
      panel.appendChild(el('div', { style: { marginTop: '8px' } },
        d.narratives.subnarratives.map(n => el('div', { class: 'lineage-code' }, n))));
    }
  }

  if (roles.length) {
    panel.appendChild(el('div', { class: 'section-title' }, 'Entity roles (subtask 1)'));
    panel.appendChild(el('div', { class: 'legend' }, roles.map(r =>
      el('span', { class: 'li' },
        el('span', { class: 'sw', style: { background: ROLE_COLOR[r] || 'var(--series-4)' } }), r))));
  }

  const bad = d.spans.filter(s => !s.offset_ok);
  if (bad.length) {
    panel.appendChild(el('div', { class: 'stage-note', style: { marginLeft: 0, marginRight: 0 } },
      `${bad.length} gold span(s) do not match the text at their stated offsets — highlighted anyway, ` +
      `flagged in the span list.`));
  }

  const anns = d.spans.map((s, i) => ({
    start: s.start, end: s.end + 1,
    color: ROLE_COLOR[s.main_role] || 'var(--series-4)',
    sup: s.main_role[0], title: `${s.entity} — ${s.main_role}: ${s.fine_roles.join(', ')}`,
    payload: s,
  }));
  const doc = el('div', { class: 'doc' });
  doc.appendChild(renderAnnotated(d.text, anns, a => inspectSpan(a.payload)));
  panel.appendChild(doc);

  panel.appendChild(el('div', { class: 'section-title' }, `Annotated entities (${d.spans.length})`));
  panel.appendChild(el('table', { class: 'grid' },
    el('thead', {}, el('tr', {}, el('th', {}, 'entity'), el('th', {}, 'offsets'),
      el('th', {}, 'main role'), el('th', {}, 'fine roles'))),
    el('tbody', {}, d.spans.map(s => el('tr', { class: 'clickable', onclick: () => inspectSpan(s) },
      el('td', {}, s.entity, !s.offset_ok ? el('span', { style: { marginLeft: '6px' } }, tag('orphan', 'offset drift')) : null),
      el('td', { class: 'num' }, `${s.start}–${s.end}`),
      el('td', {}, el('span', { style: { color: ROLE_COLOR[s.main_role] || 'var(--series-4)' } }, '● '), s.main_role),
      el('td', { class: 'muted' }, s.fine_roles.join(', ')),
    )))));

  if (d.explanations?.length) {
    panel.appendChild(el('div', { class: 'section-title' }, 'Narrative explanations (subtask 3)'));
    for (const e of d.explanations) {
      panel.appendChild(el('div', { class: 'card' },
        el('h3', {}, e.narrative),
        e.subnarrative && e.subnarrative !== 'none' && el('div', { class: 'cs' }, e.subnarrative),
        el('p', { class: 'muted', style: { fontSize: '12.5px', margin: 0 } }, e.explanation)));
    }
  }
  return root;
}

function inspectSpan(s) {
  inspect(s.entity, `${s.start}–${s.end}`, el('div', {},
    kv([
      ['main role', el('span', { style: { color: ROLE_COLOR[s.main_role] } }, s.main_role)],
      ['fine roles', s.fine_roles.join(', ')],
      ['offsets', `${s.start}–${s.end}`],
      ['text at offsets', s.actual],
      ['offset check', s.offset_ok ? 'matches' : el('span', { style: { color: 'var(--critical)' } }, 'MISMATCH')],
    ]), rawDetails(s)));
}

async function viewGoldOverview() {
  const sv = S.index.semeval.find(s => s.root === S.semeval.root);
  const root = el('div', {});
  root.appendChild(tiles(
    tile(sv.langs.length, 'languages'),
    tile(sv.langs.reduce((a, l) => a + l.n_articles, 0), 'articles'),
    tile(sv.langs.reduce((a, l) => a + l.n_spans, 0), 'entity spans'),
  ));
  root.appendChild(el('div', { class: 'section-title' }, 'Articles per language'));
  root.appendChild(barRows(sv.langs.map(l => ({ label: l.lang, value: l.n_articles }))));
  root.appendChild(el('div', { class: 'section-title' }, 'Gold entity spans per language'));
  root.appendChild(barRows(sv.langs.map(l => ({ label: l.lang, value: l.n_spans, color: 'var(--series-2)' }))));
  root.appendChild(el('div', { class: 'section-title' }, 'Coverage'));
  root.appendChild(el('table', { class: 'grid' },
    el('thead', {}, el('tr', {}, el('th', {}, 'language'), el('th', {}, 'articles'),
      el('th', {}, 'with entity spans'), el('th', {}, 'with narratives'), el('th', {}, 'spans'))),
    el('tbody', {}, sv.langs.map(l => el('tr', {},
      el('td', {}, l.lang),
      el('td', { class: 'num' }, l.n_articles),
      el('td', { class: 'num' }, l.articles.filter(a => a.n_spans > 0).length),
      el('td', { class: 'num' }, l.articles.filter(a => a.n_narratives > 0).length),
      el('td', { class: 'num' }, l.n_spans),
    )))));
  return root;
}

// =========================================================================
// URL hash routing -- makes any view linkable / reloadable
// =========================================================================
let suppressHash = false;
function writeHash() {
  if (suppressHash) return;
  const p = new URLSearchParams();
  if (S.runId) p.set('run', S.runId);
  if (S.tab) p.set('tab', S.tab);
  if (S.semeval.lang) p.set('lang', S.semeval.lang);
  if (S.semeval.article) p.set('art', S.semeval.article);
  const h = '#' + p.toString();
  if (decodeURIComponent(location.hash) !== decodeURIComponent(h)) {
    history.replaceState(null, '', h);
  }
}

async function applyHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  const run = p.get('run');
  if (!run) return false;
  suppressHash = true;
  S.pendingTab = p.get('tab');
  try {
    if (run.startsWith('@semeval:')) {
      await selectSemeval(run.slice(9));
      if (p.get('lang')) S.semeval.lang = p.get('lang');
      if (p.get('art')) { S.semeval.article = p.get('art'); S.semeval.data = null; }
      if (p.get('lang') || p.get('art')) await renderTabs();
    } else {
      await selectRun(run);
    }
  } finally {
    suppressHash = false;
  }
  writeHash();
  return true;
}

// =========================================================================
// boot
// =========================================================================
async function boot() {
  $('#insp-close').addEventListener('click', closeInspector);
  $('#theme-btn').addEventListener('click', () => {
    const cur = document.documentElement.getAttribute('data-theme');
    const next = cur === 'light' ? 'dark' : 'light';
    document.documentElement.setAttribute('data-theme', next);
    try { localStorage.setItem('gta-theme', next); } catch {}
  });
  try {
    const t = localStorage.getItem('gta-theme');
    if (t) document.documentElement.setAttribute('data-theme', t);
  } catch {}
  $('#reload-btn').addEventListener('click', async () => {
    S.cache.clear();
    await load();
    if (S.runId?.startsWith('@semeval:')) selectSemeval(S.runId.slice(9));
    else if (S.runId) selectRun(S.runId);
  });
  document.addEventListener('keydown', e => { if (e.key === 'Escape') closeInspector(); });
  window.addEventListener('hashchange', () => { if (!suppressHash) applyHash(); });
  await load();
  if (await applyHash()) return;
  const first = S.index.runs[0];
  if (first) selectRun(first.id);
  else if (S.index.semeval?.length) selectSemeval(S.index.semeval[0].root);
}

async function load() {
  S.index = await api('/api/runs');
  $('#brand-sub').textContent =
    `${S.index.runs.length} runs · ${(S.index.semeval || []).length} corpora`;
  renderRail();
}

boot().catch(e => {
  document.body.innerHTML = `<div style="padding:40px;font:14px system-ui">
    <h2>Could not reach the viewer API</h2><pre>${esc(e.message)}</pre>
    <p>Start it with <code>python viewer/serve.py</code>.</p></div>`;
});
