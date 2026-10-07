/* Guided start: a short sequence of full-screen steps in front of the workspace.

     intro ─► dimension ─► problem ─► data ─► network ─► workspace
       └──── "I know how to use it" ─────────────────────────┘

   Each step is a <section class="wz-screen" data-step="..."> in index.html.  Picking a card stores the
   choice in `choice` and moves on; the step bar at the top shows where you are and lets you jump back to
   any step you have already passed.  Open the page with #workspace to skip the guide.

   The cards on the dimension and problem steps play small looping previews (src/minis.js).  Their
   scenes are built one after another in the background (a few ms per frame), the chosen dimension first.
*/
(function () {
  'use strict';

  const STEPS = ['intro', 'dimension', 'problem', 'data', 'network'];
  const SLIDE_MS = 600;          // long enough for the slowest screen / close transition (see #wizard in index.html)

  const choice = { dim: null, problem: null, data: null, network: null };
  let current = 'intro';
  let reached = 0;            // highest step index visited so far (steps up to it are clickable)

  const $ = (id) => document.getElementById(id);
  const root = $('wizard');
  const screen = (step) => root.querySelector(`.wz-screen[data-step="${step}"]`);

  // ---- live previews --------------------------------------------------------------------------------
  const reduceMotion = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const scenes = {};            // "problem/dim" -> scene
  const queue = [];             // keys still to build
  let building = null;
  const players = new Map();    // canvas -> { key, player }

  function want(key, first) {
    if (scenes[key] || building === key) return;
    const i = queue.indexOf(key);
    if (i >= 0) queue.splice(i, 1);
    if (first) queue.unshift(key); else queue.push(key);
    if (!building) buildNext();
  }

  function buildNext() {
    // previews are only trained while the guide is open (a page opened at #workspace never trains them)
    if (!document.body.classList.contains('wizard-open')) { building = null; return; }
    building = queue.shift() || null;
    if (!building) return;
    const [problem, dim] = building.split('/');
    Minis.sceneAsync(problem, +dim, (sc) => { scenes[building] = sc; buildNext(); });
  }

  /** Point a card's canvas at a scene (it shows "training…" until the scene exists). */
  function bindCard(card, problem, dim) {
    const canvas = card.querySelector('canvas');
    const key = `${problem}/${dim}`;
    const cur = players.get(canvas);
    if (!cur || cur.key !== key) players.set(canvas, { key, dim, player: null });
  }

  function bindProblemCards(dim, first) {
    screen('problem').querySelectorAll('.wz-card').forEach((card) => {
      bindCard(card, card.dataset.value, dim);
      want(`${card.dataset.value}/${dim}`, first);
      const d = card.querySelector('.d[data-d2]');
      if (d) d.textContent = d.dataset[`d${dim}`];
    });
  }

  function drawLoading(canvas) {
    const ctx = canvas.getContext('2d'), w = canvas.clientWidth, h = canvas.clientHeight, dpr = window.devicePixelRatio || 1;
    if (!w) return;
    if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#8b949e'; ctx.font = '12px ui-sans-serif, system-ui, sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('training a tiny network…', w / 2, h / 2);
  }

  let odd = false;
  function frame(now) {
    if (!document.body.classList.contains('wizard-open')) { animating = false; return; }
    odd = !odd;
    if (odd) { requestAnimationFrame(frame); return; }          // previews at 30 fps: smooth enough, half the work
    const sec = screen(current);
    players.forEach((p, canvas) => {
      if (!sec.contains(canvas)) return;
      if (!p.player && scenes[p.key]) p.player = Minis.player(canvas, scenes[p.key], p.dim);
      if (p.player) {
        const stage = p.player.draw(now / 1000, reduceMotion);
        if (p.onStage) p.onStage(stage);
      } else drawLoading(canvas);
    });
    requestAnimationFrame(frame);
  }
  let animating = false;
  function startAnimation() { if (!animating) { animating = true; requestAnimationFrame(frame); } }

  screen('dimension').querySelectorAll('.wz-card').forEach((card) => bindCard(card, 'none', +card.dataset.value));
  want('none/2'); want('none/3');


  // ---- data step: a list of cards, and a detail view with settings for the picked one ---------------
  const esc = (t) => String(t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  let dataFor = '';             // "problem/dim" the card list was built for
  let detail = null;            // { problem, dim, opts, title, blurb } while the detail view is open

  /** Cards for the current problem: [{ group, id, name, blurb, opts }]. */
  function dataOptions(problem, dim) {
    if (problem === 'classify') {
      return Minis.DATASETS.filter((d) => d.dims.includes(dim)).map((d) => ({ group: d.group, id: d.id, name: d.name,
        blurb: d.blurb, info: d, opts: { dataset: d.id, n: 400, noise: 0.08, seed: 0 } }));
    }
    if (problem === 'transform') {
      return Minis.MAPS.map((m) => ({ group: 'Linear map', id: m.id, name: m.name, blurb: m.blurb, info: m,
        opts: { map: m.id, amount: m.amount } }));
    }
    const label = (k) => NN.SHAPES[k].label;
    const cap = (t) => t[0].toUpperCase() + t.slice(1);
    return Minis.PAIRS[dim].map(([a, b]) => ({ group: 'Shape pair', id: `${a}>${b}`, name: `${cap(label(a))} → ${label(b)}`,
      blurb: Minis.PAIR_BLURBS[`${a}>${b}`], opts: { pair: `${a}>${b}` } }));
  }

  const TEXT = {
    classify: ['Which data?', 'Blue and red are the two classes. Open a card to adjust it.'],
    transform: ['Which linear map?', 'The network learns to send every point x to A·x. Open a card to set the amount.'],
    morph: ['Which shapes?', 'Point i of the first shape should land on point i of the second.'],
  };

  /** Show a hidden sub-view with the rise animation (restarted every time). */
  function appear(node) {
    node.classList.remove('hidden', 'appear');
    void node.offsetWidth;
    node.classList.add('appear');
  }

  function forgetDetached() {
    players.forEach((_, canvas) => { if (!canvas.isConnected) players.delete(canvas); });
  }

  function buildDataList() {
    const dim = +choice.dim, problem = choice.problem;
    const key = `${problem}/${dim}`;
    closeDetail(true);
    if (dataFor === key) return;
    dataFor = key;
    choice.data = null;
    $('wzDataTitle').textContent = TEXT[problem][0];
    $('wzDataLead').textContent = TEXT[problem][1];
    const opts = dataOptions(problem, dim), groups = [...new Set(opts.map((o) => o.group))];
    $('wzDataCards').innerHTML = groups.map((g) => `<div class="wz-group">${groups.length > 1 ? `<h3>${esc(g)}</h3>` : ''}
      <div class="wz-cards">${opts.filter((o) => o.group === g).map((o) => `
        <button class="wz-card wz-data" data-id="${esc(o.id)}"><div class="pv live"><canvas></canvas></div>
          <div class="t">${esc(o.name)}</div></button>`).join('')}</div></div>`).join('');
    forgetDetached();
    $('wzDataCards').querySelectorAll('.wz-data').forEach((card) => {
      const o = opts.find((x) => x.id === card.dataset.id);
      card._option = o;
      const sc = Minis.dataScene(problem, dim, { ...o.opts, n: Math.min(o.opts.n || 0, 240) || undefined, dotSize: 1.4 });
      players.set(card.querySelector('canvas'), { key: 'data', dim, player: Minis.player(card.querySelector('canvas'), sc, dim) });
    });
  }

  function slider(id, label, min, max, step, value, fmt, tip) {
    return `<div class="row"><label${tip ? ` data-tip="${esc(tip)}"` : ''}>${label}</label>
      <input id="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${value}"><span class="val" id="${id}Val">${fmt(value)}</span></div>`;
  }

  function openDetail(o) {
    const dim = +choice.dim, problem = choice.problem;
    detail = { problem, dim, opts: { ...o.opts }, option: o };
    $('wzDetTitle').textContent = o.name;
    $('wzDetBlurb').textContent = o.blurb || '';
    let html = '';
    if (problem === 'classify') {
      html += slider('wzN', 'Points', 100, 800, 50, detail.opts.n, (v) => v, 'How many points the network trains on.');
      if (o.info.noise) html += slider('wzNoise', 'Noise', 0, 0.3, 0.01, detail.opts.noise, (v) => (+v).toFixed(2), 'How far points scatter from the clean shape.');
      else if (o.info.exact) html += '<div class="note">Exact shape: no noise, the points lie exactly on it.</div>';
      if (!o.info.exact) html += '<div class="btns"><button id="wzResample">New random sample</button></div>';
    } else if (problem === 'transform') {
      const m = o.info, fmt = (v) => `${(+v).toFixed(2)}${m.unit ? ` ${m.unit}` : ''}`;
      html += slider('wzAmount', m.id === 'rotation' ? 'Angle' : m.id === 'shear' ? 'Shear' : 'Stretch', m.min, m.max, 0.05, detail.opts.amount, fmt);
      html += '<div class="note">The preview moves from the identity to the target map A.</div>';
    } else {
      html += '<div class="note">The preview shows each point travelling to its partner. The dashed curve is the target.</div>';
    }
    $('wzDetControls').innerHTML = html;
    const bind = (id, key, fmt) => {
      const el = $(id);
      if (el) el.oninput = () => { detail.opts[key] = +el.value; $(`${id}Val`).textContent = fmt(el.value); redrawDetail(); };
    };
    bind('wzN', 'n', (v) => v);
    bind('wzNoise', 'noise', (v) => (+v).toFixed(2));
    if (o.info && o.info.unit !== undefined) bind('wzAmount', 'amount', (v) => `${(+v).toFixed(2)}${o.info.unit ? ` ${o.info.unit}` : ''}`);
    if ($('wzResample')) $('wzResample').onclick = () => { detail.opts.seed += 1; redrawDetail(); };
    redrawDetail();
    const list = $('wzDataList'), det = $('wzDataDetail');
    list.classList.add('hidden');
    appear(det);
    screen('data').scrollTop = 0;
    $('wzDataUse').focus({ preventScroll: true });
  }

  function redrawDetail() {
    const canvas = $('wzDataCanvas');
    const sc = Minis.dataScene(detail.problem, detail.dim, { ...detail.opts, dotSize: 2.4 });
    players.set(canvas, { key: 'detail', dim: detail.dim, player: Minis.player(canvas, sc, detail.dim) });
  }

  function closeDetail(silent) {
    if (!detail && silent) return;
    detail = null;
    $('wzDataDetail').classList.add('hidden');
    if (silent) $('wzDataList').classList.remove('hidden');
    else appear($('wzDataList'));
    players.delete($('wzDataCanvas'));
  }

  $('wzDataCards').addEventListener('click', (e) => {
    const card = e.target.closest('.wz-data');
    if (card) openDetail(card._option);
  });
  $('wzDataBack').onclick = () => closeDetail();
  $('wzDataUse').onclick = () => {
    choice.data = { ...detail.opts };
    const id = detail.option.id;
    $('wzDataCards').querySelectorAll('.wz-data').forEach((c) => c.classList.toggle('picked', c.dataset.id === id));
    next();
    setTimeout(() => closeDetail(true), SLIDE_MS);
  };


  // ---- network step: sliders + presets, drawn as the weight matrices themselves ------------------------
  const App = window.__app;
  const WEIGHT_SEED = 2;          // the workspace's default weight seed (defaultState), so the colours are its weights
  const SVGNS = 'http://www.w3.org/2000/svg';
  const POS = [227, 179, 65], NEG = [188, 140, 255];        // gold +, violet − (as in the layer inspector)
  let netFor = '';

  const training = () => choice.problem !== 'none';
  function recommended() {
    const d = +choice.dim, data = choice.data || {};
    const [layers, width] = App.recommendedArch(d, choice.problem, data.dataset || '');
    return { layers, width, act: 'tanh', outputLinear: training() };
  }
  const PRESETS = {
    recommended,
    shallow: () => ({ layers: 2, width: 12, act: 'tanh', outputLinear: training() }),
    deep: () => ({ layers: 6, width: Math.max(+choice.dim, recommended().width), act: 'tanh', outputLinear: training() }),
    wide: () => ({ layers: 3, width: 16, act: 'tanh', outputLinear: training() }),
  };

  function enterNetwork() {
    const key = `${choice.problem}/${choice.dim}/${JSON.stringify(choice.data)}`;
    if (key !== netFor || !choice.network) { netFor = key; choice.network = recommended(); }
    syncNetControls();
    renderNet();
  }

  function syncNetControls() {
    const n = choice.network;
    $('wzLayers').value = n.layers; $('wzLayersVal').textContent = n.layers;
    $('wzWidth').value = n.width; $('wzWidthVal').textContent = n.width;
    $('wzAct').value = n.act;
    $('wzOutLin').checked = n.outputLinear;
    $('wzWidth').disabled = n.layers === 1;            // one layer has no hidden width
  }

  $('wzAct').innerHTML = NN.ACTIVATION_NAMES.filter((a) => a !== 'identity').map((a) => `<option value="${a}">${a}</option>`).join('');
  $('wzLayers').oninput = () => { choice.network.layers = +$('wzLayers').value; syncNetControls(); renderNet(); };
  $('wzWidth').oninput = () => { choice.network.width = +$('wzWidth').value; syncNetControls(); renderNet(); };
  $('wzAct').onchange = () => { choice.network.act = $('wzAct').value; renderNet(); };
  $('wzOutLin').onchange = () => { choice.network.outputLinear = $('wzOutLin').checked; renderNet(); };
  $('wzPresets').addEventListener('click', (e) => {
    const b = e.target.closest('[data-preset]');
    if (!b) return;
    choice.network = PRESETS[b.dataset.preset]();
    syncNetControls();
    renderNet();
  });

  /** The network exactly as the workspace will build it (same dims, activations, init and seed). */
  function buildNet() {
    const n = choice.network, d = +choice.dim;
    const dims = [d, ...Array(n.layers - 1).fill(n.width), d];
    const acts = dims.slice(1).map((_, l) => (n.outputLinear && l === n.layers - 1 ? 'identity' : n.act));
    const init = App.initFor(choice.problem);
    return new NN.MLP(dims, acts).init(init.dist, init.scale, new NN.Rng(WEIGHT_SEED));
  }

  function el(tag, attrs, parent, text) {
    const e = document.createElementNS(SVGNS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    if (parent) parent.appendChild(e);
    return e;
  }

  const SUB = '₀₁₂₃₄₅₆₇₈₉';
  const sub = (k) => String(k).split('').map((c) => SUB[+c]).join('');

  /**
   * x -> [W1 | b1] -(act)-> [W2 | b2] -(act)-> ... -> y.  Every matrix is drawn cell by cell at its real
   * shape (rows = neurons out, columns = inputs), so width and depth are visible at a glance.
   */
  function renderNet() {
    const net = buildNet(), svg = $('wzNetSvg');
    svg.textContent = '';
    const maxRows = Math.max(...net.dims);
    const c = Math.max(6, Math.min(22, Math.floor(220 / maxRows)));   // cell size
    const gap = 2, arrow = 70, top = 34, H = maxRows * c + top + 34;
    let maxAbs = 1e-9;
    for (const v of net.theta) maxAbs = Math.max(maxAbs, Math.abs(v));
    const color = (v) => {
      const a = 0.18 + 0.82 * Math.min(1, Math.abs(v) / maxAbs), k = v >= 0 ? POS : NEG;
      return `rgba(${k[0]},${k[1]},${k[2]},${a.toFixed(3)})`;
    };
    const midY = top + (maxRows * c) / 2;
    let x = 10;
    const g = el('g', {}, svg);

    const vector = (n, label, sublabel) => {
      const y0 = midY - (n * c) / 2;
      for (let i = 0; i < n; i++) el('rect', { x, y: y0 + i * c, width: c - gap, height: c - gap, rx: 2, fill: '#30363d' }, g);
      el('text', { x: x + (c - gap) / 2, y: y0 - 10, 'text-anchor': 'middle', class: 'lbl' }, g, label);
      el('text', { x: x + (c - gap) / 2, y: y0 + n * c + 16, 'text-anchor': 'middle' }, g, sublabel);
      x += c;
    };
    const link = (label) => {
      const x0 = x + 6, x1 = x + arrow - 6;
      el('line', { x1: x0, y1: midY, x2: x1, y2: midY, stroke: '#8b949e', 'stroke-width': 1.2 }, g);
      el('path', { d: `M${x1} ${midY} l-6 -4 v8 z`, fill: '#8b949e' }, g);
      if (label) {
        const w = 8 + 6.4 * label.length;
        el('rect', { x: (x0 + x1) / 2 - w / 2, y: midY - 22, width: w, height: 17, rx: 8.5, fill: '#1c2128', stroke: '#30363d' }, g);
        el('text', { x: (x0 + x1) / 2, y: midY - 9.5, 'text-anchor': 'middle', class: 'pill' }, g, label);
      }
      x += arrow;
    };

    const d = net.inDim;
    vector(d, 'x', `ℝ${d === 2 ? '²' : '³'}`);
    link('');
    demoLayer = Math.min(demoLayer, net.nLayers - 1);
    net.layout.forEach((L, l) => {
      const y0 = midY - (L.nout * c) / 2;
      const lg = el('g', { class: 'wz-layer', 'data-layer': l }, g);
      // frame around W and b: highlighted for the layer the demo below shows; also the click target
      el('rect', { class: 'frame', x: x - 5, y: y0 - 27, width: (L.nin + 1) * c + 15, height: L.nout * c + 50, rx: 6,
        fill: 'transparent', stroke: l === demoLayer ? '#58a6ff' : 'transparent', 'stroke-width': 1.5 }, lg);
      for (let i = 0; i < L.nout; i++) {
        for (let j = 0; j < L.nin; j++) {
          el('rect', { x: x + j * c, y: y0 + i * c, width: c - gap, height: c - gap, rx: 2, fill: color(net.theta[L.w + i * L.nin + j]) }, lg);
        }
        el('rect', { x: x + L.nin * c + 5, y: y0 + i * c, width: c - gap, height: c - gap, rx: 2, fill: color(net.theta[L.b + i]) }, lg);
      }
      const wMid = x + (L.nin * c) / 2;
      el('text', { x: wMid, y: y0 - 10, 'text-anchor': 'middle', class: 'lbl' }, lg, `W${sub(l + 1)}`);
      el('text', { x: x + L.nin * c + 5 + (c - gap) / 2, y: y0 - 10, 'text-anchor': 'middle' }, lg, `b${sub(l + 1)}`);
      el('text', { x: wMid + 2.5 + c / 2, y: y0 + L.nout * c + 16, 'text-anchor': 'middle' }, lg, `${L.nout}×${L.nin}`);
      x += (L.nin + 1) * c + 5;
      const act = net.acts[l];
      link(act === 'identity' ? 'linear' : act);
    });
    vector(net.outDim, 'y', `ℝ${net.outDim === 2 ? '²' : '³'}`);
    svg.setAttribute('viewBox', `0 0 ${x + 10} ${H}`);
    svg.style.maxWidth = `${Math.max(320, (x + 10) * 1.25)}px`;

    const rec = recommended(), n = choice.network;
    const isRec = n.layers === rec.layers && (n.layers === 1 || n.width === rec.width) && n.act === 'tanh' && n.outputLinear === rec.outputLinear;
    $('wzNetInfo').innerHTML = `${net.dims.join(' → ')} &nbsp;·&nbsp; <b>${net.nParams}</b> parameters`
      + (isRec ? ' &nbsp;<span class="tag">★ recommended</span>' : '');
    $('wzPresets').querySelectorAll('[data-preset]').forEach((b) => {
      const p = PRESETS[b.dataset.preset]();
      b.classList.toggle('on', p.layers === n.layers && (n.layers === 1 || p.width === n.width) && p.act === n.act && p.outputLinear === n.outputLinear);
    });
    $('wzNetNotes').innerHTML = netNotes(n).map((t) => `<li>${t}</li>`).join('');
    $('wzActEq').textContent = ACT_EQ[n.act];
    renderLayerDemo(net);
  }

  /** Plain-language warnings about what this network cannot do on the chosen problem. */
  function netNotes(n) {
    const d = +choice.dim, w = n.layers === 1 ? d : n.width, data = choice.data || {}, out = [];
    if (n.layers > 1 && w < d) {
      out.push(`Hidden width ${w} is smaller than the input dimension ${d}: the first matrix flattens space, and points that land on the same spot can never be told apart again.`);
    }
    const enclosed = { 2: ['circles', 'rings', 'shape:diskInRing'], 3: ['circles', 'rings', 'linked', 'shape:linkedRings', 'shape:nestedSpheres'] };
    if (choice.problem === 'classify' && enclosed[d].includes(data.dataset) && w <= d) {
      out.push(`With width ${w} every layer can at most bend ${d}D space without tearing it, so one class stays trapped around the other. Width ${d === 2 ? '3' : '4'} or more gives the network room to lift it out.`);
    }
    const glue = ['circle>figure8', 'twoCircles>oneCircleTwice', 'torus>torusOnSphere'];
    if (choice.problem === 'morph' && glue.includes(data.pair) && w <= d) {
      out.push(`This target glues points together. A network of width ${d} can only come close; width ${d + 1} is enough.`);
    }
    const bounded = { tanh: '−1…1', sigmoid: '0…1', sin: '−1…1' }[n.act];
    if (training() && !n.outputLinear && bounded) {
      out.push(`The last layer ends in ${n.act}, so every output stays in ${bounded}. Targets outside that range cannot be reached: tick “Linear output layer”.`);
    }
    if (n.act === 'relu' && w <= d && n.layers > 1) {
      out.push('ReLU at this width sets whole regions of space to zero: many points get squashed onto the same edge.');
    }
    return out;
  }


  // ---- activation formula and the one-layer demo --------------------------------------------------------
  const ACT_EQ = {
    tanh: 'σ(z) = tanh(z) = (eᶻ − e⁻ᶻ) / (eᶻ + e⁻ᶻ)',
    relu: 'σ(z) = max(0, z)',
    sigmoid: 'σ(z) = 1 / (1 + e⁻ᶻ)',
    gelu: 'σ(z) = ½ z (1 + tanh(√(2/π) (z + 0.044715 z³)))',
    sin: 'σ(z) = sin(z)',
    identity: 'σ(z) = z   (linear: no bending)',
  };
  let demoLayer = 0;            // which layer the demo shows (click a matrix in the diagram)

  /**
   * The top-left 2 x 2 block of layer `demoLayer` and the first two biases, as a 2D map the eye can follow.
   * A layer with only one input or output is padded with zeros (it squashes the plane onto a line).
   */
  function renderLayerDemo(net) {
    demoLayer = Math.min(demoLayer, net.nLayers - 1);
    const L = net.layout[demoLayer], th = net.theta;
    const w = (i, j) => (i < L.nout && j < L.nin ? th[L.w + i * L.nin + j] : 0);
    const W = [[w(0, 0), w(0, 1)], [w(1, 0), w(1, 1)]];
    const b = [0, 1].map((i) => (i < L.nout ? th[L.b + i] : 0));
    const act = net.acts[demoLayer];
    let maxAbs = 1e-9;
    for (const v of th) maxAbs = Math.max(maxAbs, Math.abs(v));
    const cell = (v) => {
      const a = 0.18 + 0.82 * Math.min(1, Math.abs(v) / maxAbs), k = v >= 0 ? POS : NEG;
      return `<span style="background:rgba(${k[0]},${k[1]},${k[2]},${(0.75 * a).toFixed(3)})">${v.toFixed(2)}</span>`;
    };
    const dots = (t) => `<span class="dots">${t}</span>`;
    const moreCols = L.nin > 2, moreRows = L.nout > 2;
    let wh = '';
    for (let i = 0; i < 2; i++) wh += cell(W[i][0]) + cell(W[i][1]) + (moreCols ? dots('⋯') : '');
    if (moreRows) wh += dots('⋮') + dots('⋮') + (moreCols ? dots('⋱') : '');
    $('wzLdW').style.gridTemplateColumns = `repeat(${moreCols ? 3 : 2}, auto)`;
    $('wzLdW').innerHTML = wh;
    $('wzLdB').style.gridTemplateColumns = 'auto';
    $('wzLdB').innerHTML = cell(b[0]) + cell(b[1]) + (moreRows ? dots('⋮') : '');
    $('wzLdWhich').textContent = `layer ${demoLayer + 1} of ${net.nLayers} · click a matrix above`;
    const shape = `${L.nout}×${L.nin}`;
    $('wzLdNote').textContent = L.nout === 2 && L.nin === 2
      ? `W${sub(demoLayer + 1)} is exactly 2×2: the animation is the whole layer.`
      : `W${sub(demoLayer + 1)} is ${shape}. The animation uses its top-left 2×2 block: the first two neurons, fed by the first two inputs${L.nout < 2 || L.nin < 2 ? ' (missing entries are 0)' : ''}.`;
    $('wzLdAct').textContent = ACT_EQ[act];
    const canvas = $('wzLdCanvas');
    players.set(canvas, { key: 'layer', dim: 2, player: Minis.layerPlayer(canvas, Minis.layerScene(W, b, act)), onStage: showStage });
  }

  function showStage(k) {
    $('wzLdStages').querySelectorAll('[data-s]').forEach((e) => e.classList.toggle('on', +e.dataset.s === k));
  }

  $('wzNetSvg').addEventListener('click', (e) => {
    const g = e.target.closest('g.wz-layer');
    if (!g) return;
    demoLayer = +g.dataset.layer;
    renderNet();
  });

  $('wzFinish').onclick = () => {
    App.applyGuided({ dim: +choice.dim, problem: choice.problem, data: choice.data, network: choice.network });
    close();
  };

  function renderStepBar() {
    const i = STEPS.indexOf(current);
    $('wzSteps').classList.toggle('hidden', i === 0);
    $('wzBack').classList.toggle('hidden', i === 0);
    root.querySelectorAll('#wzSteps [data-goto]').forEach((b) => {
      const j = STEPS.indexOf(b.dataset.goto);
      b.classList.toggle('now', j === i);
      b.classList.toggle('done', j < i);
      const skipped = b.dataset.goto === 'data' && choice.problem === 'none';
      b.classList.toggle('skip', skipped);
      b.disabled = j > reached || j === i || skipped;
    });
  }

  function show(step) {
    if (step === current) return;
    const from = screen(current), to = screen(step);
    const forward = STEPS.indexOf(step) > STEPS.indexOf(current);
    if (step === 'data') buildDataList();
    if (step === 'network') enterNetwork();
    current = step;
    reached = Math.max(reached, STEPS.indexOf(step));
    from.classList.remove('active');
    from.classList.add(forward ? 'leave-left' : 'leave-right');
    to.classList.add(forward ? 'enter-right' : 'enter-left');
    void to.offsetWidth;                                   // commit the start position before animating
    to.classList.remove('enter-right', 'enter-left');
    to.querySelectorAll('.wz-card').forEach((c, i) => c.style.setProperty('--i', Math.min(i, 8)));
    to.classList.add('active');
    setTimeout(() => from.classList.remove('leave-left', 'leave-right'), SLIDE_MS);
    renderStepBar();
    const focus = to.querySelector('.wz-card, button');
    if (focus) focus.focus({ preventScroll: true });
  }

  const skipped = (step) => step === 'data' && choice.problem === 'none';   // "just look" needs no data

  function next() {
    let i = STEPS.indexOf(current) + 1;
    if (skipped(STEPS[i])) i++;
    show(STEPS[i]);
  }

  function back() {
    if (current === 'data' && detail) { closeDetail(); return; }
    let i = STEPS.indexOf(current) - 1;
    if (skipped(STEPS[i])) i--;
    if (i >= 0) show(STEPS[i]);
  }

  function open() {
    document.body.classList.add('wizard-open');
    root.classList.remove('hidden', 'closing');
    STEPS.forEach((s) => screen(s).classList.toggle('active', s === current));
    renderStepBar();
    startAnimation();
    if (!building) buildNext();
  }

  function close() {
    root.classList.add('closing');
    setTimeout(() => {
      root.classList.add('hidden');
      document.body.classList.remove('wizard-open');
      window.dispatchEvent(new Event('resize'));          // the workspace was laid out behind the overlay
    }, SLIDE_MS);
  }

  // picking a card: remember it, mark it, move on
  root.addEventListener('click', (e) => {
    const card = e.target.closest('.wz-card[data-key]');
    if (card) {
      const sec = card.closest('.wz-screen');
      sec.querySelectorAll('.wz-card').forEach((c) => c.classList.toggle('picked', c === card));
      choice[card.dataset.key] = card.dataset.value;
      if (card.dataset.key === 'dim') bindProblemCards(+card.dataset.value, true);
      if (card.dataset.key === 'dim' || card.dataset.key === 'problem') reached = Math.min(reached, STEPS.indexOf(current) + 1);
      next();
      return;
    }
    const go = e.target.closest('[data-goto]');
    if (go && !go.disabled) show(go.dataset.goto);
  });

  $('wzStart').onclick = () => show('dimension');
  $('wzSkip').onclick = close;
  $('wzSkipTop').onclick = close;
  $('wzBack').onclick = back;
  $('guideBtn').onclick = () => { current = 'intro'; open(); };

  document.addEventListener('keydown', (e) => {
    if (!document.body.classList.contains('wizard-open')) return;
    if (e.key === 'Escape' || (e.key === 'Backspace' && !/input|select|textarea/i.test(e.target.tagName))) {
      e.preventDefault();
      back();
    }
  });

  ['classify', 'transform', 'morph', 'none'].forEach((p) => want(`${p}/2`));
  ['classify', 'transform', 'morph'].forEach((p) => want(`${p}/3`));

  window.Wizard = { choice, show, close, open, scenes };
  if (location.hash === '#workspace') root.classList.add('hidden');
  else open();
})();
