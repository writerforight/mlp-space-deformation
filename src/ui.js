/*
 * ui.js — application state, controls, the per-frame loop, training and analysis panels.
 *
 * Data flow:
 *   state S  ──►  net (NN.MLP)  ──►  traces (every drawable pushed through every stage, projected
 *   to 2D/3D)  ──►  items for Viz2D / Viz3D at the current (fractional) stage t.
 * Traces are recomputed only when something changes (or every N training steps); the animation just
 * interpolates between cached stages, so it stays smooth.
 */
(function () {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const ACTS = NN.ACTIVATION_NAMES;
  const PALETTE = ['#58a6ff', '#f78166', '#d2a8ff', '#3fb950', '#ffa657', '#ff7b72', '#79c0ff', '#e3b341'];
  const CLASS_COLORS = ['#58a6ff', '#ff6b6b'];   // class 1 = blue, class 2 = red
  const BASIS_COLORS = ['#ff6b6b', '#51cf66', '#4dabf7'];
  const GRID_COLOR = 'rgba(150,160,180,0.35)';
  const STATE_VERSION = 1;

  // ===========================================================================
  // State
  // ===========================================================================

  function defaultState(dim = 2) {
    return {
      version: STATE_VERSION,
      dim,
      net: { layers: 3, width: dim, defaultAct: 'tanh', temperature: 0, outputLinear: false,
        overrides: [], init: { dist: 'he', scale: 1.6, seed: 2 } },
      objects: [],
      nPoints: 300,
      display: { grid: true, circle: true, basis: true, origin: true, jac: false, det: false, dist: false, autoFit: false },
      train: { target: 'none', dataset: 'moons', transform: 'rotation', amount: 0.9, optimizer: 'adam', lr: 0.01, batch: 32, stepsPerFrame: 5,
        redrawEvery: 10, mode: 'joint', stepsPerTask: 300, nTasks: 3, method: 'none', lambda: 100 },
      pins: [],
      probe: dim === 2 ? [0.5, 0.3] : [0.5, 0.3, 0.2],
      analysis: { matrix: 'cosine', eps: 0.1 },
      sphere: { c: [0, 0, 0], r: 0.6 },
    };
  }

  let S = defaultState(2);
  let net = null, trainer = null, data = [], targetA = null;
  let traces = null;
  let tool = 'pan';
  let t = 0, tGoal = null, playing = false;
  let training = false, stepsSinceRedraw = 0;
  let analysisDirty = true, lastAnalysis = 0, lastLossDraw = 0;
  let drawing = null;           // shape being drawn with the mouse
  let objId = 1;
  let bgCache = null;
  let netVersion = 0;

  const viz2 = new Viz2D($('c2d'));
  let viz3 = null;              // created lazily (needs WebGL)

  // ===========================================================================
  // Network
  // ===========================================================================

  /** Activation of every layer: manual override > linear output > temperature rule. */
  function resolveActs() {
    const L = S.net.layers, r = new NN.Rng(S.net.init.seed * 31 + 7), acts = [];
    for (let l = 0; l < L; l++) {
      const u = r.next(), pick = ACTS[r.int(ACTS.length)];
      const ov = S.net.overrides[l];
      if (ov && ov !== 'auto') acts.push(ov);
      else if (S.net.outputLinear && l === L - 1) acts.push('identity');
      else acts.push(u < S.net.temperature ? pick : S.net.defaultAct);
    }
    return acts;
  }

  function dimsOf() {
    const dims = [S.dim];
    for (let l = 0; l < S.net.layers - 1; l++) dims.push(S.net.width);
    dims.push(S.dim);
    return dims;
  }

  /** Rebuild the network.  keepWeights: only the activations changed. */
  function rebuildNet(keepWeights = false, theta = null) {
    const dims = dimsOf(), acts = resolveActs();
    const same = net && net.dims.length === dims.length && net.dims.every((d, i) => d === dims[i]);
    if (keepWeights && same) {
      net.acts = acts;
    } else {
      net = new NN.MLP(dims, acts).init(S.net.init.dist, S.net.init.scale, new NN.Rng(S.net.init.seed));
      if (theta && theta.length === net.nParams) net.theta.set(theta);
      rebuildTrainer();
    }
    netVersion++;
    renderLayerActs();
    if ($('targetNote')) $('targetNote').textContent = targetNote() + outputWarning();
    invalidate();
  }

  // ===========================================================================
  // Training targets and trainer
  // ===========================================================================

  /*
   * Training tasks:
   *   anchors   — send every blue point to (1, 0[, 0]) and every red point to (−1, 0[, 0])  (MSE)
   *   classify  — two output logits, cross-entropy                                         (CE)
   *   transform — every x should go to A·x for a fixed rotation / shear / scaling A         (MSE)
   *   pins      — user-dragged input → output pairs                                        (MSE)
   */
  const isClassify = () => S.train.target === 'classify';
  const isAnchors = () => S.train.target === 'anchors';
  const isTransform = () => S.train.target === 'transform';
  const usesClasses = () => isClassify() || isAnchors();

  /** Target point of a class for the "anchors" task: blue → +e1, red → −e1. */
  function anchor(c) {
    const v = new Float64Array(S.dim);
    v[0] = c === 0 ? 1 : -1;
    return v;
  }

  /** Fraction of samples on the right side: argmax of the logits, or the nearer anchor. */
  function accuracy() {
    if (!usesClasses() || !data.length) return null;
    let ok = 0;
    for (const s of data) {
      const o = net.predict(s.x);
      const pred = isClassify() ? (o[1] > o[0] ? 1 : 0) : (o[0] >= 0 ? 0 : 1);
      if (pred === s.label) ok++;
    }
    return ok / data.length;
  }

  function buildData() {
    const rng = new NN.Rng(S.net.init.seed * 101 + 5);
    targetA = null;
    if (usesClasses()) {
      data = NN.makeDataset(S.train.dataset, 400, S.dim, rng).map((s) => ({
        x: s.x, label: s.y, y: isAnchors() ? anchor(s.y) : s.y }));
    } else if (isTransform()) {
      targetA = NN.targetMatrix(S.train.transform, S.dim, S.train.amount);
      data = NN.makeTransformData(targetA, 300, S.dim, rng);
    } else data = [];
  }

  /** Split samples into k chunks by angle around the origin (a natural "continual learning" order). */
  function chunkByAngle(samples, k) {
    const sorted = samples.slice().sort((a, b) => Math.atan2(a.x[1], a.x[0]) - Math.atan2(b.x[1], b.x[0]));
    const out = [];
    for (let i = 0; i < k; i++) out.push(sorted.slice(Math.floor((i * sorted.length) / k), Math.floor(((i + 1) * sorted.length) / k)));
    return out.filter((c) => c.length);
  }

  function buildTasks() {
    if (S.train.target === 'pins') return S.pins.map((p) => [{ x: Float64Array.from(p.x), y: Float64Array.from(p.y) }]);
    if (!data.length) return [];
    return S.train.mode === 'sequential' ? chunkByAngle(data, S.train.nTasks) : [data];
  }

  function trainerOpts() {
    const T = S.train;
    return { type: isClassify() ? 'ce' : 'mse', optimizer: T.optimizer, lr: T.lr, batch: T.batch, mode: T.mode,
      stepsPerTask: T.stepsPerTask, method: T.method, ewcLambda: T.lambda, seed: S.net.init.seed };
  }

  function rebuildTrainer() {
    buildData();
    trainer = new NN.Trainer(net, trainerOpts());
    trainer.setTasks(buildTasks());
    analysisDirty = true;
    updateTrainStatus();
  }

  // ===========================================================================
  // Geometry: drawables (reference objects + user objects)
  // ===========================================================================

  const vec = (...a) => Float64Array.from(a);
  const linspace = (a, b, n) => [...Array(n)].map((_, i) => a + ((b - a) * i) / (n - 1));

  function referenceDrawables() {
    const d = S.dim, D = S.display, out = [];
    if (D.grid) {
      if (d === 2) {
        for (const c of linspace(-2, 2, 9)) {
          out.push({ id: `gx${c}`, role: 'grid', kind: 'line', color: GRID_COLOR, width: 1, pts: linspace(-2, 2, 41).map((u) => vec(c, u)) });
          out.push({ id: `gy${c}`, role: 'grid', kind: 'line', color: GRID_COLOR, width: 1, pts: linspace(-2, 2, 41).map((u) => vec(u, c)) });
        }
      } else {
        for (const a of [-1, 0, 1]) for (const b of [-1, 0, 1]) {
          const ln = linspace(-1.5, 1.5, 31);
          out.push({ id: `g3x${a}${b}`, role: 'grid', kind: 'line', color: GRID_COLOR, pts: ln.map((u) => vec(u, a, b)) });
          out.push({ id: `g3y${a}${b}`, role: 'grid', kind: 'line', color: GRID_COLOR, pts: ln.map((u) => vec(a, u, b)) });
          out.push({ id: `g3z${a}${b}`, role: 'grid', kind: 'line', color: GRID_COLOR, pts: ln.map((u) => vec(a, b, u)) });
        }
      }
    }
    if (D.circle) {
      const ring = (f) => linspace(0, 2 * Math.PI, 121).slice(0, 120).map(f);
      if (d === 2) out.push({ id: 'unit', role: 'circle', kind: 'line', closed: true, color: '#e6edf3', alpha: 0.85, width: 1.6, pts: ring((a) => vec(Math.cos(a), Math.sin(a))) });
      else {
        for (const lat of [-60, -30, 0, 30, 60]) {
          const z = Math.sin((lat * Math.PI) / 180), r = Math.cos((lat * Math.PI) / 180);
          out.push({ id: `lat${lat}`, role: 'circle', kind: 'line', closed: true, color: '#e6edf3', alpha: 0.55, pts: ring((a) => vec(r * Math.cos(a), r * Math.sin(a), z)) });
        }
        for (const lon of [0, 45, 90, 135]) {
          const c = Math.cos((lon * Math.PI) / 180), s = Math.sin((lon * Math.PI) / 180);
          out.push({ id: `lon${lon}`, role: 'circle', kind: 'line', closed: true, color: '#e6edf3', alpha: 0.55, pts: ring((a) => vec(c * Math.cos(a), s * Math.cos(a), Math.sin(a))) });
        }
      }
    }
    if (D.basis) {
      for (let i = 0; i < d; i++) {
        out.push({ id: `e${i}`, role: 'basis', kind: 'line', color: BASIS_COLORS[i], width: 2.6, label: `e${i + 1}`,
          pts: linspace(0, 1, 30).map((u) => { const v = new Float64Array(d); v[i] = u; return v; }) });
      }
    }
    if (D.origin) out.push({ id: 'origin', role: 'origin', kind: 'points', color: '#ffffff', size: 4, pts: [new Float64Array(d)] });
    return out;
  }

  function userDrawables() {
    const out = [];
    for (const o of S.objects) {
      // Fibonacci spheres are one long spiral: draw it lighter so the surface stays readable
      out.push({ id: `o${o.id}`, role: 'object', kind: 'line', closed: o.closed, color: o.color, width: 2,
        alpha: o.type === 'sphere' ? 0.6 : 1, pts: o.points.map((p) => Float64Array.from(p)) });
      if (o.interior && o.interior.length) {
        out.push({ id: `oi${o.id}`, role: 'objectFill', kind: 'points', color: o.color, size: 1.6, alpha: 0.75, pts: o.interior.map((p) => Float64Array.from(p)) });
      }
    }
    return out;
  }

  /** Points where Jacobian ellipses are drawn: a lattice plus a few points of each object. */
  function jacobianProbePoints() {
    const pts = [];
    if (S.dim === 2) for (const a of [-2, -1, 0, 1, 2]) for (const b of [-2, -1, 0, 1, 2]) pts.push(vec(a, b));
    else for (const a of [-1, 0, 1]) for (const b of [-1, 0, 1]) for (const c of [-1, 0, 1]) pts.push(vec(a, b, c));
    for (const o of S.objects) {
      const k = Math.max(1, Math.floor(o.points.length / 12));
      for (let i = 0; i < o.points.length; i += k) pts.push(Float64Array.from(o.points[i]));
    }
    return pts;
  }

  function allDrawables() {
    const ds = referenceDrawables().concat(userDrawables());
    if (usesClasses() && data.length) {
      ds.push({ id: 'data', role: 'data', kind: 'points', size: 2.4, colors: data.map((s) => CLASS_COLORS[s.label]), pts: data.map((s) => s.x) });
    }
    if (S.pins.length) ds.push({ id: 'pins', role: 'pins', kind: 'points', pts: S.pins.map((p) => Float64Array.from(p.x)) });
    ds.push({ id: 'probe', role: 'probe', kind: 'points', pts: [Float64Array.from(S.probe)] });
    if (S.display.jac) ds.push({ id: 'jac', role: 'jac', kind: 'none', pts: jacobianProbePoints() });
    return ds;
  }

  // ===========================================================================
  // Traces: push every point through every stage and project to the view dimension
  // ===========================================================================

  /**
   * For each drawable and stage s, stores the projected positions proj[s] (n × vd).  Stages whose
   * dimension exceeds the view dimension vd are projected onto their top-vd principal components;
   * axis signs are aligned with the previous stage so the morph does not flip.  When Jacobian
   * overlays are on, also stores P_s·J_s (the projected Jacobian of the stage w.r.t. the input).
   */
  function computeTraces() {
    const vd = S.dim, nStages = 2 * net.nLayers + 1;
    const needJ = S.display.jac || S.display.det;
    const ds = allDrawables();
    for (const d of ds) {
      const wantJ = d.role === 'jac' || (S.display.det && (d.kind === 'line' || d.role === 'objectFill'));
      d.st = []; d.jac = [];
      for (const p of d.pts) {
        if (needJ && wantJ) { const r = net.stageJacobians(p); d.st.push(r.stages); d.jac.push(r.jacs); }
        else d.st.push(net.stages(p));
      }
      d.hasJ = needJ && wantJ;
      d.proj = []; d.pj = [];
    }
    const bases = [];
    for (let s = 0; s < nStages; s++) {
      // collect a sample of all points at this stage for PCA
      const cloud = [];
      for (const d of ds) {
        if (d.role === 'probe' || d.role === 'jac') continue;
        const k = Math.max(1, Math.floor(d.st.length / 120));
        for (let i = 0; i < d.st.length; i += k) cloud.push(d.st[i][s]);
      }
      if (!cloud.length) cloud.push(new Float64Array(net.dims[0]));
      const P = NN.pcaBasis(cloud, vd);
      const project = () => {
        for (const d of ds) {
          const n = d.st.length, out = new Float64Array(n * vd);
          for (let i = 0; i < n; i++) {
            const x = d.st[i][s];
            for (let a = 0; a < vd; a++) { let v = 0; const b = P.basis[a]; for (let j = 0; j < x.length; j++) v += b[j] * x[j]; out[i * vd + a] = v; }
          }
          d.proj[s] = out;
        }
      };
      project();
      if (!P.exact && s > 0) {            // align axis signs with the previous stage
        const corr = new Float64Array(vd);
        for (const d of ds) for (let i = 0; i < d.proj[s].length; i++) corr[i % vd] += d.proj[s][i] * d.proj[s - 1][i];
        let flipped = false;
        for (let a = 0; a < vd; a++) if (corr[a] < 0) { P.basis[a] = P.basis[a].map((v) => -v); flipped = true; }
        if (flipped) project();
      }
      bases.push(P);
      for (const d of ds) {
        if (!d.hasJ) continue;
        const n = d.st.length, d0 = vd, out = new Float64Array(n * vd * d0);
        for (let i = 0; i < n; i++) {
          const J = d.jac[i][s];       // rows: stage dim, cols: input dim
          for (let a = 0; a < vd; a++) for (let c = 0; c < d0; c++) {
            let v = 0; const b = P.basis[a];
            for (let j = 0; j < J.length; j++) v += b[j] * J[j][c];
            out[(i * vd + a) * d0 + c] = v;
          }
        }
        d.pj[s] = out;
      }
    }
    traces = { ds, nStages, bases, vd };
    bgCache = null;
    if (t > nStages - 1) { t = nStages - 1; anim = null; }
  }

  function invalidate() {
    traces = null;
    analysisDirty = true;
  }

  // ===========================================================================
  // Interpolation at the fractional stage t
  // ===========================================================================

  function stageSplit(tt) {
    const n = traces.nStages;
    if (n === 1) return [0, 0];
    const s = Math.min(n - 2, Math.max(0, Math.floor(tt)));
    return [s, Math.min(1, Math.max(0, tt - s))];
  }

  function positionsAt(d, tt) {
    const [s, f] = stageSplit(tt), a = d.proj[s], b = d.proj[Math.min(s + 1, traces.nStages - 1)];
    const out = new Float64Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] + f * (b[i] - a[i]);
    return out;
  }

  function jacAt(d, tt) {
    const [s, f] = stageSplit(tt), a = d.pj[s], b = d.pj[Math.min(s + 1, traces.nStages - 1)];
    const out = new Float64Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] + f * (b[i] - a[i]);
    return out;
  }

  const det2 = (m, o) => m[o] * m[o + 3] - m[o + 1] * m[o + 2];
  const det3 = (m, o) => m[o] * (m[o + 4] * m[o + 8] - m[o + 5] * m[o + 7]) - m[o + 1] * (m[o + 3] * m[o + 8] - m[o + 5] * m[o + 6])
    + m[o + 2] * (m[o + 3] * m[o + 7] - m[o + 4] * m[o + 6]);

  function detColors(d, tt) {
    const J = jacAt(d, tt), vd = traces.vd, k = vd * vd, n = J.length / k, out = [];
    for (let i = 0; i < n; i++) {
      const v = vd === 2 ? det2(J, i * k) : det3(J, i * k);
      out.push(v > 1e-6 ? '#3fb950' : v < -1e-6 ? '#f85149' : '#8b949e');
    }
    return out;
  }

  // ===========================================================================
  // Building the items for the renderers
  // ===========================================================================

  function stageName(s) {
    if (s === 0) return { title: 'Input space', sub: 'the objects before the network' };
    const l = Math.ceil(s / 2), act = net.acts[l - 1];
    return s % 2 === 1 ? { title: `Layer ${l} · linear`, sub: `z = W${l} a + b${l}  (${net.dims[l - 1]}→${net.dims[l]})` }
      : { title: `Layer ${l} · ${act}`, sub: `a = ${act}(z)${l === net.nLayers ? ' — output' : ''}` };
  }

  function buildItems(tt) {
    const items = [], vd = traces.vd, D = S.display, last = traces.nStages - 1;
    const atOutput = tt >= last - 1e-6, atInput = tt <= 1e-6;
    // distance colouring needs a global maximum
    let distMax = 0;
    const posCache = new Map();
    for (const d of traces.ds) {
      if (d.role === 'jac') continue;
      const p = positionsAt(d, tt);
      posCache.set(d, p);
      if (D.dist) for (let i = 0; i < d.pts.length; i++) {
        let s = 0; for (let a = 0; a < vd; a++) s += (p[i * vd + a] - d.pts[i][a]) ** 2;
        distMax = Math.max(distMax, Math.sqrt(s));
      }
    }
    if (vd === 2 && usesClasses() && tt < 0.5) items.push(backgroundImage());
    for (const d of traces.ds) {
      if (d.role === 'jac' || d.role === 'probe' || d.role === 'pins') continue;
      const p = posCache.get(d);
      let colors = d.colors;
      if (D.det && d.hasJ && d.role !== 'data') colors = detColors(d, tt);
      else if (D.dist && d.role !== 'data') {
        colors = d.pts.map((x, i) => {
          let s = 0; for (let a = 0; a < vd; a++) s += (p[i * vd + a] - x[a]) ** 2;
          return Colors.seqColor(distMax > 0 ? Math.sqrt(s) / distMax : 0);
        });
      }
      items.push({ id: d.id, kind: d.kind, pts: p, closed: d.closed, color: d.color, colors, width: d.width, alpha: d.alpha, size: d.size });
      if (d.role === 'basis' && vd === 2 && p.length >= 4) {
        const n = p.length / 2;
        items.push({ kind: 'arrow', from: [p[2 * n - 4], p[2 * n - 3]], to: [p[2 * n - 2], p[2 * n - 1]], color: d.color, width: 2.6 });
        items.push({ kind: 'label', p: [p[2 * n - 2], p[2 * n - 1]], text: d.label, color: d.color });
      }
    }
    // the target transform as a ghost, at the output stage
    if (atOutput && targetA) {
      for (const d of traces.ds) {
        if (d.role !== 'object' && d.role !== 'circle') continue;
        const g = new Float64Array(d.pts.length * vd);
        d.pts.forEach((x, i) => NN.matVec(targetA, x).forEach((v, a) => { g[i * vd + a] = v; }));
        items.push({ id: `${d.id}_target`, kind: 'line', pts: g, closed: d.closed, color: d.color || '#e6edf3', alpha: 0.35, dash: [5, 5], width: 1.5, noFit: true });
      }
    }
    // the decision boundary (equal logits) lives on the diagonal of the output plane
    if (atOutput && isClassify() && vd === 2) {
      const M = 1e3;
      items.push({ kind: 'line', pts: Float64Array.from([-M, -M, M, M]), color: '#e6edf3', alpha: 0.5, dash: [6, 6], width: 1.2, noFit: true });
      items.push({ kind: 'label', p: [0.6, 0.6], text: 'decision boundary (logit₁ = logit₂)', color: '#8b949e' });
    }
    // the two target points of the "anchors" task, and the line halfway between them
    if (atOutput && isAnchors()) {
      if (vd === 2) {
        items.push({ kind: 'line', pts: Float64Array.from([0, -1e3, 0, 1e3]), color: '#e6edf3', alpha: 0.35, dash: [6, 6], width: 1.2, noFit: true });
      }
      [0, 1].forEach((c) => {
        const a = Array.from(anchor(c));
        items.push({ id: `anchor${c}`, kind: 'marker', p: a, color: CLASS_COLORS[c], size: 9, ring: true });
        if (vd === 2) items.push({ kind: 'label', p: a, text: `${c === 0 ? 'blue' : 'red'} → (${a.map((v) => (v < 0 ? '−1' : v > 0 ? '1' : '0')).join(', ')})`, color: CLASS_COLORS[c] });
      });
    }
    // Jacobian ellipses
    const jd = traces.ds.find((d) => d.role === 'jac');
    if (jd && D.jac) {
      const c = positionsAt(jd, tt), J = jacAt(jd, tt), k = vd * vd, mats = [];
      for (let i = 0; i < jd.pts.length; i++) mats.push(Array.from(J.subarray(i * k, (i + 1) * k)));
      items.push({ id: 'jacEll', kind: 'ellipses', centers: c, mats, r: vd === 2 ? 0.12 : 0.1, color: 'rgba(255,214,102,0.85)' });
    }
    // pins: input (hollow), target (ring), current position (dot) and the remaining error
    const pd = traces.ds.find((d) => d.role === 'pins');
    if (pd) {
      const cur = posCache.get(pd);
      S.pins.forEach((pin, i) => {
        const col = PALETTE[i % PALETTE.length], c = Array.from(cur.subarray(i * vd, (i + 1) * vd));
        items.push({ id: `pinErr${i}`, kind: vd === 2 ? 'arrow' : 'segments', from: c, to: pin.y, pts: Float64Array.from([...c, ...pin.y]), color: col, dash: [4, 4], width: 1.2, alpha: 0.7 });
        items.push({ id: `pinIn${i}`, kind: 'marker', p: pin.x, color: 'rgba(230,237,243,0.5)', size: 3, ring: true });
        items.push({ id: `pinT${i}`, kind: 'marker', p: pin.y, color: col, size: 6, ring: true });
        items.push({ id: `pinC${i}`, kind: 'marker', p: c, color: col, size: 4 });
        if (vd === 2) items.push({ kind: 'label', p: pin.y, text: `pin ${i + 1}`, color: col });
      });
    }
    // probe point used by the sensitivity panel
    const pr = traces.ds.find((d) => d.role === 'probe');
    if (pr) items.push({ id: 'probe', kind: 'marker', p: Array.from(posCache.get(pr)), color: '#ffd666', size: 5, ring: true });
    // shape currently being drawn
    if (drawing) items.push(...drawingPreview());
    return { items, atInput, atOutput };
  }

  /** Predicted class probability over the visible input region (classification targets, 2D). */
  function backgroundImage() {
    const box = viz2.viewBox(), key = box.map((v) => v.toFixed(3)).join() + netVersion + trainer.step_;
    if (bgCache && bgCache.key === key) return bgCache.item;
    const W = 90, H = Math.max(10, Math.round((W * (box[3] - box[1])) / (box[2] - box[0])));
    const rgba = new Uint8ClampedArray(W * H * 4), c0 = [88, 166, 255], c1 = [255, 107, 107];
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const x = box[0] + ((i + 0.5) * (box[2] - box[0])) / W, y = box[3] - ((j + 0.5) * (box[3] - box[1])) / H;
        const o = net.predict([x, y]);
        // "red-ness": softmax probability (classify) or which anchor the point is sent closer to (anchors)
        const p1 = isClassify() ? 1 / (1 + Math.exp(o[0] - o[1])) : 1 / (1 + Math.exp(4 * o[0]));
        const k = 4 * (j * W + i);
        for (let c = 0; c < 3; c++) rgba[k + c] = c0[c] + p1 * (c1[c] - c0[c]);
        rgba[k + 3] = 26 + 60 * Math.abs(p1 - 0.5);
      }
    }
    bgCache = { key, item: { kind: 'image', bbox: box, w: W, h: H, rgba } };
    return bgCache.item;
  }

  // ===========================================================================
  // Rendering
  // ===========================================================================

  const hexToRgb = (h) => {
    if (h.startsWith('rgb')) { const m = h.match(/[\d.]+/g).map(Number); return [m[0] / 255, m[1] / 255, m[2] / 255]; }
    const v = parseInt(h.slice(1), 16);
    return [(v >> 16 & 255) / 255, (v >> 8 & 255) / 255, (v & 255) / 255];
  };

  function render() {
    if (!traces) computeTraces();
    t = Math.min(traces.nStages - 1, Math.max(0, t));
    const { items } = buildItems(t);
    if (S.display.autoFit) autoFit(items);
    if (S.dim === 2) viz2.render(items);
    else render3D(items);
    renderStageUI();
  }

  function render3D(items) {
    if (!viz3) return;
    const out = [];
    for (const it of items) {
      const id = it.id || `anon${out.length}`;
      if (it.kind === 'line' || it.kind === 'points' || it.kind === 'segments') {
        if (!it.pts || !it.pts.length) continue;
        out.push({ id, kind: it.kind, pts: it.pts, closed: it.closed, color: it.color ? cssColor3(it.color) : undefined,
          colors: it.colors ? it.colors.map(hexToRgb) : undefined, alpha: it.alpha ?? (it.color && it.color.startsWith('rgba') ? 0.35 : 1),
          size: it.kind === 'points' ? 0.012 * (it.size || 2) + 0.01 : undefined });
      } else if (it.kind === 'marker') {
        out.push({ id, kind: 'points', pts: Float64Array.from(it.p), color: cssColor3(it.color), size: 0.02 * (it.size || 4) });
      } else if (it.kind === 'ellipses') {
        out.push({ id, kind: 'ellipses', centers: it.centers, mats: it.mats, r: it.r, color: 0xffd666, alpha: 0.85 });
      }
    }
    viz3.setItems(out);
    viz3.render();
  }
  const cssColor3 = (c) => { const [r, g, b] = hexToRgb(c); return new THREE.Color(r, g, b); };

  function autoFit(items) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity, r = 0;
    for (const it of items) {
      if ((it.kind !== 'line' && it.kind !== 'points') || it.noFit) continue;  // guides (boundary, target ghost) don't count
      const vd = S.dim;
      for (let i = 0; i < it.pts.length; i += vd) {
        const x = it.pts[i], y = it.pts[i + 1];
        if (!Number.isFinite(x) || !Number.isFinite(y) || Math.abs(x) > 1e4 || Math.abs(y) > 1e4) continue;
        x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
        if (vd === 3) r = Math.max(r, Math.hypot(x, y, it.pts[i + 2]));
      }
    }
    if (!Number.isFinite(x0)) return;
    if (S.dim === 3) { if (viz3) viz3.orbit.radius += 0.1 * (Math.max(1.5, r * 2.5) - viz3.orbit.radius); return; }
    const old = { cx: viz2.cx, cy: viz2.cy, scale: viz2.scale };
    viz2.fit([x0, y0, x1, y1]);
    const k = 0.15;
    viz2.cx = old.cx + k * (viz2.cx - old.cx); viz2.cy = old.cy + k * (viz2.cy - old.cy);
    viz2.scale = old.scale * Math.pow(viz2.scale / old.scale, k);
  }

  function fitView() {
    if (!traces) computeTraces();
    const { items } = buildItems(t);
    const save = S.display.autoFit;
    if (S.dim === 2) {
      let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
      for (const it of items) {
        if ((it.kind !== 'line' && it.kind !== 'points') || it.noFit) continue;
        for (let i = 0; i < it.pts.length; i += 2) {
          const x = it.pts[i], y = it.pts[i + 1];
          if (!Number.isFinite(x) || Math.abs(x) > 1e4 || Math.abs(y) > 1e4) continue;
          x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
        }
      }
      if (Number.isFinite(x0)) viz2.fit([x0, y0, x1, y1]);
    } else if (viz3) {
      let r = 0;
      for (const it of items) if (it.pts) for (let i = 0; i < it.pts.length; i += 3) { const v = Math.hypot(it.pts[i], it.pts[i + 1], it.pts[i + 2]); if (v < 1e4) r = Math.max(r, v); }
      viz3.fitRadius(r || 1.5);
    }
    S.display.autoFit = save;
  }

  function renderStageUI() {
    const n = traces.nStages, sl = $('stage');
    sl.max = String(n - 1);
    if (document.activeElement !== sl) sl.value = String(t);
    const s = Math.round(t), near = Math.abs(t - s) < 0.02;
    if (near) {
      const nm = stageName(s);
      $('stageLabel').innerHTML = `<b>${nm.title}</b><br><span class="tag">${nm.sub}</span>`;
    } else {
      const a = stageName(Math.floor(t)), b = stageName(Math.ceil(t));
      $('stageLabel').innerHTML = `<b>${a.title} → ${b.title}</b><br><span class="tag">morphing ${(100 * (t - Math.floor(t))).toFixed(0)}%</span>`;
    }
    $('ticks').textContent = `stage ${t.toFixed(2)} / ${n - 1}`;
    $('play').textContent = playing ? '⏸' : '▶';
    const P = traces.bases[Math.min(n - 1, Math.max(0, s))];
    if (P && !P.exact) $('stageLabel').innerHTML += `<br><span class="tag">PCA view: ${(100 * P.explained).toFixed(0)}% of variance</span>`;
  }

  // ===========================================================================
  // Stage animation
  // ===========================================================================

  const ease = (x) => x * x * (3 - 2 * x);
  let anim = null;  // { from, to, start, dur }

  function animateTo(goal, dur = 650) {
    if (!traces) computeTraces();
    goal = Math.max(0, Math.min(traces.nStages - 1, goal));
    anim = { from: t, to: goal, start: performance.now(), dur: dur * Math.max(0.4, Math.abs(goal - t)) };
  }

  function advanceAnimation(now) {
    if (anim) {
      const u = Math.min(1, (now - anim.start) / anim.dur);
      t = anim.from + (anim.to - anim.from) * ease(u);
      if (u >= 1) {
        t = anim.to; anim = null;
        if (playing) {
          if (t >= traces.nStages - 1 - 1e-9) playing = false;
          else animateTo(Math.floor(t + 1e-9) + 1);
        }
      }
    }
  }

  // ===========================================================================
  // Mouse interaction
  // ===========================================================================

  const TOOLS2 = [
    ['pan', 'Pan', 'Drag to move the view, wheel to zoom.'],
    ['curve', 'Curve', 'Draw a freehand curve (drawn in input space).'],
    ['circle', 'Circle', 'Press at the centre and drag out the radius.'],
    ['region', 'Region', 'Draw a closed outline; it is filled with points.'],
    ['pin', 'Pin', 'Press on an input point and drag to where its output should go (trains with target “pinned points”).'],
    ['probe', 'Probe', 'Click to choose the point analysed in “Input sensitivity”.'],
  ];
  const TOOLS3 = [
    ['pan', 'Orbit', 'Drag to rotate, right-drag (or Shift) to pan, wheel to zoom.'],
    ['sphereDraw', 'Draw on sphere', 'Drag over the sphere (centre/radius in Objects) to draw a curve on it.'],
    ['pin', 'Pin', 'Press on the z = 0 plane for the input, drag to its target on the same plane.'],
    ['probe', 'Probe', 'Click on the z = 0 plane to choose the analysed point.'],
  ];

  function renderToolbar() {
    const tb = $('toolbar');
    tb.innerHTML = '';
    for (const [id, label, tip] of S.dim === 2 ? TOOLS2 : TOOLS3) {
      const b = document.createElement('button');
      b.textContent = label; b.dataset.tip = tip;
      b.className = tool === id ? 'on' : '';
      b.onclick = () => setTool(id);
      tb.appendChild(b);
    }
    const tip = (S.dim === 2 ? TOOLS2 : TOOLS3).find((x) => x[0] === tool);
    $('hint').textContent = tip ? tip[2] : '';
    $('toolNote').textContent = S.dim === 2 ? 'Draw with the tools on top of the view.' : 'Add spheres here or draw on a sphere with the toolbar.';
  }

  function setTool(id) {
    tool = id;
    if (id === 'pin') {
      if (S.train.target !== 'pins') { S.train.target = 'pins'; $('target').value = 'pins'; onTargetChange(); }
      if (traces) animateTo(traces.nStages - 1);
    }
    if (id === 'curve' || id === 'circle' || id === 'region' || id === 'sphereDraw' || id === 'probe') animateTo(0);
    renderToolbar();
  }

  function addObject(o) {
    o.id = objId++;
    o.color = o.color || PALETTE[(o.id - 1) % PALETTE.length];
    S.objects.push(o);
    renderObjList();
    invalidate();
  }

  /** Resample a polyline to n points equally spaced along its length. */
  function resample(pts, n, closed) {
    const P = closed ? pts.concat([pts[0]]) : pts;
    const seg = [0];
    for (let i = 1; i < P.length; i++) seg.push(seg[i - 1] + Math.hypot(...P[i].map((v, k) => v - P[i - 1][k])));
    const L = seg[seg.length - 1];
    if (L < 1e-9) return [];
    const out = [], m = closed ? n : n - 1;
    let j = 0;
    for (let i = 0; i < n; i++) {
      const s = (L * i) / m;
      while (j < seg.length - 2 && seg[j + 1] < s) j++;
      const f = (s - seg[j]) / Math.max(1e-12, seg[j + 1] - seg[j]);
      out.push(P[j].map((v, k) => v + f * (P[j + 1][k] - v)));
    }
    return out;
  }

  function pointInPolygon(x, y, poly) {
    let inside = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
    }
    return inside;
  }

  function finishDrawing() {
    const d = drawing, n = S.nPoints;
    drawing = null;
    if (!d) return;
    if (d.type === 'curve' && d.pts.length > 2) {
      const pts = resample(d.pts, n, false);
      if (pts.length) addObject({ type: 'curve', closed: false, points: pts });
    } else if (d.type === 'circle' && d.r > 1e-3) {
      addObject({ type: 'circle', closed: true, points: linspace(0, 2 * Math.PI, n + 1).slice(0, n).map((a) => [d.c[0] + d.r * Math.cos(a), d.c[1] + d.r * Math.sin(a)]) });
    } else if (d.type === 'region' && d.pts.length > 4) {
      const nb = Math.round(n * 0.5), boundary = resample(d.pts, nb, true);
      const xs = d.pts.map((p) => p[0]), ys = d.pts.map((p) => p[1]);
      const x0 = Math.min(...xs), x1 = Math.max(...xs), y0 = Math.min(...ys), y1 = Math.max(...ys);
      // choose a lattice spacing that gives about n − nb interior points
      let area = 0;
      for (let i = 0, j = boundary.length - 1; i < boundary.length; j = i++) area += (boundary[j][0] + boundary[i][0]) * (boundary[j][1] - boundary[i][1]) / 2;
      const h = Math.sqrt(Math.abs(area) / Math.max(1, n - nb)) || 0.05, interior = [];
      for (let y = y0 + h / 2; y < y1; y += h) for (let x = x0 + h / 2; x < x1; x += h) if (pointInPolygon(x, y, d.pts)) interior.push([x, y]);
      if (boundary.length) addObject({ type: 'region', closed: true, points: boundary, interior });
    } else if (d.type === 'sphereDraw' && d.pts.length > 2) {
      const pts = resample(d.pts, n, false).map((p) => {     // re-project onto the sphere surface
        const c = S.sphere.c, v = p.map((x, k) => x - c[k]), r = Math.hypot(...v) || 1;
        return v.map((x, k) => c[k] + (S.sphere.r * x) / r);
      });
      addObject({ type: 'sphere path', closed: false, points: pts });
    } else if (d.type === 'pin') {
      S.pins.push({ x: d.x, y: d.y });
      onPinsChanged();
    }
  }

  function drawingPreview() {
    const d = drawing, out = [];
    if (d.type === 'curve' || d.type === 'region' || d.type === 'sphereDraw') {
      if (d.pts.length > 1) out.push({ id: 'preview', kind: 'line', pts: Float64Array.from(d.pts.flat()), closed: d.type === 'region', color: '#ffffff', width: 2 });
    } else if (d.type === 'circle') {
      const pts = linspace(0, 2 * Math.PI, 65).flatMap((a) => [d.c[0] + d.r * Math.cos(a), d.c[1] + d.r * Math.sin(a)]);
      out.push({ id: 'preview', kind: 'line', pts: Float64Array.from(pts), color: '#ffffff', width: 2 });
    } else if (d.type === 'pin') {
      out.push({ id: 'preview', kind: S.dim === 2 ? 'arrow' : 'segments', from: d.x, to: d.y, pts: Float64Array.from([...d.x, ...d.y]), color: '#ffd666', width: 2 });
    }
    return out;
  }

  function bind2DMouse() {
    const cv = $('c2d');
    let pan = null;
    cv.addEventListener('contextmenu', (e) => e.preventDefault());
    cv.addEventListener('pointerdown', (e) => {
      const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top, w = viz2.toWorld(px, py);
      cv.setPointerCapture(e.pointerId);
      if (e.button !== 0 || tool === 'pan') { pan = { x: e.clientX, y: e.clientY }; return; }
      if (tool === 'curve' || tool === 'region') { t = 0; anim = null; drawing = { type: tool, pts: [w] }; }
      else if (tool === 'circle') { t = 0; anim = null; drawing = { type: 'circle', c: w, r: 0 }; }
      else if (tool === 'pin') drawing = { type: 'pin', x: w, y: w.slice() };
      else if (tool === 'probe') { S.probe = w; invalidate(); }
    });
    cv.addEventListener('pointermove', (e) => {
      const r = cv.getBoundingClientRect(), w = viz2.toWorld(e.clientX - r.left, e.clientY - r.top);
      if (pan) { viz2.panBy(e.clientX - pan.x, e.clientY - pan.y); pan.x = e.clientX; pan.y = e.clientY; userMovedView(); return; }
      if (!drawing) return;
      if (drawing.type === 'circle') drawing.r = Math.hypot(w[0] - drawing.c[0], w[1] - drawing.c[1]);
      else if (drawing.type === 'pin') drawing.y = w;
      else {
        const last = drawing.pts[drawing.pts.length - 1];
        if (Math.hypot(w[0] - last[0], w[1] - last[1]) * viz2.scale > 2) drawing.pts.push(w);
      }
    });
    const up = () => { pan = null; finishDrawing(); };
    cv.addEventListener('pointerup', up);
    cv.addEventListener('pointercancel', () => { pan = null; drawing = null; });
    cv.addEventListener('wheel', (e) => {
      e.preventDefault();
      const r = cv.getBoundingClientRect();
      viz2.zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-wheelPixels(e) * 0.0015));
      userMovedView();
    }, { passive: false });
  }

  /** Manual zoom / pan wins over auto-fit (otherwise auto-fit would undo it on the next frame). */
  function userMovedView() {
    if (!S.display.autoFit) return;
    S.display.autoFit = false;
    $('autoFit').checked = false;
    flashHint('Auto-fit turned off so your zoom stays. Turn it back on under Display → Auto-fit view.');
  }

  let hintTimer = null;
  function flashHint(text) {
    $('hint').textContent = text;
    $('hint').style.color = '#e3b341';
    clearTimeout(hintTimer);
    hintTimer = setTimeout(() => { $('hint').style.color = ''; renderToolbar(); }, 4500);
  }

  function bind3DMouse() {
    const el = viz3.renderer.domElement;
    viz3.allowRotate = (e) => tool === 'pan' || e.button === 2;
    viz3.onUserZoom = userMovedView;
    const pos = (e) => { const r = el.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    el.addEventListener('pointerdown', (e) => {
      if (tool === 'pan' || e.button !== 0) return;
      const [px, py] = pos(e);
      el.setPointerCapture(e.pointerId);
      if (tool === 'sphereDraw') {
        t = 0; anim = null;
        const p = viz3.pickSphere(px, py, S.sphere.c, S.sphere.r);
        drawing = { type: 'sphereDraw', pts: p ? [p] : [] };
      } else if (tool === 'pin') {
        const p = viz3.pickPlaneZ0(px, py);
        if (p) drawing = { type: 'pin', x: p, y: p.slice() };
      } else if (tool === 'probe') {
        const p = viz3.pickPlaneZ0(px, py);
        if (p) { S.probe = p; invalidate(); }
      }
    });
    el.addEventListener('pointermove', (e) => {
      if (!drawing) return;
      const [px, py] = pos(e);
      if (drawing.type === 'sphereDraw') { const p = viz3.pickSphere(px, py, S.sphere.c, S.sphere.r); if (p) drawing.pts.push(p); }
      else if (drawing.type === 'pin') { const p = viz3.pickPlaneZ0(px, py); if (p) drawing.y = p; }
    });
    el.addEventListener('pointerup', finishDrawing);
  }

  // ===========================================================================
  // Training loop
  // ===========================================================================

  function trainSteps(k) {
    if (!trainer.tasks.length) { training = false; updateTrainStatus('Nothing to train on: choose a target (or add pins).'); return; }
    for (let i = 0; i < k; i++) {
      const r = trainer.step();
      if (r === null) { training = false; break; }
      if (++stepsSinceRedraw >= S.train.redrawEvery) { stepsSinceRedraw = 0; invalidate(); netVersion++; }
    }
    if (!training) { invalidate(); netVersion++; }
    if (trainer.lossHistory.length > 20000) trainer.lossHistory.splice(0, trainer.lossHistory.length - 20000);
    updateTrainStatus();
  }

  function updateTrainStatus(msg) {
    const el = $('trainStatus'), T = trainer;
    $('trainPlay').textContent = training ? '⏸ Pause' : '▶ Train';
    if (msg) { el.textContent = msg; return; }
    if (!T) return;
    const parts = [`step ${T.step_}`];
    if (T.lossHistory.length) parts.push(`loss ${fmt(T.lossHistory[T.lossHistory.length - 1])}`);
    const acc = accuracy();
    if (acc !== null && T.step_ > 0) parts.push(`accuracy ${(100 * acc).toFixed(1)}%`);
    if (S.train.mode === 'sequential' && T.tasks.length) parts.push(T.done ? 'all tasks done' : `task ${T.task + 1}/${T.tasks.length}`);
    if (T.diverged) parts.push('diverged — lower the learning rate and press Reset');
    if (!T.tasks.length) parts.push(S.train.target === 'pins' ? 'add pins with the Pin tool' : 'choose a target');
    el.textContent = parts.join(' · ');
  }

  /** One plain sentence saying what the chosen task asks the network to do. */
  function targetNote() {
    const z = S.dim === 3 ? ', 0' : '';
    switch (S.train.target) {
      case 'anchors': return `Every blue point should land on (1, 0${z}) and every red point on (−1, 0${z}). Loss: squared distance to its point. Watch each class shrink onto its point.`;
      case 'classify': return 'Two output numbers (logits): blue points should end up on one side of the line logit₁ = logit₂, red points on the other. Loss: cross-entropy.';
      case 'transform': return 'Every input x should go to A·x, where A is the chosen linear map (drawn dashed at the output).';
      case 'pins': return 'With the Pin tool, drag from an input point to where its output should go; the network bends space to satisfy every pin.';
      default: return 'No training: the network keeps its random weights. Move the stage slider to see what they do to space.';
    }
  }

  /** Warn (instead of silently changing settings) when the output activation cannot reach the targets. */
  function outputWarning() {
    if (!net || !['anchors', 'transform', 'pins'].includes(S.train.target)) return '';
    const act = net.acts[net.nLayers - 1];
    const range = { tanh: '−1…1', sigmoid: '0…1', sin: '−1…1', relu: '≥ 0', gelu: '≥ −0.17' }[act];
    return range ? ` ⚠ The output layer uses ${act} (range ${range}), so some targets are out of reach — tick “Linear output layer” or press ★ Recommended.` : '';
  }

  function syncTargetUI() {
    $('datasetRow').classList.toggle('hidden', !usesClasses());
    $('transformRow').classList.toggle('hidden', !isTransform());
    $('amountRow').classList.toggle('hidden', !isTransform());
    $('targetNote').textContent = targetNote() + outputWarning();
  }

  function onTargetChange() {
    syncTargetUI();
    rebuildTrainer();
    invalidate();
  }

  function onPinsChanged() {
    rebuildTrainer();
    renderPinList();
    invalidate();
  }

  /** Pins with their input → target coordinates, each removable. */
  function renderPinList() {
    const box = $('pinList');
    box.innerHTML = '';
    $('clearPins').disabled = !S.pins.length;
    const f = (v) => '(' + Array.from(v).map((x) => x.toFixed(2)).join(', ') + ')';
    S.pins.forEach((p, i) => {
      const row = document.createElement('div');
      row.className = 'obj';
      row.innerHTML = `<span class="dot" style="background:${PALETTE[i % PALETTE.length]}"></span><span>pin ${i + 1} · ${f(p.x)} → ${f(p.y)}</span>`;
      const del = document.createElement('button');
      del.textContent = '✕'; del.dataset.tip = 'Delete this pin';
      del.onclick = () => { training = false; S.pins.splice(i, 1); onPinsChanged(); };
      row.appendChild(del);
      box.appendChild(row);
    });
  }

  // ===========================================================================
  // Analysis panels
  // ===========================================================================

  function fmt(v) {
    if (v === null || v === undefined || !Number.isFinite(v)) return '—';
    const a = Math.abs(v);
    return a !== 0 && (a < 1e-3 || a >= 1e4) ? v.toExponential(2) : v.toFixed(a < 1 ? 4 : 3);
  }

  function updateAnalysis() {
    drawLoss();
    // forgetting
    const F = trainer.forgetting(), names = trainer.tasks.map((_, i) => (S.train.target === 'pins' ? `pin ${i + 1}` : `task ${i + 1}`));
    if (!trainer.tasks.length) $('forgetting').innerHTML = '<div class="note">No tasks yet.</div>';
    else if (S.train.mode !== 'sequential') {
      $('forgetting').innerHTML = '<div class="note">Switch Training → Mode to <b>sequential</b> to measure forgetting.</div>'
        + table(['task', 'loss now'], trainer.tasks.map((_, i) => [names[i], fmt(trainer.taskLoss(i))]));
    } else {
      $('forgetting').innerHTML = table(['task', 'L after own', 'L now', 'F'], F.map((f, i) => f
        ? [names[i], fmt(f.after), fmt(f.now), { v: fmt(f.F), cls: f.F > 1e-3 ? 'bad' : 'good' }]
        : [names[i], '—', fmt(trainer.taskLoss(i)), '—']));
    }
    // interference / NTK
    const kind = S.analysis.matrix;
    let samples = [];
    if (S.train.target === 'pins') samples = S.pins.map((p) => ({ x: Float64Array.from(p.x), y: Float64Array.from(p.y) }));
    else if (data.length) for (let i = 0; i < 16; i++) samples.push(data[Math.floor((i * data.length) / 16)]);
    if (kind === 'ntk' && !samples.length) samples = jacobianProbePoints().slice(0, 9).map((x) => ({ x }));
    if (kind === 'cosine' && !samples.length) {
      Charts.heatmap($('heatmap'), [], { empty: 'choose a target or add pins' });
      $('heatNote').textContent = '';
    } else {
      const M = kind === 'ntk' ? NN.ntkMatrix(net, samples.map((s) => s.x)) : NN.gradientCosine(net, samples, isClassify() ? 'ce' : 'mse');
      Charts.heatmap($('heatmap'), M, { mode: kind === 'ntk' ? 'sequential' : 'diverging' });
      $('heatNote').textContent = kind === 'ntk'
        ? `K(xᵢ, xⱼ) = tr(Jᵢ Jⱼᵀ) for ${samples.length} points; brighter = training one point moves the other's output more.`
        : `cos(∇Lᵢ, ∇Lⱼ) for ${samples.length} ${S.train.target === 'pins' ? 'pins' : 'samples'}: red = they agree, blue = they fight (interference).`;
    }
    // sensitivity at the probe point
    drawSensitivity();
    // invertibility
    const pts = [];
    for (const d of (traces ? traces.ds : [])) if (d.role !== 'jac') { const k = Math.max(1, Math.floor(d.pts.length / 60)); for (let i = 0; i < d.pts.length; i += k) pts.push(d.pts[i]); }
    const rep = NN.layerReport(net, pts.length ? pts : [new Float64Array(S.dim)]);
    $('layerTable').innerHTML = table(['layer', 'map', 'act', 'rank', 'σ(W)'], rep.map((r) => [
      String(r.layer), `${r.nin}→${r.nout}`, r.act, { v: `${r.rank}/${Math.min(r.nin, r.nout)}`, cls: r.rank < Math.min(r.nin, r.nout) ? 'bad' : '' },
      r.sv.slice(0, 3).map((v) => v.toFixed(2)).join(' ') + (r.sv.length > 3 ? ' …' : '')]))
      + rep.filter((r) => r.warnings.length).map((r) => `<div class="note">Layer ${r.layer}:</div><ul class="warnlist">${r.warnings.map((w) => `<li>${w}</li>`).join('')}</ul>`).join('');
  }

  function table(head, rows) {
    const cell = (c) => (typeof c === 'object' ? `<td class="${c.cls}">${c.v}</td>` : `<td>${c}</td>`);
    return `<table><tr>${head.map((h) => `<th>${h}</th>`).join('')}</tr>${rows.map((r) => `<tr>${r.map(cell).join('')}</tr>`).join('')}</table>`;
  }

  function drawLoss() {
    const h = trainer.lossHistory;
    Charts.line($('lossChart'), [{ ys: h, color: '#58a6ff' }], { logY: true, xLabel: `${h.length} steps`, empty: 'train to see the loss curve' });
    $('lossNote').textContent = h.length ? `current ${fmt(h[h.length - 1])} · min ${fmt(Math.min(...h.slice(-5000)))}` : '';
  }

  /** Directional gain ‖f(p + εu) − f(p)‖/ε versus the linear prediction ‖J u‖, around the probe point. */
  function drawSensitivity() {
    const p = Float64Array.from(S.probe), eps = S.analysis.eps, d = S.dim;
    const J = net.inputJacobian(p), f0 = net.predict(p);
    const dirs = d === 2 ? linspace(0, 2 * Math.PI, 91).map((a) => [Math.cos(a), Math.sin(a)]) : NN.fibonacciSphere(90).map((v) => Array.from(v));
    const lin = [], fd = [];
    for (const u of dirs) {
      lin.push(NN.norm(NN.matVec(J, u)));
      const f1 = net.predict(p.map((v, k) => v + eps * u[k]));
      fd.push(NN.norm(f1.map((v, k) => v - f0[k])) / eps);
    }
    Charts.line($('sensChart'), [{ ys: lin, color: '#8b949e', dash: [4, 4] }, { ys: fd, color: '#ffd666' }],
      { xLabel: d === 2 ? 'direction angle 0 → 2π' : 'direction (Fibonacci order)' });
    const sv = NN.singularValues(J);
    const lip = NN.lipschitz(net, new NN.Rng(12345), 2, 800);
    const detJ = d === 2 ? J[0][0] * J[1][1] - J[0][1] * J[1][0] : null;
    $('sensInfo').innerHTML = [
      ['probe x', '(' + Array.from(p).map((v) => v.toFixed(2)).join(', ') + ')'],
      ['‖J‖₂ (max stretch)', fmt(sv[0])], ['σ_min (min stretch)', fmt(sv[sv.length - 1])],
      ...(detJ !== null ? [['det J', fmt(detJ)]] : []),
      ['Lipschitz (sampled, ≥)', fmt(lip.empirical)], ['Lipschitz bound Π‖W‖·max|σ′|', fmt(lip.bound)],
      ['legend', '<span style="color:#ffd666">— ‖Δf‖/ε</span> · <span style="color:#8b949e">- - ‖J u‖</span>'],
    ].map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join('');
  }

  // ===========================================================================
  // Controls
  // ===========================================================================

  function bindRange(id, get, set, show = (v) => v, onInput = () => {}) {
    const el = $(id), val = $(id + 'Val');
    const sync = () => { el.value = get(); if (val) val.textContent = show(+el.value); };
    el.addEventListener('input', () => { set(+el.value); if (val) val.textContent = show(+el.value); onInput(); });
    sync();
    return sync;
  }

  const syncers = [];
  function setupControls() {
    $('defaultAct').innerHTML = ACTS.map((a) => `<option value="${a}">${a}</option>`).join('');
    syncers.push(bindRange('layers', () => S.net.layers, (v) => { S.net.layers = v; }, (v) => v, () => rebuildNet()));
    syncers.push(bindRange('width', () => S.net.width, (v) => { S.net.width = v; }, (v) => v, () => rebuildNet()));
    syncers.push(bindRange('temperature', () => S.net.temperature, (v) => { S.net.temperature = v; }, (v) => v.toFixed(2), () => rebuildNet(true)));
    syncers.push(bindRange('initScale', () => S.net.init.scale, (v) => { S.net.init.scale = v; }, (v) => v.toFixed(2), () => rebuildNet()));
    syncers.push(bindRange('nPoints', () => S.nPoints, (v) => { S.nPoints = v; }));
    syncers.push(bindRange('amount', () => S.train.amount, (v) => { S.train.amount = v; }, (v) => v.toFixed(2), () => { rebuildTrainer(); invalidate(); }));
    syncers.push(bindRange('lr', () => Math.log10(S.train.lr), (v) => { S.train.lr = Math.pow(10, v); trainer.setOpts({ lr: S.train.lr }); }, (v) => Math.pow(10, v).toPrecision(2)));
    syncers.push(bindRange('batch', () => S.train.batch, (v) => { S.train.batch = v; trainer.setOpts({ batch: v }); }));
    syncers.push(bindRange('stepsPerFrame', () => S.train.stepsPerFrame, (v) => { S.train.stepsPerFrame = v; }));
    syncers.push(bindRange('redrawEvery', () => S.train.redrawEvery, (v) => { S.train.redrawEvery = v; }));
    syncers.push(bindRange('stepsPerTask', () => S.train.stepsPerTask, (v) => { S.train.stepsPerTask = v; trainer.setOpts({ stepsPerTask: v }); }));
    syncers.push(bindRange('nTasks', () => S.train.nTasks, (v) => { S.train.nTasks = v; }, (v) => v, () => rebuildTrainer()));
    syncers.push(bindRange('lambda', () => Math.log10(S.train.lambda), (v) => { S.train.lambda = Math.pow(10, v); trainer.setOpts({ ewcLambda: S.train.lambda }); }, (v) => Math.pow(10, v).toPrecision(2)));
    syncers.push(bindRange('eps', () => Math.log10(S.analysis.eps), (v) => { S.analysis.eps = Math.pow(10, v); analysisDirty = true; }, (v) => Math.pow(10, v).toPrecision(2)));

    $('defaultAct').onchange = (e) => { S.net.defaultAct = e.target.value; rebuildNet(true); };
    $('outputLinear').onchange = (e) => { S.net.outputLinear = e.target.checked; rebuildNet(true); };
    $('initDist').onchange = (e) => { S.net.init.dist = e.target.value; rebuildNet(); };
    $('seed').onchange = (e) => { S.net.init.seed = Math.max(0, Math.round(+e.target.value || 0)); rebuildNet(); };
    $('resample').onclick = () => { S.net.init.seed++; $('seed').value = S.net.init.seed; rebuildNet(); };

    for (const k of ['grid', 'circle', 'basis', 'origin', 'jac', 'det', 'dist', 'autoFit']) {
      const el = $('show' + k[0].toUpperCase() + k.slice(1)) || $(k);
      el.onchange = () => { S.display[k] = el.checked; invalidate(); };
    }
    $('fitBtn').onclick = fitView;
    $('resetView').onclick = () => { viz2.cx = viz2.cy = 0; viz2.scale = Math.min(viz2.w, viz2.h) / 6; if (viz3) { viz3.orbit = { theta: 0.8, phi: 1.1, radius: 7 }; viz3.target.set(0, 0, 0); } };

    $('target').onchange = (e) => { S.train.target = e.target.value; training = false; onTargetChange(); };
    $('dataset').onchange = (e) => { S.train.dataset = e.target.value; training = false; rebuildTrainer(); invalidate(); };
    $('transform').onchange = (e) => { S.train.transform = e.target.value; training = false; rebuildTrainer(); invalidate(); };
    $('optimizer').onchange = (e) => { S.train.optimizer = e.target.value; trainer.setOpts({ optimizer: e.target.value }); };
    $('method').onchange = (e) => { S.train.method = e.target.value; trainer.setOpts({ method: e.target.value }); syncModeUI(); };
    $('modeJoint').onclick = () => { S.train.mode = 'joint'; syncModeUI(); rebuildTrainer(); };
    $('modeSeq').onclick = () => { S.train.mode = 'sequential'; syncModeUI(); rebuildTrainer(); };
    $('trainPlay').onclick = () => {
      if (!training && trainer.done) rebuildTrainerKeepWeights();
      training = !training; updateTrainStatus();
    };
    $('trainStep').onclick = () => { training = false; trainSteps(1); invalidate(); netVersion++; };
    $('trainReset').onclick = () => { training = false; rebuildNet(); };
    $('matrixKind').onchange = (e) => { S.analysis.matrix = e.target.value; analysisDirty = true; };

    $('play').onclick = () => {
      if (!traces) computeTraces();
      if (playing) { playing = false; anim = null; return; }
      playing = true;
      if (t >= traces.nStages - 1 - 1e-9) t = 0;
      animateTo(Math.floor(t + 1e-9) + 1);
    };
    $('stepFwd').onclick = () => { playing = false; animateTo(Math.floor(t + 1e-6) + 1); };
    $('stepBack').onclick = () => { playing = false; animateTo(Math.ceil(t - 1e-6) - 1); };
    $('stage').addEventListener('input', (e) => { playing = false; anim = null; t = +e.target.value; });

    $('dim2').onclick = () => setDim(2);
    $('dim3').onclick = () => setDim(3);
    $('addSphere').onclick = () => {
      readSphereInputs();
      addObject({ type: 'sphere', closed: false, points: NN.fibonacciSphere(S.nPoints, S.sphere.c, S.sphere.r).map((p) => Array.from(p)) });
    };
    for (const id of ['sx', 'sy', 'sz', 'sr']) $(id).onchange = readSphereInputs;
    $('clearObjects').onclick = () => { S.objects = []; renderObjList(); invalidate(); };
    $('clearPins').onclick = () => { training = false; S.pins = []; onPinsChanged(); };
    $('examples').onclick = addExamples;
    $('exportBtn').onclick = exportState;
    $('recommendBtn').onclick = applyRecommended;
    $('importBtn').onclick = () => $('importFile').click();
    $('importFile').onchange = (e) => { const f = e.target.files[0]; if (f) f.text().then(importState).catch((err) => alert('Import failed: ' + err.message)); e.target.value = ''; };
  }

  /**
   * ★ Recommended: one click aligns everything to settings that work well for the current task and
   * data — architecture, activations, initialisation, optimiser and the view.  Nothing else in the app
   * changes settings on its own; this button is the only "auto" behaviour.
   */
  function applyRecommended() {
    const d = S.dim, T = S.train;
    let arch;
    if (T.target === 'anchors' || T.target === 'classify') {
      arch = { moons: [4, d], circles: [4, d === 3 ? 6 : 4], spirals: [6, 6] }[T.dataset];
    } else if (T.target === 'transform') arch = [2, 4];
    else if (T.target === 'pins') arch = [3, 6];
    else arch = [3, d];
    const training = T.target !== 'none';
    Object.assign(S.net, { layers: arch[0], width: arch[1], defaultAct: 'tanh', temperature: 0, overrides: [],
      outputLinear: training, init: { ...S.net.init, dist: training ? 'xavier' : 'he', scale: training ? 1 : 1.6 } });
    Object.assign(T, { optimizer: 'adam', lr: 0.01, batch: 32, stepsPerFrame: 10, redrawEvery: 10 });
    training_stop();
    rebuildNet();
    syncControls();
    fitView();
    flashHint(`★ Recommended: ${arch[0]} layers, width ${arch[1]}, tanh${training ? ', linear output, Xavier init' : ', He init ×1.6'}, Adam lr 0.01 — view fitted.`);
  }
  function training_stop() { training = false; updateTrainStatus(); }

  /** After a finished sequential run, "Train" starts a new run on the current weights. */
  function rebuildTrainerKeepWeights() {
    trainer = new NN.Trainer(net, trainerOpts());
    trainer.setTasks(buildTasks());
  }

  function readSphereInputs() {
    S.sphere = { c: [+$('sx').value || 0, +$('sy').value || 0, +$('sz').value || 0], r: Math.max(0.05, +$('sr').value || 0.6) };
  }

  function syncModeUI() {
    $('modeJoint').className = S.train.mode === 'joint' ? 'on' : '';
    $('modeSeq').className = S.train.mode === 'sequential' ? 'on' : '';
    document.querySelectorAll('.seqOnly').forEach((el) => el.classList.toggle('hidden', S.train.mode !== 'sequential'));
    $('lambdaRow').classList.toggle('hidden', S.train.mode !== 'sequential' || S.train.method !== 'ewc');
  }

  function syncControls() {
    for (const s of syncers) s();
    $('defaultAct').value = S.net.defaultAct;
    $('outputLinear').checked = S.net.outputLinear;
    $('initDist').value = S.net.init.dist;
    $('seed').value = S.net.init.seed;
    for (const k of ['grid', 'circle', 'basis', 'origin', 'jac', 'det', 'dist', 'autoFit']) {
      const el = $('show' + k[0].toUpperCase() + k.slice(1)) || $(k);
      el.checked = !!S.display[k];
    }
    $('target').value = S.train.target;
    $('dataset').value = S.train.dataset;
    $('transform').value = S.train.transform;
    $('optimizer').value = S.train.optimizer;
    $('method').value = S.train.method;
    $('matrixKind').value = S.analysis.matrix;
    $('dim2').className = S.dim === 2 ? 'on' : '';
    $('dim3').className = S.dim === 3 ? 'on' : '';
    $('sphereControls').classList.toggle('hidden', S.dim !== 3);
    $('showCircle').parentElement.lastChild.textContent = S.dim === 2 ? ' Unit circle' : ' Unit sphere';
    [$('sx').value, $('sy').value, $('sz').value] = S.sphere.c;
    $('sr').value = S.sphere.r;
    syncTargetUI();
    syncModeUI();
    renderObjList();
    renderPinList();
    renderToolbar();
  }

  /** Per-layer activation pickers: "auto" follows the default / temperature rule. */
  function renderLayerActs() {
    const box = $('layerActs'), acts = net.acts;
    box.innerHTML = '';
    for (let l = 0; l < S.net.layers; l++) {
      const ov = S.net.overrides[l] || 'auto';
      const row = document.createElement('div');
      row.className = 'row';
      row.innerHTML = `<label data-tip="Activation of layer ${l + 1}; “auto” follows the default activation and temperature.">Layer ${l + 1}</label>`;
      const sel = document.createElement('select');
      sel.innerHTML = `<option value="auto">auto (${ov === 'auto' ? acts[l] : '…'})</option>` + ACTS.map((a) => `<option value="${a}">${a}</option>`).join('');
      sel.value = ov;
      sel.onchange = () => { S.net.overrides[l] = sel.value; rebuildNet(true); };
      row.appendChild(sel);
      const tag = document.createElement('span');
      tag.className = 'tag';
      tag.textContent = `${net.dims[l]}→${net.dims[l + 1]}`;
      row.appendChild(tag);
      box.appendChild(row);
    }
  }

  function renderObjList() {
    const box = $('objList');
    box.innerHTML = S.objects.length ? '' : '<div class="note">No objects yet.</div>';
    for (const o of S.objects) {
      const row = document.createElement('div');
      row.className = 'obj';
      row.innerHTML = `<span class="dot" style="background:${o.color}"></span><span>${o.type} · ${o.points.length + (o.interior ? o.interior.length : 0)} pts</span>`;
      const del = document.createElement('button');
      del.textContent = '✕'; del.dataset.tip = 'Delete this object';
      del.onclick = () => { S.objects = S.objects.filter((x) => x !== o); renderObjList(); invalidate(); };
      row.appendChild(del);
      box.appendChild(row);
    }
  }

  function addExamples() {
    const n = S.nPoints;
    if (S.dim === 2) {
      addObject({ type: 'circle', closed: true, points: linspace(0, 2 * Math.PI, n + 1).slice(0, n).map((a) => [0.75 + 0.35 * Math.cos(a), 0.55 + 0.35 * Math.sin(a)]) });
      addObject({ type: 'curve', closed: false, points: linspace(-1.7, -0.1, n).map((x) => [x, -0.75 + 0.3 * Math.sin(5 * x)]) });
    } else {
      addObject({ type: 'sphere', closed: false, points: NN.fibonacciSphere(n, [0.6, 0.3, 0.2], 0.45).map((p) => Array.from(p)) });
    }
  }

  function setDim(d) {
    if (d === S.dim) return;
    const keep = { net: S.net, train: S.train, display: S.display, nPoints: S.nPoints, analysis: S.analysis };
    S = defaultState(d);
    Object.assign(S, keep);
    S.net.width = S.net.width === 5 - d ? d : S.net.width;   // follow the "width = dimension" default
    S.objects = []; S.pins = [];
    if (S.train.target === 'pins') S.train.target = 'none';
    tool = 'pan'; t = 0; anim = null; playing = false; training = false;
    $('c2d').classList.toggle('hidden', d !== 2);
    $('c3d').classList.toggle('hidden', d !== 3);
    if (d === 3 && !viz3) {
      try { viz3 = new Viz3D($('c3d')); bind3DMouse(); }
      catch (err) { alert('3D view needs WebGL: ' + err.message); return setDim(2); }
    }
    if (viz3) viz3.resize();
    rebuildNet();
    addExamples();
    syncControls();
  }

  // ===========================================================================
  // Export / import
  // ===========================================================================

  /** The full state as a plain object: settings, objects, pins and the current weights. */
  function serializeState() {
    const round = (p) => p.map((v) => +v.toFixed(5));
    const out = JSON.parse(JSON.stringify(S));
    out.objects = S.objects.map((o) => ({ ...o, points: o.points.map(round), interior: o.interior ? o.interior.map(round) : undefined }));
    out.weights = Array.from(net.theta);
    out.dims = net.dims;
    return out;
  }

  function exportState() {
    const blob = new Blob([JSON.stringify(serializeState())], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `mlp-space-${S.dim}d-seed${S.net.init.seed}.json`;
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  }

  function importState(text) {
    const j = JSON.parse(text);
    if (!j || (j.dim !== 2 && j.dim !== 3) || !j.net) throw new Error('not a setup file');
    const base = defaultState(j.dim);
    const merged = { ...base, ...j, net: { ...base.net, ...j.net, init: { ...base.net.init, ...(j.net.init || {}) } },
      display: { ...base.display, ...(j.display || {}) }, train: { ...base.train, ...(j.train || {}) },
      analysis: { ...base.analysis, ...(j.analysis || {}) }, sphere: { ...base.sphere, ...(j.sphere || {}) } };
    delete merged.weights; delete merged.dims;
    const t0 = merged.train.target;                       // older files: one menu for task + dataset
    if (['moons', 'circles', 'spirals'].includes(t0)) Object.assign(merged.train, { target: 'classify', dataset: t0 });
    if (['rotation', 'shear', 'scaling'].includes(t0)) Object.assign(merged.train, { target: 'transform', transform: t0 });
    const prevDim = S.dim;
    S = merged;
    objId = 1 + Math.max(0, ...S.objects.map((o) => o.id || 0));
    if (S.dim !== prevDim) {
      $('c2d').classList.toggle('hidden', S.dim !== 2);
      $('c3d').classList.toggle('hidden', S.dim !== 3);
      if (S.dim === 3 && !viz3) { viz3 = new Viz3D($('c3d')); bind3DMouse(); }
    }
    net = null;
    rebuildNet(false, j.weights ? Float64Array.from(j.weights) : null);
    tool = 'pan'; t = 0; training = false;
    syncControls();
    fitView();
  }

  // ===========================================================================
  // Tooltips
  // ===========================================================================

  function setupTooltips() {
    const tip = $('tip');
    document.addEventListener('mouseover', (e) => {
      const el = e.target.closest('[data-tip]');
      if (!el) { tip.classList.add('hidden'); return; }
      tip.textContent = el.dataset.tip;
      tip.classList.remove('hidden');
    });
    document.addEventListener('mousemove', (e) => {
      if (tip.classList.contains('hidden')) return;
      const x = Math.min(window.innerWidth - tip.offsetWidth - 8, e.clientX + 14);
      const y = e.clientY + 18 + tip.offsetHeight > window.innerHeight ? e.clientY - tip.offsetHeight - 10 : e.clientY + 18;
      tip.style.left = x + 'px'; tip.style.top = y + 'px';
    });
  }

  // ===========================================================================
  // Main loop
  // ===========================================================================

  function frame(now) {
    advanceAnimation(now);
    if (training) trainSteps(S.train.stepsPerFrame);
    try { render(); } catch (err) { console.error(err); }
    if (now - lastLossDraw > 120 && training) { drawLoss(); lastLossDraw = now; }
    if (analysisDirty && now - lastAnalysis > (training ? 700 : 150)) {
      analysisDirty = false; lastAnalysis = now;
      try { updateAnalysis(); } catch (err) { console.error(err); }
    }
    requestAnimationFrame(frame);
  }

  function onResize() {
    viz2.resize();
    if (viz3) viz3.resize();
    analysisDirty = true;
  }

  function init() {
    setupControls();
    setupTooltips();
    bind2DMouse();
    rebuildNet();
    addExamples();
    syncControls();
    viz2.resize();
    viz2.scale = Math.min(viz2.w, viz2.h) / 6;
    window.addEventListener('resize', onResize);
    if (window.ResizeObserver) new ResizeObserver(onResize).observe($('view'));
    requestAnimationFrame(frame);
    window.__app = { get S() { return S; }, get net() { return net; }, get trainer() { return trainer; }, setDim, importState,
      get t() { return t; }, set t(v) { t = v; }, setTool, finishDrawing, get traces() { return traces; },
      serializeState, get training() { return training; }, set training(v) { training = v; }, viz2 };
  }

  init();
})();
