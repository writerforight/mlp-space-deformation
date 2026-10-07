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
  // shape pairs for the "turn one shape into another" task, per dimension
  const MORPHS = {
    2: [['circle', 'square'], ['circle', 'star'], ['circle', 'figure8'], ['twoCircles', 'oneCircleTwice']],
    3: [['sphere', 'ellipsoid'], ['sphere', 'cube'], ['unknot', 'trefoil'], ['torus', 'torusOnSphere']],
  };

  // ===========================================================================
  // State
  // ===========================================================================

  function defaultState(dim = 2) {
    return {
      version: STATE_VERSION,
      dim,
      net: { layers: 3, width: dim, defaultAct: 'tanh', temperature: 0, outputLinear: false, homeo: false,
        overrides: [], init: { dist: 'he', scale: 1.6, seed: 2 } },
      objects: [],
      nPoints: 300,
      display: { grid: true, circle: true, basis: true, origin: true, jac: false, det: false, dist: false, autoFit: false, bg: true, lift: true, basisInfo: true, mats: false },
      train: { target: 'none', goals: [], dataset: 'moons', nData: 400, noise: 0.08, dataSeed: 0, transform: 'rotation', morph: MORPHS[dim][0].join('>'), amount: 0.9, optimizer: 'adam', lr: 0.01, batch: 32, stepsPerFrame: 5,
        redrawEvery: 10, mode: 'joint', stepsPerTask: 300, nTasks: 3, method: 'none', lambda: 100 },
      pins: [],
      custom: [],          // painted points for the "paint your own" dataset: [{ x, label }]
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
    if ($('netNote')) renderNetNote();
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
  const isMorph = () => S.train.target === 'morph';
  const isGoals = () => S.train.target === 'goals';
  const usesClasses = () => isClassify() || isAnchors() || isGoals();
  /** "My goals": a goal sends some points somewhere; several goals train together (see buildData). */
  const goalSplit = () => isGoals() && S.train.goals.some((g) => g.kind === 'classSplit');
  const classTarget = (c) => { const g = isGoals() && S.train.goals.find((x) => x.kind === 'classPoint' && x.cls === c); return g ? g.target : null; };
  const HOMEO_FLOOR = 0.2;                  // smallest singular value kept in homeomorphism mode
  const morphPair = () => S.train.morph.split('>');

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
    for (const s of (isGoals() ? goalClassData : data)) {
      const o = net.predict(s.x);
      const v = bgValue(o);
      if (v === null) return null;
      if ((v > 0 ? 0 : 1) === s.label) ok++;
    }
    return ok / Math.max(1, (isGoals() ? goalClassData : data).length);
  }

  /** Largest distance between an output and its target over a set of samples (vector targets only). */
  function worstOn(set) {
    if (!set || !set.length || isClassify() || !set[0].y || typeof set[0].y === 'number') return null;
    let w = 0;
    for (const s of set) { const o = net.predict(s.x); let d = 0; for (let k = 0; k < o.length; k++) d += (o[k] - s.y[k]) ** 2; w = Math.max(w, d); }
    return Math.sqrt(w);
  }

  /*
   * Shape tasks are also checked on a 50× finer copy of the shape: the network only sees 400 training
   * points and can pass a curve *between* two of them (a tear the training points never see).
   */
  let denseCache = null, worstCache = { at: 0, version: -1, value: null };
  function denseData() {
    const key = `${S.dim}|${S.train.target}|${S.train.dataset}|${S.train.morph}`;
    if (denseCache && denseCache.key === key) return denseCache.data;
    let d = null;
    if (isMorph()) { const [a, b] = morphPair(); d = NN.morphData(a, b, 20000); }
    else if (isAnchors() && S.train.dataset.startsWith('shape:')) {
      d = NN.makeShape(S.train.dataset.slice(6), 20000).components.flatMap((c) => c.pts.map((x) => ({ x, y: anchor(c.label) })));
    }
    denseCache = { key, data: d };
    return d;
  }
  function worstDense() {
    const now = performance.now();
    if (worstCache.version === netVersion + trainer.step_ || now - worstCache.at < 400) return worstCache.value;
    worstCache = { at: now, version: netVersion + trainer.step_, value: worstOn(denseData()) };
    return worstCache.value;
  }

  /**
   * Signed "which class wins here" for an output o: > 0 blue, < 0 red, null if the task has no such rule.
   * Logits (classify, or a "split" goal): o₁ − o₂.  Two class points A (blue) and B (red): which one is
   * nearer, scaled so ±1 means "at that point".  Anchors: the first coordinate.
   */
  function bgValue(o) {
    if (isClassify() || goalSplit()) return (o[0] - o[1]) / 4;
    if (isAnchors()) return o[0];
    const A = classTarget(0), B = classTarget(1);
    if (isGoals() && A && B) {
      let num = 0, den = 0;
      for (let k = 0; k < A.length; k++) { const d = A[k] - B[k]; num += (o[k] - (A[k] + B[k]) / 2) * d; den += d * d; }
      return den ? (2 * num) / den : 0;
    }
    return null;
  }

  let goalClassData = [];         // the class points shown in "my goals" mode (each point once)
  let goalNextId = 1;
  const GOAL_KINDS = {
    classPoint: { name: 'Class → point', formula: (g) => `L = mean ‖f(x) − A‖² over the ${g.cls === 0 ? 'blue' : 'red'} points` },
    classSplit: { name: 'Separate the classes', formula: () => 'L = mean cross-entropy(softmax f(x), class): blue wins logit₁, red logit₂' },
    objectPoint: { name: 'Object → point', formula: () => 'L = mean ‖f(x) − A‖² over the object' },
    objectStay: { name: 'Object stays', formula: () => 'L = mean ‖f(x) − x‖² over the object: keep it where it is' },
    pins: { name: 'Pinned points', formula: () => 'L = mean ‖f(xᵢ) − yᵢ‖² over the pins (add them with the Pin tool, ＋ Objects)' },
  };

  function defaultGoals() {
    const z = S.dim === 3 ? [0] : [];
    return [{ id: goalNextId++, kind: 'classPoint', cls: 0, target: [1, 0, ...z], weight: 1 },
      { id: goalNextId++, kind: 'classPoint', cls: 1, target: [-1, 0, ...z], weight: 1 }];
  }

  /** Every goal becomes samples { x, y, type, w, goal }; w = weight · N / (K · nᵢ), so the batch loss is (1/K) Σ wᵢ Lᵢ. */
  function goalSamples(raw) {
    const groups = [];
    for (const g of S.train.goals) {
      let smp = [];
      const obj = g.obj !== undefined && S.objects.find((o) => o.id === g.obj);
      if (g.kind === 'classPoint') smp = raw.filter((r) => r.y === g.cls).map((r) => ({ x: r.x, label: r.y, y: Float64Array.from(g.target), type: 'mse' }));
      else if (g.kind === 'classSplit') smp = raw.map((r) => ({ x: r.x, label: r.y, y: r.y, type: 'ce' }));
      else if (obj && g.kind === 'objectPoint') smp = obj.points.map((p) => ({ x: Float64Array.from(p), y: Float64Array.from(g.target), type: 'mse' }));
      else if (obj && g.kind === 'objectStay') smp = obj.points.map((p) => ({ x: Float64Array.from(p), y: Float64Array.from(p), type: 'mse' }));
      else if (g.kind === 'pins') smp = S.pins.map((p) => ({ x: Float64Array.from(p.x), y: Float64Array.from(p.y), type: 'mse' }));
      smp.forEach((q) => { q.goal = g.id; });
      groups.push({ g, smp });
    }
    const live = groups.filter((x) => x.smp.length), N = live.reduce((a, x) => a + x.smp.length, 0);
    for (const { g, smp } of live) for (const q of smp) q.w = (g.weight * N) / (live.length * smp.length);
    return live.flatMap((x) => x.smp);
  }

  /** A goal's weight or target changed: update its samples in place, so training keeps its momentum. */
  function refreshGoalSamples(g) {
    const N = data.length, K = new Set(data.map((q) => q.goal)).size, mine = data.filter((q) => q.goal === g.id);
    for (const q of mine) {
      q.w = (g.weight * N) / (K * mine.length);
      if (g.kind === 'classPoint' || g.kind === 'objectPoint') q.y = Float64Array.from(g.target);
    }
    analysisDirty = true;
  }

  function buildData() {
    // dataSeed 0 keeps the data tied to the weight seed (as before); the guided start can draw another sample
    const rng = new NN.Rng(S.net.init.seed * 101 + 5 + 7919 * (S.train.dataSeed || 0));
    targetA = null;
    if (isGoals()) {
      const ds = S.train.dataset;
      const raw = ds === 'custom' ? S.custom.map((p) => ({ x: Float64Array.from(p.x), y: p.label }))
        : ds.startsWith('shape:') ? NN.makeShape(ds.slice(6)).components.flatMap((c) => c.pts.map((x) => ({ x, y: c.label })))
        : NN.makeDataset(ds, S.train.nData || 400, S.dim, rng, S.train.noise ?? 0.08);
      goalClassData = raw.map((r) => ({ x: r.x, label: r.y }));
      data = goalSamples(raw);
    } else if (usesClasses()) {
      const ds = S.train.dataset;
      const raw = ds === 'custom' ? S.custom.map((p) => ({ x: Float64Array.from(p.x), y: p.label }))
        : ds.startsWith('shape:') ? NN.makeShape(ds.slice(6)).components.flatMap((c) => c.pts.map((x) => ({ x, y: c.label })))
        : NN.makeDataset(ds, S.train.nData || 400, S.dim, rng, S.train.noise ?? 0.08);
      data = raw.map((s) => ({ x: s.x, label: s.y, y: isAnchors() ? anchor(s.y) : s.y }));
    } else if (isTransform()) {
      targetA = NN.targetMatrix(S.train.transform, S.dim, S.train.amount);
      data = NN.makeTransformData(targetA, 300, S.dim, rng);
    } else if (isMorph()) {
      const [from, to] = morphPair();
      data = NN.morphData(from, to);
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
      stepsPerTask: T.stepsPerTask, method: T.method, ewcLambda: T.lambda, seed: S.net.init.seed,
      invertibleFloor: S.net.homeo ? HOMEO_FLOOR : 0 };
  }

  function rebuildTrainer() {
    buildData();
    trainer = new NN.Trainer(net, trainerOpts());
    trainer.setTasks(buildTasks());
    resetHistory();
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
      if (o.hidden) continue;
      // Fibonacci spheres are one long spiral: draw it lighter so the surface stays readable
      out.push({ id: `o${o.id}`, role: 'object', kind: 'line', closed: o.closed, color: o.color, width: 2,
        alpha: o.type === 'sphere' || o.surface ? 0.6 : 1, pts: o.points.map((p) => Float64Array.from(p)) });
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
      if (o.hidden) continue;
      const k = Math.max(1, Math.floor(o.points.length / 12));
      for (let i = 0; i < o.points.length; i += k) pts.push(Float64Array.from(o.points[i]));
    }
    return pts;
  }

  /** The points compared by the interference / NTK matrix (row/column i ↔ point i + 1 on screen). */
  function interferenceSamples() {
    if (S.train.target === 'pins') return S.pins.map((p) => ({ x: Float64Array.from(p.x), y: Float64Array.from(p.y) }));
    if (data.length) return [...Array(16)].map((_, i) => data[Math.floor((i * data.length) / 16)]);
    if (S.analysis.matrix === 'ntk') return jacobianProbePoints().slice(0, 9).map((x) => ({ x }));
    return [];
  }

  function allDrawables() {
    const ds = referenceDrawables().concat(userDrawables());
    if (usesClasses() && S.train.dataset.startsWith('shape:')) {     // exact shapes: dense curves show every tear
      NN.makeShape(S.train.dataset.slice(6), 4000).components.forEach((c, k) => ds.push({ id: `shape${k}`, role: 'data',
        kind: c.closed ? 'line' : 'points', closed: c.closed, color: CLASS_COLORS[c.label], width: 2, size: 1.4, pts: c.pts }));
    } else if (usesClasses() && (data.length || goalClassData.length)) {
      const shown = isGoals() ? goalClassData : data;
      if (shown.length) ds.push({ id: 'data', role: 'data', kind: 'points', size: 2.4, colors: shown.map((s) => CLASS_COLORS[s.label]), pts: shown.map((s) => s.x) });
    }
    if (isMorph()) {
      NN.makeShape(morphPair()[0], 3000).components.forEach((c, k) => ds.push({ id: `msrc${k}`, role: 'object', kind: c.filled || !c.closed ? 'points' : 'line',
        closed: c.closed, color: c.label === undefined ? PALETTE[0] : CLASS_COLORS[c.label], width: 2, size: 1.8, pts: c.pts }));
    }
    if (S.pins.length) ds.push({ id: 'pins', role: 'pins', kind: 'points', pts: S.pins.map((p) => Float64Array.from(p.x)) });
    ds.push({ id: 'probe', role: 'probe', kind: 'points', pts: [Float64Array.from(S.probe)] });
    if (S.display.jac) ds.push({ id: 'jac', role: 'jac', kind: 'none', pts: jacobianProbePoints() });
    if (S.train.target !== 'pins') {         // pins are already labelled "pin 1, pin 2, …"
      const smp = interferenceSamples();
      if (smp.length) ds.push({ id: 'interf', role: 'interf', kind: 'none', pts: smp.map((p) => p.x) });
    }
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
    depthInfo = [];
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
      d.proj = []; d.pj = []; d.depth = [];
    }
    const bases = [];
    for (let s = 0; s < nStages; s++) {
      // collect a sample of all points at this stage for PCA
      const cloud = [];
      for (const d of ds) {
        if (d.role === 'probe' || d.role === 'jac' || d.role === 'interf') continue;
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
      if (!P.exact && s > 0) {
        // Turn the view plane to match the previous stage as well as possible (orthogonal Procrustes): with
        // Y = this stage's view coordinates and Y' the previous one's, Yᵀ Y' = U S Vᵀ and the best rotation
        // (or reflection) is R = U Vᵀ.  The plane stays the same — only its axes turn — so when two principal
        // directions swap or flip between stages, the animation does not jump.
        const M = [...Array(vd)].map(() => new Float64Array(vd));
        for (const d of ds) {
          const a = d.proj[s], b = d.proj[s - 1];
          for (let i = 0; i < a.length; i += vd) for (let r = 0; r < vd; r++) for (let c = 0; c < vd; c++) M[r][c] += a[i + r] * b[i + c];
        }
        const { U, V } = NN.squareSvd(M);
        const R = [...Array(vd)].map((_, i) => [...Array(vd)].map((__, j) => U.reduce((acc, u, k) => acc + u[i] * V[k][j], 0)));
        const B = P.basis;
        P.basis = [...Array(vd)].map((_, a) => {
          const out = new Float64Array(B[0].length);
          for (let c = 0; c < vd; c++) for (let j = 0; j < out.length; j++) out[j] += R[c][a] * B[c][j];
          return out;
        });
        project();
      }
      bases.push(P);
      // 2D mode, layer wider than 2: the third principal direction becomes depth (seen when the view is tilted)
      const P3 = vd === 2 && S.display.lift && cloud[0].length > 2 ? NN.pcaBasis(cloud, 3) : null;
      depthInfo[s] = P3 ? { explained: P3.explained, dim: cloud[0].length } : null;
      for (const d of ds) {
        const n = d.st.length, z = new Float64Array(n);
        if (P3) {
          const b = P3.basis[2];
          for (let i = 0; i < n; i++) { const x = d.st[i][s]; let v = 0; for (let j = 0; j < x.length; j++) v += b[j] * x[j]; z[i] = v; }
        }
        d.depth[s] = z;
      }
      if (P3 && s > 0) {                  // keep the depth axis pointing the same way as in the previous stage
        let c = 0;
        for (const d of ds) for (let i = 0; i < d.depth[s].length; i++) c += d.depth[s][i] * d.depth[s - 1][i];
        if (c < 0) for (const d of ds) d.depth[s] = d.depth[s].map((v) => -v);
      }
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

  // ---- 2D tilt camera: turn the plane to look at the third principal direction of wide layers ----------
  const cam = { yaw: 0, pitch: 0 };
  let depthInfo = [];             // per stage: { explained, dim } when a depth axis exists
  const tilted = () => S.dim === 2 && (Math.abs(cam.yaw) > 1e-4 || Math.abs(cam.pitch) > 1e-4);
  /** (x, y, depth) -> screen plane: turn about the vertical axis (yaw), then tip forward (pitch). */
  function camXY(x, y, z) {
    const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
    const x1 = cy * x + sy * z, z1 = -sy * x + cy * z;
    return [x1, cp * y - sp * z1];
  }

  function positionsAt(d, tt) {
    const [s, f] = stageSplit(tt), s2 = Math.min(s + 1, traces.nStages - 1), a = d.proj[s], b = d.proj[s2];
    const out = new Float64Array(a.length);
    for (let i = 0; i < a.length; i++) out[i] = a[i] + f * (b[i] - a[i]);
    if (tilted() && d.depth && d.depth[s]) {
      const za = d.depth[s], zb = d.depth[s2];
      for (let i = 0; i < za.length; i++) {
        const q = camXY(out[2 * i], out[2 * i + 1], za[i] + f * (zb[i] - za[i]));
        out[2 * i] = q[0]; out[2 * i + 1] = q[1];
      }
    }
    return out;
  }

  /** Small axes in the corner while the view is tilted: which way the two view directions and depth point. */
  function renderGizmo(depth, pca) {
    const g = $('gizmo');
    g.classList.toggle('hidden', !tilted());
    if (!tilted()) return;
    const c = 42, k = 28, names = pca ? ['PC1', 'PC2', depth ? 'PC3' : 'depth'] : ['x', 'y', 'depth'];
    const cols = ['#ff6b6b', '#51cf66', '#4dabf7'];
    const ax = [[1, 0, 0], [0, 1, 0], [0, 0, 1]].map((v) => camXY(v[0], v[1], v[2]));
    g.innerHTML = ax.map((q, i) => {
      const x = c + k * q[0], y = c - k * q[1];
      return `<line x1="${c}" y1="${c}" x2="${x.toFixed(1)}" y2="${y.toFixed(1)}" stroke="${cols[i]}" stroke-width="2" stroke-linecap="round"${i === 2 && !depth ? ' stroke-dasharray="3 3"' : ''}/>
        <text x="${(c + (k + 9) * q[0]).toFixed(1)}" y="${(c - (k + 9) * q[1] + 3).toFixed(1)}" fill="${cols[i]}" font-size="9.5" text-anchor="middle">${names[i]}</text>`;
    }).join('') + `<circle cx="${c}" cy="${c}" r="2" fill="#e6edf3"/>`;
  }

  /** Animate the camera back to looking straight down. */
  function flatView() {
    const y0 = cam.yaw, p0 = cam.pitch, t0 = performance.now();
    (function step(now) {
      const u = Math.min(1, (now - t0) / 400), e = 1 - (1 - u) ** 3;
      cam.yaw = y0 * (1 - e); cam.pitch = p0 * (1 - e);
      if (u < 1) requestAnimationFrame(step); else { cam.yaw = 0; cam.pitch = 0; }
    })(t0);
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
      out.push(v > 1e-6 ? '#3fb950' : v < -1e-6 ? '#bc8cff' : '#8b949e');
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
    // class background: how the rest of the network would classify a point sitting here, at every stage
    const showBg = vd === 2 && usesClasses() && D.bg && !tilted() && bgValue(new Float64Array(vd)) !== null;
    if (showBg) items.push(backgroundImage(Math.round(tt)));
    // say what the colours are: a reading rule applied to the network's output, not part of the network
    const legend = $('legend');
    legend.classList.toggle('hidden', !showBg);
    if (showBg) legend.textContent = isGoals() && !goalSplit()
      ? 'Background colour: which class point the rest of the network sends each spot nearer to — brighter blue toward the blue point A, brighter red toward the red point B, dark = halfway.'
      : isAnchors()
      ? 'Background colour = f(x)·(1, 0): how far the network moves each input toward the blue point (1, 0) (brighter blue) or the red point (−1, 0) (brighter red). Dark = halfway, i.e. the network is undecided there. Full colour at ±1.'
      : 'Background colour = logit₁ − logit₂: brighter blue where the first logit wins by more, brighter red where the second does; dark = undecided. Full colour at ±4.';
    const bgStage = Math.round(tt), bgBasis = showBg && traces.bases[bgStage];
    if (bgBasis && !bgBasis.exact) {
      const ag = bgCache && bgCache.agree !== null ? ` Here it agrees with the network on ${(100 * bgCache.agree).toFixed(0)}% of the data points${
        bgCache.agree < 0.95 ? ': the rest are separated along directions the slice does not show (tilt the view)' : ''}.` : '';
      legend.textContent = `At this stage (ℝ${bgBasis.basis[0].length}) the colour is how the rest of the network classifies the PCA plane `
        + `through the points — a 2D slice.${ag} ` + legend.textContent;
    } else if (showBg && bgStage > 0) legend.textContent = 'How the rest of the network classifies each spot of this stage. ' + legend.textContent;
    for (const d of traces.ds) {
      if (d.role === 'jac' || d.role === 'probe' || d.role === 'pins' || d.role === 'interf') continue;
      const p = posCache.get(d);
      let colors = d.colors;
      if (D.det && d.hasJ && d.role !== 'data') colors = detColors(d, tt);
      else if (D.dist && d.role !== 'data') {
        colors = d.pts.map((x, i) => {
          let s = 0; for (let a = 0; a < vd; a++) s += (p[i * vd + a] - x[a]) ** 2;
          return Colors.seqColor(distMax > 0 ? Math.sqrt(s) / distMax : 0);
        });
      }
      items.push({ id: d.id, kind: d.kind, pts: p, closed: d.closed, color: d.color, colors, width: d.width, alpha: d.alpha, size: d.size, fromDrawable: true });
      if (d.role === 'basis' && vd === 2 && p.length >= 4) {
        const n = p.length / 2;
        items.push({ kind: 'arrow', from: [p[2 * n - 4], p[2 * n - 3]], to: [p[2 * n - 2], p[2 * n - 1]], color: d.color, width: 2.6, rotated: true });
        items.push({ kind: 'label', p: [p[2 * n - 2], p[2 * n - 1]], text: d.label, color: d.color, rotated: true });
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
    if (atOutput && isMorph()) {
      NN.makeShape(morphPair()[1], 3000).components.forEach((c, k) => items.push({ id: `mtgt${k}`, kind: c.filled || !c.closed ? 'points' : 'line',
        pts: Float64Array.from(c.pts.flatMap((p) => Array.from(p))), closed: c.closed, color: '#e6edf3', alpha: 0.4, dash: [5, 5], width: 1.5, size: 1.2, noFit: true }));
    }
    // the decision boundary (equal logits) lives on the diagonal of the output plane
    // the selected object: its box and resize handle, while it can be dragged (input step, flat 2D, Pan)
    if (atInput && atInputFlat()) {
      const o = S.objects.find((x) => x.id === selObj && !x.hidden);
      if (o) {
        const [x0, y0, x1, y1] = objBox(o), m = 0.04;
        items.push({ kind: 'line', closed: true, pts: Float64Array.from([x0 - m, y0 - m, x1 + m, y0 - m, x1 + m, y1 + m, x0 - m, y1 + m]),
          color: o.color, alpha: 0.7, dash: [4, 4], width: 1, noFit: true });
        items.push({ kind: 'marker', p: [x1 + m, y0 - m], color: o.color, size: 5 });
      }
    }
    // my goals: target points (drag them at the output), and the shape an "object stays" goal keeps
    if (atOutput && isGoals()) {
      for (const g of S.train.goals) {
        const obj = g.obj !== undefined && S.objects.find((o) => o.id === g.obj);
        if ((g.kind === 'classPoint' || g.kind === 'objectPoint') && g.target) {
          const col = g.kind === 'classPoint' ? CLASS_COLORS[g.cls] : (obj ? obj.color : '#e6edf3');
          items.push({ id: `goal${g.id}`, kind: 'marker', p: g.target.slice(0, vd), color: col, size: 9, ring: true });
          if (vd === 2) items.push({ kind: 'label', p: g.target, text: `${g.kind === 'classPoint' ? (g.cls === 0 ? 'blue' : 'red') : (obj ? obj.type : 'object')} → A${dragGoal && dragGoal.id === g.id ? '' : ' (drag)'}`, color: col });
        } else if (g.kind === 'objectStay' && obj) {
          items.push({ id: `goalStay${g.id}`, kind: 'line', pts: Float64Array.from(obj.points.flatMap((p) => p.slice(0, vd))), closed: obj.closed,
            color: obj.color, alpha: 0.45, dash: [5, 5], width: 1.5, noFit: true });
        }
      }
    }
    if (atOutput && (isClassify() || goalSplit()) && vd === 2) {
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
        items.push({ id: `pinErr${i}`, kind: vd === 2 ? 'arrow' : 'segments', from: c, to: pin.y, pts: Float64Array.from([...c, ...pin.y]), color: col, dash: [4, 4], width: 1.2, alpha: 0.7, fromRotated: true });
        items.push({ id: `pinIn${i}`, kind: 'marker', p: pin.x, color: 'rgba(230,237,243,0.5)', size: 3, ring: true });
        items.push({ id: `pinT${i}`, kind: 'marker', p: pin.y, color: col, size: 6, ring: true });
        items.push({ id: `pinC${i}`, kind: 'marker', p: c, color: col, size: 4, rotated: true });
        if (vd === 2) items.push({ kind: 'label', p: pin.y, text: `pin ${i + 1}`, color: col });
      });
    }
    // numbered rings: the points compared by the interference matrix
    const id_ = traces.ds.find((d) => d.role === 'interf');
    if (id_) {
      const p = posCache.get(id_);
      for (let i = 0; i < id_.pts.length; i++) {
        const c = Array.from(p.subarray(i * vd, (i + 1) * vd));
        items.push({ id: `interf${i}`, kind: 'marker', p: c, color: 'rgba(230,237,243,0.9)', size: 5, ring: true, rotated: true });
        if (vd === 2) items.push({ kind: 'label', p: c, text: String(i + 1), color: '#e6edf3', rotated: true });
      }
    }
    // probe point used by the sensitivity panel
    const pr = traces.ds.find((d) => d.role === 'probe');
    if (pr) items.push({ id: 'probe', kind: 'marker', p: Array.from(posCache.get(pr)), color: '#ffd666', size: 5, ring: true, rotated: true });
    // shape currently being drawn
    if (drawing) items.push(...drawingPreview());
    if (tilted()) {                      // everything not taken from the traced objects sits in the output plane (depth 0)
      const rot = (p) => camXY(p[0], p[1], 0);
      for (let i = items.length - 1; i >= 0; i--) {
        const it = items[i];
        if (it.kind === 'ellipses' || it.kind === 'image') items.splice(i, 1);
        else if ((it.kind === 'line' || it.kind === 'points') && !it.fromDrawable) {
          const q = new Float64Array(it.pts.length);
          for (let k = 0; k < q.length; k += 2) { const r = rot([it.pts[k], it.pts[k + 1]]); q[k] = r[0]; q[k + 1] = r[1]; }
          it.pts = q;
        } else if ((it.kind === 'marker' || it.kind === 'label') && !it.rotated) it.p = rot(it.p);
        else if (it.kind === 'arrow' && !it.rotated) { if (!it.fromRotated) it.from = rot(it.from); it.to = rot(it.to); }
      }
    }
    return { items, atInput, atOutput };
  }

  /**
   * Background over the visible input region (2D, class tasks).  The colour is a plain measured number,
   * no thresholding: v = f(x)·(1, 0) — how far the output moved toward the blue point (1, 0) versus the
   * red point (−1, 0) — or, for logits, v = (logit₁ − logit₂)/4.  v = +1 → full blue, −1 → full red,
   * 0 → dark (halfway); values beyond ±1 stay fully coloured.
   */
  /**
   * Network from stage s to the output.  s = 0: the input; odd s = after layer l's linear step (apply σ_l,
   * then layers l+1..L); even s = after layer l's activation.  A 2D view point p stands for a point of the
   * stage space: exact when the stage is 2D, on the PCA plane through the points when it is wider.
   */
  function restOfNetwork(s, p) {
    // the view point p = B·x; the closest point of the PCA plane through the cloud's mean m is m + Bᵀ(p − B·m)
    const P = traces.bases[s], B = P.basis, m = P.mean, dim = B[0].length;
    const bm = [0, 1].map((k) => { let v = 0; for (let j = 0; j < dim; j++) v += B[k][j] * m[j]; return v; });
    let a = new Float64Array(dim);
    for (let j = 0; j < dim; j++) a[j] = m[j] + B[0][j] * (p[0] - bm[0]) + B[1][j] * (p[1] - bm[1]);
    let l = Math.ceil(s / 2);                                // layer whose output (or linear part) this is
    if (s % 2 === 1) { const f = NN.ACTIVATIONS[net.acts[l - 1]].f; a = a.map(f); }
    for (; l < net.nLayers; l++) {
      const L = net.layout[l], th = net.theta, f = NN.ACTIVATIONS[net.acts[l]].f, out = new Float64Array(L.nout);
      for (let i = 0; i < L.nout; i++) {
        let v = th[L.b + i];
        for (let j = 0; j < L.nin; j++) v += th[L.w + i * L.nin + j] * a[j];
        out[i] = f(v);
      }
      a = out;
    }
    return a;
  }

  function backgroundImage(stage = 0) {
    const box = viz2.viewBox(), key = box.map((v) => v.toFixed(3)).join() + netVersion + trainer.step_ + '|' + stage;
    if (bgCache && bgCache.key === key) return bgCache.item;
    const W = 90, H = Math.max(10, Math.round((W * (box[3] - box[1])) / (box[2] - box[0])));
    const rgba = new Uint8ClampedArray(W * H * 4), blue = [88, 166, 255], red = [255, 107, 107];
    for (let j = 0; j < H; j++) {
      for (let i = 0; i < W; i++) {
        const x = box[0] + ((i + 0.5) * (box[2] - box[0])) / W, y = box[3] - ((j + 0.5) * (box[3] - box[1])) / H;
        const o = stage === 0 ? net.predict([x, y]) : restOfNetwork(stage, [x, y]);
        const raw = bgValue(o);
        const v = Math.max(-1, Math.min(1, Number.isFinite(raw) ? raw : 0));
        const col = v >= 0 ? blue : red, k = 4 * (j * W + i);
        for (let c = 0; c < 3; c++) rgba[k + c] = col[c];
        rgba[k + 3] = Math.round(150 * Math.abs(v));      // brightness ∝ |v|: dark near the halfway line
      }
    }
    // in a wide stage the picture is a 2D slice: say how often it agrees with the network on the data points
    let agree = null;
    const dd = traces.ds.find((x) => x.role === 'data');
    if (stage > 0 && !traces.bases[stage].exact && dd && dd.st.length) {
      let ok = 0;
      for (let i = 0; i < dd.st.length; i++) {
        const o = restOfNetwork(stage, [dd.proj[stage][2 * i], dd.proj[stage][2 * i + 1]]), y = net.predict(dd.pts[i]);
        const side = (v) => bgValue(v) > 0;
        if (side(o) === side(y)) ok++;
      }
      agree = ok / dd.st.length;
    }
    bgCache = { key, agree, item: { kind: 'image', bbox: box, w: W, h: H, rgba } };
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

  // ---- top left: what the current layer does to the origin and the basis vectors ---------------------------
  const SUBD = '₀₁₂₃₄₅₆₇₈₉';
  const subN = (k) => String(k).split('').map((c) => SUBD[+c]).join('');
  function fmtVec(v) {
    const a = Array.from(v), f = (x) => (Math.abs(x) < 0.005 ? '0.00' : x.toFixed(2)).replace('-', '−');
    return a.length <= 4 ? `(${a.map(f).join(', ')})` : `(${a.slice(0, 3).map(f).join(', ')}, … +${a.length - 3})`;   // hover shows all
  }
  let basisKey = '';
  function renderBasisBox() {
    const box = $('basisBox');
    if (!S.display.basisInfo || !net) { box.classList.add('hidden'); return; }
    const s = Math.round(t), key = `${s}|${netVersion}|${trainer ? trainer.step_ : 0}|${S.dim}`;
    box.classList.remove('hidden');
    if (key === basisKey) return;
    basisKey = key;
    const d = S.dim, pts = [new Float64Array(d)];
    for (let i = 0; i < d; i++) { const e = new Float64Array(d); e[i] = 1; pts.push(e); }
    const names = ['o', ...[...Array(d)].map((_, i) => `e${subN(i + 1)}`)], colors = ['#e6edf3', ...BASIS_COLORS];
    const st = pts.map((p) => net.stages(p));
    let head, cols, now;
    if (s === 0) {
      head = ['input x'];
      cols = [0]; now = 0;
    } else {
      const l = Math.ceil(s / 2), act = net.acts[l - 1];
      const sup = (k) => String(k).split('').map((c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c]).join('');
      head = [l === 1 ? 'input x' : `a${subN(l - 1)} ∈ ℝ${sup(net.dims[l - 1])}`, `z = W${subN(l)}${l === 1 ? 'x' : `a${subN(l - 1)}`} + b${subN(l)}`, `${act === 'identity' ? 'linear' : act}(z)${l === net.nLayers ? ' = output' : ''}`];
      cols = [2 * l - 2, 2 * l - 1, 2 * l]; now = s % 2 === 1 ? 1 : 2;
    }
    const title = s === 0 ? 'Origin and basis vectors before the network' : `Layer ${Math.ceil(s / 2)}: where o and the basis vectors go (real coordinates)`;
    box.innerHTML = `<div class="bx-title">${title}</div><table><tr><th></th>${head.map((h, j) => `<th class="${j === now ? 'now' : ''}">${h}</th>`).join('')}</tr>${
      st.map((row, i) => `<tr><td class="k" style="color:${colors[i]}">${names[i]}</td>${cols.map((c, j) => `<td class="${j === now ? 'now' : ''}" title="${Array.from(row[c]).map((x) => x.toFixed(4)).join(', ')}">${fmtVec(row[c])}</td>`).join('')}</tr>`).join('')}</table>`;
  }

  // ---- model drawer: the weight matrices as numbers ----------------------------------------------------------
  const matOpen = new Set([1]);
  let matKey = '', matAt = 0;
  function renderMats(force) {
    const box = $('matBox');
    box.classList.toggle('hidden', !S.display.mats);
    if (!S.display.mats || !net || !isOpen('matbox')) return;
    const key = `${netVersion}|${trainer ? trainer.step_ : 0}|${net.nParams}`;
    if (!force && (key === matKey || performance.now() - matAt < 300)) return;
    matKey = key; matAt = performance.now();
    let mx = 1e-9;
    for (const v of net.theta) mx = Math.max(mx, Math.abs(v));
    const cell = (v) => {
      const a = (0.12 + 0.6 * Math.min(1, Math.abs(v) / mx)).toFixed(3);
      const bg = v >= 0 ? `rgba(227,179,65,${a})` : `rgba(188,140,255,${a})`;
      return `<span style="background:${bg}">${v.toFixed(2).replace('-', '−')}</span>`;
    };
    box.innerHTML = net.layout.map((L, l) => {
      const act = net.acts[l], th = net.theta;
      let g = '<span class="h"></span>' + [...Array(L.nin)].map((_, j) => `<span class="h">a${subN(j + 1)}</span>`).join('') + '<span class="gap"></span><span class="h">b</span>';
      for (let i = 0; i < L.nout; i++) {
        g += `<span class="h">z${subN(i + 1)}</span>`;
        for (let j = 0; j < L.nin; j++) g += cell(th[L.w + i * L.nin + j]);
        g += '<span class="gap"></span>' + cell(th[L.b + i]);
      }
      return `<details data-l="${l + 1}"${matOpen.has(l + 1) ? ' open' : ''}><summary><b>W${subN(l + 1)}</b> ${L.nout}×${L.nin} · ${act === 'identity' ? 'linear' : act}<span class="tag">${L.nout * (L.nin + 1)} numbers</span></summary>
        <div class="mgrid" style="grid-template-columns: auto repeat(${L.nin}, auto) 6px auto">${g}</div></details>`;
    }).join('');
    box.querySelectorAll('details').forEach((dt) => dt.addEventListener('toggle', () => {
      const l = +dt.dataset.l;
      if (dt.open) matOpen.add(l); else matOpen.delete(l);
    }));
  }

  function render() {
    if (!traces) computeTraces();
    t = Math.min(traces.nStages - 1, Math.max(0, t));
    const { items } = buildItems(t);
    if (S.display.autoFit) autoFit(items);
    if (S.dim === 2) viz2.render(items);
    else render3D(items);
    renderStageUI();
    renderBasisBox();
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
    if (P && !P.exact) {
      const D3 = depthInfo[Math.min(n - 1, Math.max(0, s))];
      const pct = (v) => (v >= 0.9995 ? '100' : v > 0.99 ? (Math.floor(1000 * v) / 10).toFixed(1) : (100 * v).toFixed(0));
      $('stageLabel').innerHTML += `<br><span class="tag">ℝ${P.basis[0].length} shown through PCA: ${pct(P.explained)}% of the spread in 2 directions${
        D3 ? `, ${pct(D3.explained)}% with depth — right-drag to tilt` : ''}</span>`;
    }
    if (tilted()) $('stageLabel').innerHTML += '<br><span class="tag">tilted view · double-click or ⟲ Flat to look straight down</span>';
    renderGizmo(depthInfo[Math.min(n - 1, Math.max(0, s))], P && !P.exact);
  }

  // ===========================================================================
  // Stage animation
  // ===========================================================================

  const ease = (x) => x * x * (3 - 2 * x);
  let anim = null;  // { from, to, start, dur }

  /** Jump to a (fractional) stage, e.g. while dragging along the model strip. */
  function scrubTo(v) {
    if (!traces) computeTraces();
    anim = null; playing = false;
    t = Math.max(0, Math.min(traces.nStages - 1, v));
  }

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
    ['pan', 'Pan', 'Drag to move the view, wheel to zoom, right-drag to tilt (wide layers).'],
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

  const PAINT_TOOLS = [
    ['paint0', 'Paint blue', 'Drag to spray blue (class 1) training points.'],
    ['paint1', 'Paint red', 'Drag to spray red (class 2) training points.'],
  ];
  function toolList() {
    if (S.dim === 3) return TOOLS3;
    return usesClasses() && S.train.dataset === 'custom' ? TOOLS2.concat(PAINT_TOOLS) : TOOLS2;
  }

  function renderToolbar() {
    const tb = $('toolbar');
    tb.innerHTML = '';
    if (!toolList().some((x) => x[0] === tool)) tool = 'pan';
    for (const [id, label, tip] of toolList()) {
      const b = document.createElement('button');
      b.textContent = label; b.dataset.tip = tip;
      b.className = tool === id ? 'on' : '';
      b.disabled = id === 'pan' && tool === 'pan';          // already in the normal mode
      b.onclick = () => setTool(tool === id ? 'pan' : id);  // the active tool again = back to normal
      tb.appendChild(b);
    }
    const tip = toolList().find((x) => x[0] === tool);
    $('hint').textContent = tip ? tip[2] : '';
    const active = tool !== 'pan' && tip;
    $('toolChip').classList.toggle('hidden', !active);
    if (active) $('toolChip').innerHTML = `<b>${tip[1]}</b> tool <span class="tag">— click here, the tool again, or Esc to stop</span> ✕`;
    $('toolNote').textContent = S.dim === 2 ? 'Pick a tool, then draw on the view. Pan moves the view.' : 'Pick a tool and draw on a sphere, or add a sphere under Drawing settings.';
  }

  function setTool(id) {
    tool = id;
    if (id === 'pan') drawing = null;
    if (id === 'pin') {
      if (isGoals()) { if (!S.train.goals.some((g) => g.kind === 'pins')) addGoal('pins'); }   // pins join the other goals
      else if (S.train.target !== 'pins') { S.train.target = 'pins'; $('target').value = 'pins'; onTargetChange(); }
      if (traces) animateTo(traces.nStages - 1);
    }
    if (['curve', 'circle', 'region', 'sphereDraw', 'probe', 'paint0', 'paint1'].includes(id)) animateTo(0);
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
    } else if (d.type === 'paint') {
      rebuildTrainer();
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
    let pan = null, orbit = null;
    cv.addEventListener('contextmenu', (e) => e.preventDefault());
    cv.addEventListener('pointerdown', (e) => {
      const r = cv.getBoundingClientRect(), px = e.clientX - r.left, py = e.clientY - r.top, w = viz2.toWorld(px, py);
      try { cv.setPointerCapture(e.pointerId); } catch (err) { /* synthetic events have no capturable pointer */ }
      if (e.button === 2 && S.display.lift) { orbit = { x: e.clientX, y: e.clientY }; return; }
      if (e.button === 0) { const g = goalAt(px, py); if (g) { dragGoal = g; return; } }
      if (e.button === 0) {
        const h = objectHit(px, py);
        if (h) {
          if (selObj !== h.o.id) { selObj = h.o.id; renderObjList(); }
          ensureBase(h.o);
          const c = h.o.base.c.map((v, k) => v + h.o.offset[k]);
          dragObj = { o: h.o, mode: h.mode, start: w, off0: h.o.offset.slice(), s0: h.o.scale, c, d0: Math.hypot(w[0] - c[0], w[1] - c[1]) || 1e-6 };
          return;
        }
      }
      if (e.button !== 0 || tool === 'pan') { pan = { x: e.clientX, y: e.clientY, x0: e.clientX, y0: e.clientY, left: e.button === 0 }; return; }
      if (tilted()) { cam.yaw = 0; cam.pitch = 0; }          // drawing happens on the flat input plane
      if (tool === 'curve' || tool === 'region') { t = 0; anim = null; drawing = { type: tool, pts: [w] }; }
      else if (tool === 'circle') { t = 0; anim = null; drawing = { type: 'circle', c: w, r: 0 }; }
      else if (tool === 'pin') drawing = { type: 'pin', x: w, y: w.slice() };
      else if (tool === 'probe') { S.probe = w; invalidate(); }
      else if (tool === 'paint0' || tool === 'paint1') { t = 0; anim = null; drawing = { type: 'paint', label: +tool.slice(-1), last: w }; spray(w, drawing.label); }
    });
    cv.addEventListener('pointermove', (e) => {
      const r = cv.getBoundingClientRect(), w = viz2.toWorld(e.clientX - r.left, e.clientY - r.top);
      if (drawing && drawing.type === 'paint') {
        if (Math.hypot(w[0] - drawing.last[0], w[1] - drawing.last[1]) * viz2.scale > 6) { spray(w, drawing.label); drawing.last = w; }
        return;
      }
      if (dragObj) {
        const D = dragObj, o = D.o;
        if (D.mode === 'move') { o.offset[0] = D.off0[0] + w[0] - D.start[0]; o.offset[1] = D.off0[1] + w[1] - D.start[1]; }
        else o.scale = Math.max(0.1, Math.min(5, D.s0 * Math.hypot(w[0] - D.c[0], w[1] - D.c[1]) / D.d0));
        applyEdit(o);
        return;
      }
      if (!pan && !orbit && !drawing && !dragGoal) {             // cursor hints
        const h = objectHit(e.clientX - r.left, e.clientY - r.top);
        cv.style.cursor = h ? (h.mode === 'scale' ? 'nwse-resize' : 'move') : '';
      }
      if (dragGoal) {
        dragGoal.target[0] = +w[0].toFixed(2); dragGoal.target[1] = +w[1].toFixed(2);
        refreshGoalSamples(dragGoal);
        const card = document.querySelector(`.gcard[data-goal="${dragGoal.id}"]`);
        if (card) [0, 1].forEach((k) => { const el = card.querySelector(`[data-k="t${k}"]`); if (el) { el.value = dragGoal.target[k]; el.parentElement.querySelector('.val').textContent = dragGoal.target[k].toFixed(2); } });
        return;
      }
      if (orbit) {
        cam.yaw = Math.max(-1.5, Math.min(1.5, cam.yaw + 0.008 * (e.clientX - orbit.x)));
        cam.pitch = Math.max(-1.5, Math.min(1.5, cam.pitch + 0.008 * (e.clientY - orbit.y)));
        orbit.x = e.clientX; orbit.y = e.clientY;
        return;
      }
      if (pan) { viz2.panBy(e.clientX - pan.x, e.clientY - pan.y); pan.x = e.clientX; pan.y = e.clientY; userMovedView(); return; }
      if (!drawing) return;
      if (drawing.type === 'circle') drawing.r = Math.hypot(w[0] - drawing.c[0], w[1] - drawing.c[1]);
      else if (drawing.type === 'pin') drawing.y = w;
      else {
        const last = drawing.pts[drawing.pts.length - 1];
        if (Math.hypot(w[0] - last[0], w[1] - last[1]) * viz2.scale > 2) drawing.pts.push(w);
      }
    });
    const up = () => {
      if (dragObj) { dragObj = null; renderObjList(); }        // the card shows the new size and position
      // a plain click on empty space (no drag) clears the object selection
      if (pan && pan.left && Math.hypot(pan.x - pan.x0, pan.y - pan.y0) < 3 && selObj !== null && atInputFlat()) { selObj = null; renderObjList(); invalidate(); }
      pan = null; orbit = null; dragGoal = null; finishDrawing();
    };
    cv.addEventListener('pointerup', up);
    cv.addEventListener('pointercancel', () => { pan = null; orbit = null; drawing = null; });
    cv.addEventListener('dblclick', () => { if (tilted()) flatView(); });
    cv.addEventListener('wheel', (e) => {
      e.preventDefault();
      const r = cv.getBoundingClientRect();
      viz2.zoomAt(e.clientX - r.left, e.clientY - r.top, Math.exp(-wheelPixels(e) * 0.0015));
      userMovedView();
    }, { passive: false });
  }

  /** Spray a few training points of one class around a world position (the "paint your own" dataset). */
  function spray(w, label) {
    const g = () => { let u = 0; while (!u) u = Math.random(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * Math.random()); };
    for (let k = 0; k < 3 && S.custom.length < 1500; k++) S.custom.push({ x: [w[0] + 0.06 * g(), w[1] + 0.06 * g()], label });
    buildData();
    syncTargetUI();
    invalidate();
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

  /** 3D: a goal's target point A under the pointer (output step only), or null. */
  function goalAt3(px, py) {
    if (!isGoals() || S.dim !== 3 || !traces || t < traces.nStages - 1 - 1e-6) return null;
    for (const g of S.train.goals) {
      if (!g.target) continue;
      const q = viz3.toScreen(g.target);
      if (Math.hypot(q[0] - px, q[1] - py) < 14) return g;
    }
    return null;
  }

  function bind3DMouse() {
    const el = viz3.renderer.domElement;
    const pos = (e) => { const r = el.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top]; };
    // grabbing a goal's target must not also rotate the camera (viz3 asks this before it starts a drag)
    viz3.allowRotate = (e) => !(e.button === 0 && tool === 'pan' && goalAt3(...pos(e))) && (tool === 'pan' || e.button === 2);
    viz3.onUserZoom = userMovedView;
    let drag3 = null;           // { g, z } while a goal's target is dragged on its horizontal plane
    el.addEventListener('pointerdown', (e) => {
      if (e.button === 0 && tool === 'pan') {
        const g = goalAt3(...pos(e));
        if (g) { drag3 = { g, z: g.target[2] }; try { el.setPointerCapture(e.pointerId); } catch (err) { /* no capturable pointer (synthetic event) */ } }
        return;
      }
      if (tool === 'pan' || e.button !== 0) return;
      const [px, py] = pos(e);
      try { el.setPointerCapture(e.pointerId); } catch (err) { /* no capturable pointer (synthetic event) */ }
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
      if (drag3) {
        const p = viz3.pickPlaneZ(...pos(e), drag3.z);
        if (p) {
          drag3.g.target[0] = +p[0].toFixed(2); drag3.g.target[1] = +p[1].toFixed(2);
          refreshGoalSamples(drag3.g);
          const card = document.querySelector(`.gcard[data-goal="${drag3.g.id}"]`);
          if (card) [0, 1].forEach((k) => { const s2 = card.querySelector(`[data-k="t${k}"]`); if (s2) { s2.value = drag3.g.target[k]; s2.parentElement.querySelector('.val').textContent = drag3.g.target[k].toFixed(2); } });
          lastSig = '';                                       // redraw the marker
        }
        return;
      }
      if (!drawing) return;
      const [px, py] = pos(e);
      if (drawing.type === 'sphereDraw') { const p = viz3.pickSphere(px, py, S.sphere.c, S.sphere.r); if (p) drawing.pts.push(p); }
      else if (drawing.type === 'pin') { const p = viz3.pickPlaneZ0(px, py); if (p) drawing.y = p; }
    });
    el.addEventListener('pointerup', () => { drag3 = null; finishDrawing(); });
  }

  // ===========================================================================
  // Training loop
  // ===========================================================================

  function trainSteps(k) {
    if (!trainer.tasks.length) { training = false; updateTrainStatus('Nothing to train on: choose a target (or add pins).'); return; }
    for (let i = 0; i < k; i++) {
      const r = trainer.step();
      if (r === null) { training = false; break; }
      if (++stepsSinceRedraw >= S.train.redrawEvery) { stepsSinceRedraw = 0; invalidate(); netVersion++; pushHistory(); }
    }
    if (!training) { invalidate(); netVersion++; pushHistory(); }
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
    if (acc !== null && T.step_ > 0) parts.push(`accuracy ${(100 * acc).toFixed(1)}% (${isAnchors() ? 'nearest target point' : isGoals() && !goalSplit() ? 'nearer class point' : 'larger logit'})`);
    if (T.step_ > 0 && (isAnchors() || isMorph())) {
      const wt = worstOn(data), wd = denseData() ? worstDense() : null;
      if (wd !== null && wd !== undefined) {
        parts.push(`worst point off by ${wd.toFixed(2)} (on a 50× finer copy of the shape; ${wt.toFixed(2)} on the training points)`);
        if (wd > 3 * wt + 0.2) parts.push('⚠ the training points look fine, but the curve between two of them is stretched far away — only the finer check sees it');
      } else if (wt !== null) parts.push(`worst point off by ${wt.toFixed(2)}`);
    }
    if (S.train.mode === 'sequential' && T.tasks.length) parts.push(T.done ? 'all tasks done' : `task ${T.task + 1}/${T.tasks.length}`);
    if (T.diverged) parts.push('diverged — lower the learning rate and press Reset');
    if (!T.tasks.length) parts.push(S.train.target === 'pins' ? 'add pins with the Pin tool' : 'choose a target');
    el.textContent = parts.join(' · ');
    renderExpResult();
  }

  /** One plain sentence saying what the chosen task asks the network to do. */
  function targetNote() {
    const z = S.dim === 3 ? ', 0' : '';
    switch (S.train.target) {
      case 'anchors': return `Every blue point should land on (1, 0${z}) and every red point on (−1, 0${z}). Loss: squared distance to its point. Watch each class shrink onto its point.`;
      case 'classify': return 'Two output numbers (logits): blue points should end up on one side of the line logit₁ = logit₂, red points on the other. Loss: cross-entropy.';
      case 'transform': return 'Every input x should go to A·x, where A is the chosen linear map (drawn dashed at the output).';
      case 'morph': { const [a, b] = morphPair(); return `Point i of the ${NN.SHAPES[a].label} should land on point i of the ${NN.SHAPES[b].label} (drawn dashed at the output). Watch “worst point off by”: a topological obstruction leaves some points far from their target.`; }
      case 'pins': return 'With the Pin tool, drag from an input point to where its output should go; the network bends space to satisfy every pin.';
      case 'goals': return 'Your own goals, trained together. Each goal says which points should go where; its weight says how much it counts. Drag a target point A at the output to move it.';
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

  const DATASET_NOTES = {
    'shape:diskInRing': 'An exact disk inside a ring (no noise). Sending them to two points needs the disk to leave the ring — impossible for a homeomorphism of the plane.',
    'shape:linkedRings': 'Two exact linked rings (no noise). No homeomorphism of 3D space unlinks them.',
    'shape:nestedSpheres': 'An exact ball inside a spherical shell. No homeomorphism of 3D space gets the ball out.',
    blobs: 'Two separate clouds: one straight line already splits them, so the network hardly needs to bend space.',
    moons: 'Two interleaving half-moons: one bend is enough.',
    circles: 'A disk inside a ring: no line separates them, and no smooth invertible bending of the plane can move the disk out of the ring — the network has to squash space or lift the disk into an extra dimension (width ≥ 3).',
    rings: 'Blue–red–blue rings: separating the middle ring takes two folds.',
    xor: 'XOR: opposite quadrants share a colour (3D: octants by sign). No single line works — the network must fold space.',
    wave: 'A wavy boundary: smooth but curved. Try sin activations.',
    spirals: 'Two interleaved spirals: many folds are needed — use more layers or width.',
    checker: 'Checkerboard: many small regions — a test of capacity (width × depth).',
    linked: 'Two linked rings. No smooth invertible deformation of 3D space can unlink them: with width 3 the network only gets close by crushing a dimension (see Invertibility); width ≥ 4 separates them cleanly.',
    custom: 'Paint your own data with “Paint blue” / “Paint red” in the toolbar (drag to spray points).',
  };

  function syncTargetUI() {
    // datasets that only exist in one dimension
    const sel = $('dataset');
    sel.querySelector('option[value="linked"]').hidden = S.dim !== 3;
    sel.querySelector('option[value="custom"]').hidden = S.dim !== 2;
    for (const o of sel.querySelectorAll('option[value^="shape:"]')) o.hidden = NN.SHAPES[o.value.slice(6)].dim !== S.dim;
    const dsBad = (S.train.dataset.startsWith('shape:') && NN.SHAPES[S.train.dataset.slice(6)].dim !== S.dim);
    if (dsBad || (S.train.dataset === 'linked' && S.dim !== 3) || (S.train.dataset === 'custom' && S.dim !== 2)) {
      S.train.dataset = 'moons'; sel.value = 'moons';
    }
    const custom = usesClasses() && S.train.dataset === 'custom';
    $('datasetNote').classList.toggle('hidden', !usesClasses());
    $('datasetNote').textContent = DATASET_NOTES[S.train.dataset] || '';
    $('customRow').classList.toggle('hidden', !custom);
    $('customCount').textContent = `${S.custom.filter((p) => p.label === 0).length} blue · ${S.custom.filter((p) => p.label === 1).length} red`;
    $('clearCustom').disabled = !S.custom.length;
    $('datasetRow').classList.toggle('hidden', !usesClasses());
    // morph pairs of the current dimension
    const mp = $('morphPair'), pairs = MORPHS[S.dim];
    if (!pairs.some((p) => p.join('>') === S.train.morph)) S.train.morph = pairs[0].join('>');
    mp.innerHTML = pairs.map((p) => `<option value="${p.join('>')}">${NN.SHAPES[p[0]].label} → ${NN.SHAPES[p[1]].label}</option>`).join('');
    mp.value = S.train.morph;
    $('goalsBox').classList.toggle('hidden', !isGoals());
    if (isGoals()) renderGoals();
    $('morphRow').classList.toggle('hidden', !isMorph());
    $('transformRow').classList.toggle('hidden', !isTransform());
    $('amountRow').classList.toggle('hidden', !isTransform());
    $('targetNote').textContent = targetNote() + outputWarning();
  }

  function onTargetChange() {
    if (isGoals() && !S.train.goals.length) S.train.goals = defaultGoals();
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
    $('clearObjects').disabled = !S.objects.length && !S.pins.length;
    box.innerHTML = '';
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

  /** Is a panel showing?  Closed <details> and panels in a hidden popover / sheet / drawer are skipped. */
  const isOpen = (sec) => {
    const el = document.querySelector(`[data-sec="${sec}"]`);
    if (!el) return true;
    if (el.tagName === 'DETAILS' && !el.open) return false;
    return el.getClientRects().length > 0;
  };

  function updateAnalysis() {
    if (isOpen('loss')) drawLoss();
    if (isOpen('forgetting')) updateForgetting();
    if (isOpen('interference')) updateInterference();
    if (isOpen('sensitivity')) drawSensitivity();
    if (isOpen('invertibility')) updateInvertibility();
  }

  function updateForgetting() {
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
  }

  function updateInterference() {
    const kind = S.analysis.matrix;
    const samples = interferenceSamples();
    if (kind === 'cosine' && !samples.length) {
      Charts.heatmap($('heatmap'), [], { empty: 'choose a target or add pins' });
      $('heatNote').textContent = '';
    } else {
      const M = kind === 'ntk' ? NN.ntkMatrix(net, samples.map((s) => s.x)) : NN.gradientCosine(net, samples, isClassify() ? 'ce' : 'mse');
      Charts.heatmap($('heatmap'), M, { mode: kind === 'ntk' ? 'sequential' : 'diverging' });
      $('heatNote').textContent = kind === 'ntk'
        ? `K(xᵢ, xⱼ) = tr(Jᵢ Jⱼᵀ) for the ${samples.length} ${S.train.target === 'pins' ? 'pins' : 'numbered points in the view'}; brighter = training one point moves the other's output more.`
        : `cos(∇Lᵢ, ∇Lⱼ) for the ${samples.length} ${S.train.target === 'pins' ? 'pins' : 'numbered points in the view'}: gold = they agree, violet = they fight (interference).`;
    }
  }

  function updateInvertibility() {
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

    for (const k of ['grid', 'circle', 'basis', 'origin', 'jac', 'det', 'dist', 'autoFit', 'bg', 'lift', 'basisInfo', 'mats']) {
      const el = $('show' + k[0].toUpperCase() + k.slice(1)) || $(k);
      el.onchange = () => { S.display[k] = el.checked; invalidate(); };
    }
    $('fitBtn').onclick = fitView;
    $('showMats').addEventListener('change', () => renderMats(true));
    $('flatBtn').onclick = flatView;
    $('showLift').addEventListener('change', () => { if (!S.display.lift) flatView(); });
    $('resetView').onclick = () => { viz2.cx = viz2.cy = 0; viz2.scale = Math.min(viz2.w, viz2.h) / 6; if (viz3) { viz3.orbit = { theta: 0.8, phi: 1.1, radius: 7 }; viz3.target.set(0, 0, 0); } };

    $('target').onchange = (e) => { S.train.target = e.target.value; training = false; onTargetChange(); };
    $('dataset').onchange = (e) => { S.train.dataset = e.target.value; training = false; rebuildTrainer(); syncTargetUI(); renderToolbar(); invalidate(); };
    $('clearCustom').onclick = () => { training = false; S.custom = []; rebuildTrainer(); syncTargetUI(); invalidate(); };
    $('morphPair').onchange = (e) => { S.train.morph = e.target.value; training = false; rebuildTrainer(); syncTargetUI(); invalidate(); };
    $('homeo').onchange = (e) => {
      S.net.homeo = e.target.checked;
      if (S.net.homeo) NN.clampSingularValues(net, HOMEO_FLOOR);   // make the current weights invertible right away
      trainer.setOpts({ invertibleFloor: S.net.homeo ? HOMEO_FLOOR : 0 });
      netVersion++; renderNetNote(); invalidate();
    };
    $('addShape').onclick = addShape;
    bindObjects();
    bindGoals();
    $('expRun').onclick = () => runExperiment(0);
    $('expPlus').onclick = () => runExperiment(1);
    $('transform').onchange = (e) => { S.train.transform = e.target.value; training = false; rebuildTrainer(); invalidate(); };
    $('optimizer').onchange = (e) => { S.train.optimizer = e.target.value; trainer.setOpts({ optimizer: e.target.value }); };
    $('method').onchange = (e) => { S.train.method = e.target.value; trainer.setOpts({ method: e.target.value }); syncModeUI(); };
    $('modeJoint').onclick = () => { S.train.mode = 'joint'; syncModeUI(); rebuildTrainer(); };
    $('modeSeq').onclick = () => { S.train.mode = 'sequential'; syncModeUI(); rebuildTrainer(); };
    $('trainPlay').onclick = () => {
      if (!training) branchFromView();
      if (!training && trainer.done) rebuildTrainerKeepWeights();
      training = !training; updateTrainStatus();
    };
    $('toolChip').onclick = () => setTool('pan');
    $('trainStep').onclick = () => { training = false; branchFromView(); trainSteps(1); invalidate(); netVersion++; };
    $('trainReset').onclick = () => { training = false; rebuildNet(); };
    $('matrixKind').onchange = (e) => { S.analysis.matrix = e.target.value; invalidate(); };

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
    $('clearObjects').onclick = () => {                     // one bin for everything placed in space: objects and pins
      const hadPins = S.pins.length > 0;
      S.objects = []; S.pins = [];
      renderObjList(); invalidate();
      if (hadPins) { training = false; onPinsChanged(); }
    };
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
  /** [layers, width] that train well for a task and data set (shared with the guided start). */
  function recommendedArch(d, target, dataset) {
    if (target === 'anchors' || target === 'classify' || target === 'goals') {   // goals: class data, sent to points
      const a = { blobs: [2, d], moons: [4, d], circles: [4, d === 3 ? 6 : 4], rings: [5, 6], xor: [3, 4], wave: [4, 6],
        spirals: [6, 6], checker: [6, 8], linked: [4, 4], custom: [5, 6] }[dataset] || (dataset.startsWith('shape:') ? [4, d + 1] : [4, 6]);
      // squashing a whole class onto one point needs room to fold: width d left two-moons / XOR at ~85 %
      // after 3000 steps, width d + 2 reaches 100 % (blobs and the exact topology shapes keep theirs)
      if ((target === 'anchors' || target === 'goals') && dataset !== 'blobs' && !dataset.startsWith('shape:')) return [a[0], Math.max(a[1], d + 2)];
      return a;
    }
    if (target === 'transform') return [2, 4];
    if (target === 'morph') return [4, d + 1];
    if (target === 'pins') return [3, 6];
    return [3, d];
  }

  /** Initialisation that goes with a task: Xavier for training, strong He weights for "just look". */
  const initFor = (target) => (target !== 'none' ? { dist: 'xavier', scale: 1 } : { dist: 'he', scale: 1.6 });

  function applyRecommended() {
    const T = S.train;
    const arch = recommendedArch(S.dim, T.target, T.dataset);
    const training = T.target !== 'none';
    Object.assign(S.net, { layers: arch[0], width: arch[1], defaultAct: 'tanh', temperature: 0, overrides: [],
      outputLinear: training, homeo: false, init: { ...S.net.init, ...initFor(T.target) } });
    Object.assign(T, { optimizer: 'adam', lr: 0.01, batch: 32, stepsPerFrame: 10, redrawEvery: 10 });
    training_stop();
    rebuildNet();
    syncControls();
    fitView();
    flashHint(`★ Recommended: ${arch[0]} layers, width ${arch[1]}, tanh${training ? ', linear output, Xavier init' : ', He init ×1.6'}, Adam lr 0.01 — view fitted.`);
  }
  function training_stop() { training = false; updateTrainStatus(); }

  /**
   * Set everything the guided start chose: dimension, task, data and network.
   *   g = { dim, problem, data: {dataset, n, noise, seed} | {map, amount} | {pair}, network: {layers, width, act, outputLinear} }
   * The weight seed stays at its default, so the data sample is the one the guide showed.
   */
  function applyGuided(g) {
    training_stop();
    setDim(g.dim);
    const T = S.train, d = g.data || {};
    T.target = g.problem;
    if (g.problem === 'classify' || g.problem === 'goals') Object.assign(T, { dataset: d.dataset, nData: d.n, noise: d.noise, dataSeed: d.seed || 0 });
    if (g.problem === 'goals') T.goals = defaultGoals();
    else if (g.problem === 'transform') Object.assign(T, { transform: d.map, amount: d.amount });
    else if (g.problem === 'morph') T.morph = d.pair;
    Object.assign(T, { optimizer: 'adam', lr: 0.01, batch: 32, stepsPerFrame: 10, redrawEvery: 10, mode: 'joint', method: 'none' });
    const n = g.network;
    Object.assign(S.net, { layers: n.layers, width: n.width, defaultAct: n.act, temperature: 0, overrides: [],
      outputLinear: n.outputLinear, homeo: false, init: { ...defaultState(g.dim).net.init, ...initFor(g.problem) } });
    t = 0; anim = null; playing = false;
    syncTargetUI();
    rebuildNet();
    syncControls();
    fitView();
    flashHint(g.problem === 'none'
      ? 'Press ▶ in the strip below (or Space) to send the grid through the network, layer by layer.'
      : 'Press ▶ Train (bottom right, or T) to start training, and watch the space bend.');
  }

  // ===========================================================================
  // Shapes and topology experiments
  // ===========================================================================

  // shape pairs offered by the "turn one shape into another" task
  const SHAPE_HIDDEN = new Set(['oneCircleTwice', 'torusOnSphere']);   // only used as morph targets

  function renderShapeSel() {
    $('shapeSel').innerHTML = Object.entries(NN.SHAPES).filter(([k, v]) => v.dim === S.dim && !SHAPE_HIDDEN.has(k))
      .map(([k, v]) => `<option value="${k}">${v.label}</option>`).join('');
  }

  /** Add a ready-made shape as objects (one object per component). */
  function addShape() {
    const sh = NN.makeShape($('shapeSel').value, S.nPoints);
    for (const c of sh.components) {
      const color = c.label === undefined ? undefined : CLASS_COLORS[c.label];
      const surface = S.dim === 3 && !c.closed;
      if (c.filled) {                                   // filled disk: boundary circle + interior points
        const r = Math.max(...c.pts.map((p) => Math.hypot(...p)));
        addObject({ type: `${sh.label} (filled)`, closed: true, color, interior: c.pts.map((p) => Array.from(p)),
          points: linspace(0, 2 * Math.PI, 121).slice(0, 120).map((a) => [r * Math.cos(a), r * Math.sin(a)]) });
      } else addObject({ type: sh.label, closed: c.closed, surface, color, points: c.pts.map((p) => Array.from(p)) });
    }
  }

  /*
   * Each experiment pushes the network to the limit of what a homeomorphism can do: width = dimension,
   * invertible layers (homeomorphism mode), tanh.  "+1 dimension" repeats it with width d + 1.
   * The measured numbers come from test/topology_experiments.js (4 layers, 6000 Adam steps).
   */
  /*
   * Each experiment runs at the limit of what a homeomorphism can do (width = d, invertible layers,
   * tanh) and again with one extra hidden dimension.  Outcomes: 'works' | 'fails' | 'sometimes' |
   * 'partly'.  Numbers come from test/topology_experiments.js ("worst" = worst point on a 50× finer
   * copy of the shape).
   */
  const EXPERIMENTS = {
    2: [
      { title: 'Circle → square', task: 'morph', morph: 'circle>square', layers: 4, lr: 0.01, seed: 1,
        goal: 'Turn a circle into a square.',
        atD: 'Works: a circle and a square are the same shape topologically — only bending is needed.',
        plus: 'Works too (control experiment).',
        d: 'works', d1: 'works', measured: 'width 2: worst 0.10–0.14 (4 of 4 runs) · width 3: ≤ 0.07' },
      { title: 'Disk out of the ring', task: 'anchors', dataset: 'shape:diskInRing', layers: 4, lr: 0.01, seed: 1,
        goal: 'Send the disk to (1, 0) and the ring around it to (−1, 0).',
        atD: 'Fails: the disk would have to cross the ring. Bending the plane can never move something out of a closed ring.',
        plus: 'Works: in 3D the disk simply lifts over the ring.',
        d: 'fails', d1: 'works', measured: 'width 2: worst 1.5–2.1 (0 of 4 runs solved) · width 3: ≤ 0.25 (4 of 4)' },
      { title: 'Two circles → one', task: 'morph', morph: 'twoCircles>oneCircleTwice', layers: 4, lr: 0.01, seed: 1,
        goal: 'Merge two separate circles into one circle.',
        atD: 'Fails: merging glues two different points onto one — an invertible map never does that.',
        plus: 'Works: the last layer goes from 3D back to 2D, and that projection can put two points on top of each other.',
        d: 'fails', d1: 'works', measured: 'width 2: worst 1.1–1.7 (0 of 4) · width 3: ≤ 0.02 (4 of 4)' },
      { title: 'Circle → figure eight', task: 'morph', morph: 'circle>figure8', layers: 4, lr: 0.01, seed: 1,
        goal: 'Turn a circle into a figure eight (∞).',
        atD: 'Fails: the figure eight crosses itself, so two points of the circle must land on the same spot (gluing).',
        plus: 'Works: in 3D the curve crosses over itself; projected back to 2D it looks like the ∞.',
        d: 'fails', d1: 'works', measured: 'width 2: worst 0.47–0.65 (0 of 4) · width 3: ≤ 0.02 (4 of 4)' },
    ],
    3: [
      { title: 'Sphere → ellipsoid', task: 'morph', morph: 'sphere>ellipsoid', layers: 4, lr: 0.01, seed: 1,
        goal: 'Stretch and tilt a sphere into an ellipsoid.',
        atD: 'Works: same shape topologically — only bending and stretching.',
        plus: 'Works too (control experiment).',
        d: 'works', d1: 'works', measured: 'width 3: worst ≤ 0.02 (4 of 4) · width 4: ≤ 0.01' },
      { title: 'Unlink the rings', task: 'anchors', dataset: 'shape:linkedRings', layers: 8, lr: 0.003, seed: 1,
        goal: 'Send one of two linked rings to (1, 0, 0) and the other to (−1, 0, 0).',
        atD: 'Fails: linked rings stay linked under any bending of 3D space. (It can look solved on the training points — the curve between two of them gets stretched around the other ring; the worst-point check catches it.)',
        plus: 'Possible: in 4D one ring can slip past the other. Training finds it only sometimes — this experiment uses a seed where it does.',
        d: 'fails', d1: 'sometimes', measured: 'width 3: worst ≈ 2 (0 of 4) · width 4: solved in 2 of 8 runs' },
      { title: 'Ball out of the shell', task: 'anchors', dataset: 'shape:nestedSpheres', layers: 6, lr: 0.003, seed: 1,
        goal: 'Send the inner ball to (1, 0, 0) and the shell around it to (−1, 0, 0).',
        atD: 'Fails: the 3D version of the disk in the ring — the ball cannot get through the closed shell.',
        plus: 'Works: in 4D the ball passes around the shell.',
        d: 'fails', d1: 'works', measured: 'width 3: worst 1.9–2.1 (0 of 4) · width 4: ≤ 0.07 (4 of 4)' },
      { title: 'Unknot → trefoil', task: 'morph', morph: 'unknot>trefoil', layers: 4, lr: 0.01, seed: 3,
        goal: 'Tie a plain ring into a trefoil knot.',
        atD: 'Fails: bending 3D space can never tie or untie a knot — the curve gets stuck at the crossings.',
        plus: 'Works: in 4D every knot can be tied and untied.',
        d: 'fails', d1: 'works', measured: 'width 3: worst 0.37–0.51 (0 of 4) · width 4: ≤ 0.04 (3 of 4 runs; this seed works)' },
      { title: 'Torus → sphere', task: 'morph', morph: 'torus>torusOnSphere', layers: 4, lr: 0.01, seed: 1,
        goal: 'Close the hole of a torus (donut) so it becomes a sphere.',
        atD: 'Fails: the hole can only close by folding the tube onto itself everywhere (gluing).',
        plus: 'Partly: the extra dimension lets it fold, and the error drops a lot, but it never gets close to zero here.',
        d: 'fails', d1: 'partly', measured: 'width 3: worst 0.39–0.62 · width 4: 0.17–0.26' },
    ],
  };
  const OUTCOME = {
    works: { icon: '✓', cls: 'ok', word: 'works' },
    fails: { icon: '✗', cls: 'no', word: 'fails' },
    sometimes: { icon: '◐', cls: 'mid', word: 'sometimes' },
    partly: { icon: '◐', cls: 'mid', word: 'partly' },
  };
  let currentExp = null, expRun = null;      // expRun = { exp, plus }: the run the result line refers to

  /** Table of experiments: what happens at width d and with one more dimension. */
  function renderExperiments() {
    const d = S.dim, cell = (o) => `<td class="oc ${OUTCOME[o].cls}">${OUTCOME[o].icon} ${OUTCOME[o].word}</td>`;
    if (currentExp && !EXPERIMENTS[d].includes(currentExp)) currentExp = null;
    $('expTable').innerHTML = `<tr><th>Experiment (${d}D)</th><th>width ${d}</th><th>+1 dim (${d + 1})</th></tr>`
      + EXPERIMENTS[d].map((e, i) => `<tr data-i="${i}" class="${currentExp === e ? 'sel' : ''}"><td>${e.title}</td>${cell(e.d)}${cell(e.d1)}</tr>`).join('');
    for (const tr of $('expTable').querySelectorAll('tr[data-i]')) {
      tr.onclick = () => { currentExp = EXPERIMENTS[d][+tr.dataset.i]; renderExperiments(); };
    }
    showExperiment();
  }

  function showExperiment() {
    const e = currentExp;
    $('expCard').classList.toggle('hidden', !e);
    if (!e) return;
    const d = S.dim, expect = (o) => (o === 'works' ? 'expected: works' : o === 'fails' ? 'expected: fails' : `expected: ${o}`);
    $('expText').innerHTML = `<div class="exp-title">${e.title}</div>
      <div class="exp-row"><span>Goal</span><span>${e.goal}</span></div>
      <div class="exp-row"><span>Width ${d}</span><span class="${OUTCOME[e.d].cls}">${OUTCOME[e.d].icon}</span><span>${e.atD}</span></div>
      <div class="exp-row"><span>Width ${d + 1}</span><span class="${OUTCOME[e.d1].cls}">${OUTCOME[e.d1].icon}</span><span>${e.plus}</span></div>
      <div class="exp-row"><span>Watch</span><span>“worst point off by” under Training: near 0 = solved; around 2 = some point landed at the wrong target.</span></div>
      <div class="tag">Measured: ${e.measured}.</div>`;
    $('expRun').textContent = `▶ Width ${d} — ${expect(e.d)}`;
    $('expPlus').textContent = `▶ Width ${d + 1} — ${expect(e.d1)}`;
    renderExpResult();
  }

  /** Live verdict for the experiment that is (or was last) trained. */
  function renderExpResult() {
    const el = $('expResult');
    if (!el) return;
    if (!expRun || expRun.exp !== currentExp || !trainer || !trainer.step_) { el.textContent = ''; el.className = 'exp-result'; return; }
    const w = worstDense(), d = S.dim + expRun.plus;
    if (w === null || w === undefined) return;
    const solved = w < 0.3;
    el.className = 'exp-result ' + (solved ? 'ok' : 'no');
    el.textContent = `Your run (width ${d}, ${trainer.step_} steps): worst point off by ${w.toFixed(2)} → ${solved ? '✓ solved' : training ? '✗ not solved yet' : '✗ not solved'}`;
  }

  /** Configure everything for the experiment and start training. plus = 1 adds a hidden dimension. */
  function runExperiment(plus) {
    const e = currentExp;
    if (!e) return;
    const d = S.dim;
    Object.assign(S.train, { target: e.task, mode: 'joint', optimizer: 'adam', lr: e.lr, batch: 64, stepsPerFrame: 20, redrawEvery: 20 });
    if (e.morph) S.train.morph = e.morph;
    if (e.dataset) S.train.dataset = e.dataset;
    Object.assign(S.net, { layers: e.layers, width: d + plus, defaultAct: 'tanh', temperature: 0, overrides: [], outputLinear: true,
      homeo: true, init: { dist: 'xavier', scale: 1, seed: e.seed } });
    training = false;
    rebuildNet();
    // invertible matrices come in two pieces (det > 0, det < 0) and homeomorphism-mode training cannot
    // cross between them, so start orientation-preserving like the targets
    NN.orientPositive(net);
    NN.clampSingularValues(net, HOMEO_FLOOR);
    syncControls();
    t = 1e9;
    fitView();
    training = true;
    expRun = { exp: e, plus };
    updateTrainStatus();
    flashHint(`${e.title}: ${e.layers} layers, width ${d + plus}${plus ? ' (one extra dimension)' : ''}, tanh, invertible layers, seed ${e.seed} — training…`);
  }

  /** Under the network settings: what homeomorphism mode guarantees with the current activations. */
  function renderNetNote() {
    const el = $('netNote');
    if (!el || !net) return;
    if (!S.net.homeo) { el.textContent = ''; return; }
    const bad = [...new Set(net.acts.filter((a) => !NN.ACTIVATIONS[a].injective))];
    const wide = S.net.width !== S.dim;
    el.textContent = bad.length
      ? `⚠ ${bad.join(', ')} ${bad.length > 1 ? 'are' : 'is'} not injective, so the network can still fold space even with invertible weights. Use tanh, sigmoid or identity.`
      : wide ? `Square layers are kept invertible. With width ${S.net.width} ≠ ${S.dim} the first and last layers change dimension, so the whole network is no longer a homeomorphism of the ${S.dim}D space.`
        : 'Every layer is invertible: the network is a homeomorphism — it can bend and stretch space but cannot tear or glue it.';
  }

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
    $('homeo').checked = !!S.net.homeo;
    renderShapeSel();
    renderPresetButtons();
    renderExperiments();
    renderNetNote();
    $('initDist').value = S.net.init.dist;
    $('seed').value = S.net.init.seed;
    for (const k of ['grid', 'circle', 'basis', 'origin', 'jac', 'det', 'dist', 'autoFit', 'bg', 'lift', 'basisInfo', 'mats']) {
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

  // ---- my goals: cards in the Goal column -------------------------------------------------------------------
  function renderGoals() {
    const box = $('goalsList'), axes = S.dim === 2 ? ['x', 'y'] : ['x', 'y', 'z'];
    const objOpts = (sel) => S.objects.map((o) => `<option value="${o.id}"${o.id === sel ? ' selected' : ''}>${escHtml(o.type)}</option>`).join('');
    const slider = (gid, key, label, min, max, step, v) => `<div class="row"><label>${label}</label>
      <input data-g="${gid}" data-k="${key}" type="range" min="${min}" max="${max}" step="${step}" value="${v}"><span class="val">${(+v).toFixed(2)}</span></div>`;
    box.innerHTML = S.train.goals.map((g) => {
      const K = GOAL_KINDS[g.kind];
      let body = '';
      if (g.kind === 'classPoint') {
        body += `<div class="row"><label>Class</label><select data-g="${g.id}" data-k="cls"><option value="0"${g.cls === 0 ? ' selected' : ''}>blue</option><option value="1"${g.cls === 1 ? ' selected' : ''}>red</option></select></div>`;
      }
      if (g.kind === 'objectPoint' || g.kind === 'objectStay') {
        body += S.objects.length ? `<div class="row"><label>Object</label><select data-g="${g.id}" data-k="obj">${objOpts(g.obj)}</select></div>`
          : '<div class="note">Add an object first (＋ Objects).</div>';
      }
      if (g.target) body += axes.map((a, k) => slider(g.id, `t${k}`, `A ${a}`, -2.5, 2.5, 0.05, g.target[k])).join('');
      body += slider(g.id, 'weight', 'Weight', 0, 3, 0.05, g.weight);
      return `<div class="gcard" data-goal="${g.id}"><div class="ghead"><b>${K.name}</b><span class="gloss tag" data-loss="${g.id}"></span>
        <button class="x" data-g="${g.id}" data-k="remove" data-tip="Remove this goal">✕</button></div>${body}<div class="gformula">${K.formula(g)}</div></div>`;
    }).join('') || '<div class="note">No goals yet: add one below.</div>';
    $('goalsTotal').textContent = S.train.goals.length > 1 ? `Total: L = (1/${S.train.goals.length}) Σ wᵢ Lᵢ` : '';
    updateGoalLosses(true);
  }

  let goalLossAt = 0;
  /** Each goal's own (unweighted) loss, next to its name. */
  function updateGoalLosses(force) {
    if (!isGoals() || (!force && performance.now() - goalLossAt < 400) || !isOpen('goalsbox')) return;
    goalLossAt = performance.now();
    for (const g of S.train.goals) {
      const el = document.querySelector(`[data-loss="${g.id}"]`), mine = data.filter((q) => q.goal === g.id);
      if (!el) continue;
      if (!mine.length) { el.textContent = ''; continue; }
      let L = 0;
      for (const q of mine) L += NN.sampleLoss(net, { ...q, w: 1 }, q.type);
      el.textContent = `L = ${fmt(L / mine.length)}`;
    }
  }

  function addGoal(kind) {
    const z = S.dim === 3 ? [0] : [], g = { id: goalNextId++, kind, weight: 1 };
    if (kind === 'classPoint') { const used = S.train.goals.filter((x) => x.kind === 'classPoint').map((x) => x.cls); g.cls = used.includes(0) ? 1 : 0; g.target = [g.cls ? -1 : 1, 0, ...z]; }
    if (kind === 'objectPoint') { g.target = [0, 1, ...z]; }
    if (kind === 'objectPoint' || kind === 'objectStay') g.obj = S.objects.length ? S.objects[S.objects.length - 1].id : undefined;
    S.train.goals.push(g);
    renderGoals(); rebuildTrainer(); invalidate();
  }

  function bindGoals() {
    $('goalAdd').addEventListener('click', (e) => { const b = e.target.closest('[data-kind]'); if (b) addGoal(b.dataset.kind); });
    const box = $('goalsList');
    box.addEventListener('click', (e) => {
      const b = e.target.closest('[data-k="remove"]');
      if (!b) return;
      S.train.goals = S.train.goals.filter((g) => g.id !== +b.dataset.g);
      renderGoals(); rebuildTrainer(); invalidate();
    });
    box.addEventListener('input', (e) => {
      const el = e.target, g = S.train.goals.find((x) => x.id === +el.dataset.g), k = el.dataset.k;
      if (!g || !k || el.tagName === 'SELECT') return;
      if (k === 'weight') g.weight = +el.value;
      else if (k[0] === 't') g.target[+k.slice(1)] = +el.value;
      el.parentElement.querySelector('.val').textContent = (+el.value).toFixed(2);
      refreshGoalSamples(g);
    });
    box.addEventListener('change', (e) => {
      const el = e.target, g = S.train.goals.find((x) => x.id === +el.dataset.g), k = el.dataset.k;
      if (!g || el.tagName !== 'SELECT') return;
      if (k === 'cls') g.cls = +el.value;
      if (k === 'obj') g.obj = +el.value;
      renderGoals(); rebuildTrainer(); invalidate();
    });
  }

  /*
   * Move and resize objects right on the view (2D, flat, at the input step, Pan tool): click an object's
   * curve to select it (a dashed box appears), drag inside the box to move it, drag the corner square to
   * resize it.  The object's card follows.
   */
  let dragObj = null;           // { o, mode: 'move' | 'scale', start: [x, y], off0, s0, c }
  const atInputFlat = () => S.dim === 2 && !tilted() && t < 0.02 && tool === 'pan';
  function objBox(o) {
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (const p of o.points) { x0 = Math.min(x0, p[0]); x1 = Math.max(x1, p[0]); y0 = Math.min(y0, p[1]); y1 = Math.max(y1, p[1]); }
    return [x0, y0, x1, y1];
  }
  /** What is under the pointer: the selected object's corner handle, its box, or any object's curve. */
  function objectHit(px, py) {
    if (!atInputFlat()) return null;
    const sel = S.objects.find((o) => o.id === selObj && !o.hidden);
    if (sel) {
      const [x0, y0, x1, y1] = objBox(sel), [hx, hy] = viz2.toScreen(x1, y0);
      if (Math.abs(px - hx) < 9 && Math.abs(py - hy) < 9) return { o: sel, mode: 'scale' };
      const [ax, ay] = viz2.toScreen(x0, y1), [bx, by] = viz2.toScreen(x1, y0);
      if (px > ax - 4 && px < bx + 4 && py > ay - 4 && py < by + 4) return { o: sel, mode: 'move' };
    }
    for (let k = S.objects.length - 1; k >= 0; k--) {
      const o = S.objects[k];
      if (o.hidden) continue;
      for (const p of o.points) { const [qx, qy] = viz2.toScreen(p[0], p[1]); if (Math.abs(qx - px) < 6 && Math.abs(qy - py) < 6) return { o, mode: 'move' }; }
    }
    return null;
  }

  /** Drag a goal's target point A at the output stage (2D, flat view). */
  let dragGoal = null;
  function goalAt(px, py) {
    if (!isGoals() || S.dim !== 2 || tilted() || !traces || t < traces.nStages - 1 - 1e-6) return null;
    for (const g of S.train.goals) {
      if (!g.target) continue;
      const q = viz2.toScreen(g.target[0], g.target[1]);
      if (q && Math.hypot(q[0] - px, q[1] - py) < 14) return g;
    }
    return null;
  }

  // ---- objects: chips, a card for the selected one, ready-made shapes, transformation log ----------------
  const escHtml = (t) => String(t).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  let selObj = null;            // id of the object whose card is open

  /** Ready-made shapes for the Objects menu, centred at the origin; the card then moves and resizes them. */
  const PRESETS = {
    2: [['circle', 'Circle'], ['square', 'Square'], ['star', 'Star'], ['wave', 'Wave'], ['disk', 'Disk'], ['figure8', 'Figure 8']],
    3: [['sphere', 'Sphere'], ['cube', 'Cube'], ['torus', 'Torus'], ['ring', 'Ring'], ['trefoil', 'Knot']],
  };

  function addPreset(kind) {
    const n = S.nPoints, shape = (name) => NN.makeShape(name, n).components[0].pts.map((p) => Array.from(p));
    const name = Object.fromEntries(PRESETS[S.dim])[kind].toLowerCase();
    let o;
    if (kind === 'circle') o = { closed: true, points: linspace(0, 2 * Math.PI, n + 1).slice(0, n).map((a) => [0.6 * Math.cos(a), 0.6 * Math.sin(a)]) };
    else if (kind === 'wave') o = { closed: false, points: linspace(-1, 1, n).map((x) => [x, 0.3 * Math.sin(4 * x)]) };
    else if (kind === 'disk') {
      const r = 0.6, ga = Math.PI * (3 - Math.sqrt(5)), k = Math.round(n * 0.6);
      o = { closed: true, points: linspace(0, 2 * Math.PI, n - k + 1).slice(0, n - k).map((a) => [r * Math.cos(a), r * Math.sin(a)]),
        interior: [...Array(k)].map((_, i) => { const q = r * 0.97 * Math.sqrt((i + 0.5) / k); return [q * Math.cos(ga * i), q * Math.sin(ga * i)]; }) };
    } else if (kind === 'sphere') o = { closed: false, surface: true, points: NN.fibonacciSphere(n, [0, 0, 0], 0.6).map((p) => Array.from(p)) };
    else if (kind === 'ring') o = { closed: true, points: linspace(0, 2 * Math.PI, n + 1).slice(0, n).map((a) => [0.7 * Math.cos(a), 0.7 * Math.sin(a), 0]) };
    else {
      const map = { square: 'square', star: 'star', figure8: 'figure8', cube: 'cube', torus: 'torus', trefoil: 'trefoil' }[kind];
      const sh = NN.makeShape(map, n).components[0];
      o = { closed: sh.closed, surface: S.dim === 3 && !sh.closed, points: shape(map) };
    }
    addObject({ type: name, ...o });
    selObj = S.objects[S.objects.length - 1].id;
    renderObjList();
  }

  function renderPresetButtons() {
    $('presetBtns').innerHTML = PRESETS[S.dim].map(([k, label]) => `<button data-preset="${k}">${label}</button>`).join('');
  }

  /** Remember the shape as drawn, so size, position and point count can be changed without drift. */
  function ensureBase(o) {
    if (o.base) return;
    const c = o.points[0].map((_, k) => o.points.reduce((a, p) => a + p[k], 0) / o.points.length);
    o.base = { points: o.points.map((p) => p.slice()), interior: o.interior ? o.interior.map((p) => p.slice()) : null, c };
    o.scale = 1; o.offset = c.map(() => 0); o.n = o.points.length;
  }

  function applyEdit(o) {
    const B = o.base, place = (p) => p.map((v, k) => B.c[k] + o.scale * (v - B.c[k]) + o.offset[k]);
    const pts = o.n === B.points.length ? B.points : resample(B.points, o.n, o.closed);
    o.points = pts.map(place);
    if (B.interior) o.interior = B.interior.map(place);
    invalidate();
    renderLogSoon();
  }

  function renderObjList() {
    const box = $('objList');
    $('clearObjects').disabled = !S.objects.length && !S.pins.length;
    if (!S.objects.length) { box.innerHTML = '<div class="note">No objects yet: add a shape above, or draw one.</div>'; return; }
    if (!S.objects.some((o) => o.id === selObj)) selObj = null;
    box.innerHTML = '<div class="ochips">' + S.objects.map((o) => `<button class="ochip${o.id === selObj ? ' on' : ''}${o.hidden ? ' off' : ''}" data-id="${o.id}">
      <span class="dot" style="background:${o.color}"></span>${escHtml(o.type)}</button>`).join('') + '</div>';
    const o = S.objects.find((x) => x.id === selObj);
    if (!o) return;
    ensureBase(o);
    const axes = S.dim === 2 ? ['x', 'y'] : ['x', 'y', 'z'];
    const slider = (id, label, min, max, step, v, fmt) => `<div class="row"><label>${label}</label>
      <input data-edit="${id}" type="range" min="${min}" max="${max}" step="${step}" value="${v}"><span class="val">${fmt(v)}</span></div>`;
    box.insertAdjacentHTML('beforeend', `<div class="ocard">
      <div class="row"><label>Colour</label><input data-edit="color" type="color" value="${/^#[0-9a-f]{6}$/i.test(o.color) ? o.color : '#e3b341'}">
        <label class="inline"><input data-edit="visible" type="checkbox"${o.hidden ? '' : ' checked'}> visible</label></div>
      ${slider('scale', 'Size', 0.2, 3, 0.05, o.scale, (v) => `×${(+v).toFixed(2)}`)}
      ${axes.map((a, k) => slider(`off${k}`, `Move ${a}`, -2.5, 2.5, 0.05, o.offset[k], (v) => (+v >= 0 ? '+' : '') + (+v).toFixed(2))).join('')}
      ${slider('n', 'Points', 30, 500, 10, o.n, (v) => v)}
      <div class="btns"><button data-edit="log" class="primary">ⓘ How it changes</button><button data-edit="delete">Delete</button></div>
    </div>`);
  }

  function bindObjects() {
    $('presetBtns').addEventListener('click', (e) => { const b = e.target.closest('[data-preset]'); if (b) addPreset(b.dataset.preset); });
    const box = $('objList');
    box.addEventListener('click', (e) => {
      const chip = e.target.closest('.ochip');
      if (chip) { selObj = +chip.dataset.id === selObj ? null : +chip.dataset.id; renderObjList(); return; }
      const o = S.objects.find((x) => x.id === selObj);
      const act = e.target.closest('[data-edit]');
      if (!o || !act) return;
      if (act.dataset.edit === 'delete') { S.objects = S.objects.filter((x) => x !== o); selObj = null; renderObjList(); invalidate(); }
      else if (act.dataset.edit === 'log') { logObj = o.id; if (window.Shell) window.Shell.openSheet('object', false); renderObjLog(); }
    });
    box.addEventListener('input', (e) => {
      const o = S.objects.find((x) => x.id === selObj), el = e.target, k = el.dataset.edit;
      if (!o || !k) return;
      if (k === 'color') { o.color = el.value; box.querySelector(`.ochip[data-id="${o.id}"] .dot`).style.background = el.value; invalidate(); renderLogSoon(); return; }
      if (k === 'visible') { o.hidden = !el.checked; box.querySelector(`.ochip[data-id="${o.id}"]`).classList.toggle('off', o.hidden); invalidate(); return; }
      if (k === 'scale') o.scale = +el.value;
      else if (k === 'n') o.n = +el.value;
      else if (k.startsWith('off')) o.offset[+k.slice(3)] = +el.value;
      const val = el.parentElement.querySelector('.val');
      if (val) val.textContent = k === 'scale' ? `×${(+el.value).toFixed(2)}` : k === 'n' ? el.value : ((+el.value >= 0 ? '+' : '') + (+el.value).toFixed(2));
      applyEdit(o);
    });
  }

  // ---- transformation log: what every layer does to one object --------------------------------------------
  let logObj = null, logKey = '', logTimer = 0;
  function renderLogSoon() { logKey = ''; }

  function segCross(a, b, c, d) {
    const o = (p, q, r) => (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]);
    const d1 = o(a, b, c), d2 = o(a, b, d), d3 = o(c, d, a), d4 = o(c, d, b);
    return d1 * d2 < 0 && d3 * d4 < 0;
  }

  /** Per stage: length and stretch (in the stage's full space), area and self-crossings (2D stages only). */
  function objectStats(d, closed) {
    const n = d.st.length, out = [];
    const seg = (s) => {
      const L = [];
      for (let i = 0; i < n - 1 + (closed ? 1 : 0); i++) {
        const a = d.st[i][s], b = d.st[(i + 1) % n][s];
        let q = 0; for (let k = 0; k < a.length; k++) q += (a[k] - b[k]) ** 2;
        L.push(Math.sqrt(q));
      }
      return L;
    };
    const L0 = seg(0), len0 = L0.reduce((a, b) => a + b, 0);
    for (let s = 0; s < traces.nStages; s++) {
      const L = s ? seg(s) : L0, len = L.reduce((a, b) => a + b, 0), dim = d.st[0][s].length;
      let maxS = 0, minS = Infinity;
      for (let i = 0; i < L.length; i++) if (L0[i] > 1e-9) { const r = L[i] / L0[i]; maxS = Math.max(maxS, r); minS = Math.min(minS, r); }
      const row = { s, dim, len: len / (len0 || 1), maxS, minS, area: null, cross: null };
      if (dim === 2) {
        const P = d.st.map((p) => p[s]);
        if (closed) { let A = 0; for (let i = 0; i < n; i++) { const p = P[i], q = P[(i + 1) % n]; A += p[0] * q[1] - q[0] * p[1]; } row.area = A / 2; }
        let c = 0;
        const m = n - 1 + (closed ? 1 : 0), step = Math.max(1, Math.floor(n / 160));   // ≤ ~160 segments: fast enough per frame
        for (let i = 0; i < m; i += step) for (let j = i + 2 * step; j < m; j += step) {
          if (closed && i === 0 && j + step >= m) continue;
          if (segCross(P[i], P[Math.min(i + step, n - 1) % n], P[j], P[Math.min(j + step, m) % n])) c++;
        }
        row.cross = c;
      }
      out.push(row);
    }
    return out;
  }

  function renderObjLog() {
    const box = $('objLog'), o = S.objects.find((x) => x.id === logObj);
    if (!o) { box.innerHTML = '<div class="note">Pick an object and press ⓘ How it changes.</div>'; return; }
    if (!traces) computeTraces();
    const d = traces.ds.find((x) => x.id === `o${o.id}`);
    if (!d) { box.innerHTML = '<div class="note">This object is hidden.</div>'; return; }
    const st = objectStats(d, o.closed), a0 = st[0].area;
    const firstCross = st.find((r) => r.cross > 0);
    const fmtR = (r) => (r >= 10 ? r.toFixed(0) : r >= 1 ? r.toFixed(2) : r.toPrecision(2));
    let h = `<div class="olog-head"><span class="dot" style="background:${o.color}"></span><b>${escHtml(o.type)}</b> · ${d.st.length} points</div>
      <div class="note">Length and stretch are measured in each stage's own space ℝᵏ. Area and self-crossings only exist in 2D stages:
        in a wider layer a curve has room to pass around itself, and a crossing on screen is only the projection.</div>`;
    h += `<div class="note olog-sum">${o.closed && S.dim === 2 && a0 !== null && st[st.length - 1].area !== null
      ? (Math.sign(st[st.length - 1].area) !== Math.sign(a0) ? '⚠ The output is mirrored (orientation flipped). ' : '') : ''}${
      firstCross ? `⚠ The curve first crosses itself at <b>${stageName(firstCross.s).title}</b>: two different inputs land on the same point there.`
        : S.dim === 2 ? '✓ It never crosses itself in a 2D stage.' : ''}</div>`;
    h += '<table class="olog"><tr><th></th><th>stage</th><th data-tip="Length of the curve, relative to the input">length ×</th><th data-tip="Smallest … largest stretch of one piece of the curve">stretch</th><th data-tip="Enclosed area relative to the input; negative = mirrored">area ×</th><th data-tip="Places where the curve crosses itself (2D stages only)">cross</th></tr>';
    for (const r of st) {
      const nm = stageName(r.s);
      const area = r.area === null || !a0 ? '—' : `${r.area / a0 < 0 ? '−' : ''}${fmtR(Math.abs(r.area / a0))}`;   // negative: mirrored
      h += `<tr data-s="${r.s}"><td><canvas class="othumb" data-s="${r.s}"></canvas></td><td>${nm.title}<br><span class="tag">ℝ${String(r.dim).split('').map((c) => '⁰¹²³⁴⁵⁶⁷⁸⁹'[+c]).join('')}</span></td>
        <td>×${fmtR(r.len)}</td><td>${r.s ? `${fmtR(r.minS)}…${fmtR(r.maxS)}` : '—'}</td><td>${area}</td>
        <td class="${r.cross ? 'bad' : ''}">${r.cross === null ? '—' : r.cross}</td></tr>`;
    }
    box.innerHTML = h + '</table><div class="note">Click a row to show that stage.</div>';
    // thumbnails: the object at every stage, as the view shows it (projected for wide layers)
    box.querySelectorAll('canvas.othumb').forEach((cv) => {
      const s = +cv.dataset.s, P = d.proj[s], vd = traces.vd, ctx = cv.getContext('2d'), W = 44, dpr = window.devicePixelRatio || 1;
      cv.width = cv.height = W * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
      for (let i = 0; i < P.length; i += vd) { x0 = Math.min(x0, P[i]); x1 = Math.max(x1, P[i]); y0 = Math.min(y0, P[i + 1]); y1 = Math.max(y1, P[i + 1]); }
      const k = (W - 6) / Math.max(x1 - x0, y1 - y0, 1e-9);
      ctx.strokeStyle = o.color; ctx.lineWidth = 1.2; ctx.beginPath();
      for (let i = 0; i < P.length; i += vd) {
        const X = 3 + k * (P[i] - x0) + (W - 6 - k * (x1 - x0)) / 2, Y = W - 3 - k * (P[i + 1] - y0) - (W - 6 - k * (y1 - y0)) / 2;
        if (i) ctx.lineTo(X, Y); else ctx.moveTo(X, Y);
      }
      if (o.closed) ctx.closePath();
      ctx.stroke();
    });
    box.querySelectorAll('tr[data-s]').forEach((tr) => { tr.onclick = () => animateTo(+tr.dataset.s); });
  }

  /** Keep the log in step with training (at most twice a second, only while it is visible). */
  function updateObjLog(now) {
    if (!isOpen('objlog') || now - logTimer < 500) return;
    const key = `${logObj}|${netVersion}|${trainer ? trainer.step_ : 0}|${S.objects.length}`;
    if (key === logKey) return;
    logKey = key; logTimer = now;
    try { renderObjLog(); } catch (err) { console.error(err); }
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
    S.train.goals = isGoals() ? defaultGoals() : [];          // targets have the old dimension
    tool = 'pan'; t = 0; anim = null; playing = false; training = false;
    $('c2d').classList.toggle('hidden', d !== 2);
    $('c3d').classList.toggle('hidden', d !== 3);
    if (d === 3 && !viz3) {
      try { viz3 = new Viz3D($('c3d')); bind3DMouse(); }
      catch (err) { alert('3D view needs WebGL: ' + err.message); return setDim(2); }
    }
    if (viz3) viz3.resize();
    if (d === 2) viz2.resize();      // the view may have changed size (drawer, panels) while the 2D canvas was hidden
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
    goalNextId = 1 + Math.max(0, ...(S.train.goals || []).map((g) => g.id || 0));
    if (S.dim !== prevDim) {
      $('c2d').classList.toggle('hidden', S.dim !== 2);
      $('c3d').classList.toggle('hidden', S.dim !== 3);
      if (S.dim === 3 && !viz3) { viz3 = new Viz3D($('c3d')); bind3DMouse(); }
      if (S.dim === 2) viz2.resize(); else if (viz3) viz3.resize();
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

  /*
   * Draw only when something changed: the stage, the weights, the view, the camera — or for a moment after
   * any pointer / key / input event (hover effects, toggles, drags).  Nothing is drawn behind the guided
   * start.  An idle page then costs almost nothing.
   */
  let lastSig = '', activeUntil = 0;
  const poke = () => { activeUntil = performance.now() + 400; };
  for (const ev of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'keydown', 'input', 'change', 'click', 'resize']) {
    (ev === 'resize' ? window : document).addEventListener(ev, poke, { capture: true, passive: true });
  }
  function viewSig() {
    return [t.toFixed(4), netVersion, trainer ? trainer.step_ : 0, traces ? 1 : 0, viz2.cx, viz2.cy, viz2.scale, viz2.w, viz2.h,
      cam.yaw, cam.pitch, tool, drawing ? 1 : 0, S.dim].join('|');
  }

  function frame(now) {
    advanceAnimation(now);
    if (training) trainSteps(S.train.stepsPerFrame);
    if (!document.body.classList.contains('wizard-open')) {
      const sig = viewSig();
      if (sig !== lastSig || now < activeUntil || anim || playing) {
        try { render(); } catch (err) { console.error(err); }
        lastSig = viewSig();
      }
    }
    if (now - lastLossDraw > 120 && training) { if (isOpen('loss')) drawLoss(); lastLossDraw = now; }
    if (now - lastInspector > (training ? 120 : 60)) { lastInspector = now; try { renderInspectorAndNet(); } catch (err) { console.error(err); } }
    renderSummaries();
    updateObjLog(now);
    renderMats(false);
    updateGoalLosses(false);
    if (analysisDirty && now - lastAnalysis > (training ? 700 : 150)) {
      analysisDirty = false; lastAnalysis = now;
      try { updateAnalysis(); } catch (err) { console.error(err); }
    }
    requestAnimationFrame(frame);
  }

  // ===========================================================================
  // Layer inspector, network diagram, weight history
  // ===========================================================================

  const inspect = { layer: 1, hover: null, cells: [], netHits: [], key: '' };
  let history = [], lastInspector = 0;

  /** Weight snapshots for the history chart (decimated to at most ~400 points). */
  function resetHistory() {
    history = net ? [{ step: trainer ? trainer.step_ : 0, theta: Float64Array.from(net.theta) }] : [];
    liveTheta = null; viewIdx = null;
  }

  /*
   * Training timeline: the snapshots in `history` are frames of a video.  showHistory(i) puts frame i's
   * weights into the network (the live weights are kept aside); backToLive() restores them.  Training or
   * stepping from an old frame continues from there: later frames are dropped, like recording over a tape.
   */
  let liveTheta = null, viewIdx = null;
  function showHistory(i) {
    if (history.length < 2) return;
    i = Math.max(0, Math.min(history.length - 1, Math.round(i)));
    if (liveTheta === null) liveTheta = Float64Array.from(net.theta);
    training = false;
    viewIdx = i;
    net.theta.set(history[i].theta);
    netVersion++; invalidate(); inspect.key = '';
    updateTrainStatus(`watching step ${history[i].step} of ${history[history.length - 1].step} · ● live returns · ▶ Train continues from here`);
  }
  function backToLive() {
    if (liveTheta === null) return;
    net.theta.set(liveTheta);
    liveTheta = null; viewIdx = null;
    netVersion++; invalidate(); inspect.key = '';
    updateTrainStatus();
  }
  /** Continue training from the frame being watched. */
  function branchFromView() {
    if (viewIdx === null) return;
    const step = history[viewIdx].step;
    history = history.slice(0, viewIdx + 1);
    trainer.lossHistory.length = Math.max(0, Math.min(trainer.lossHistory.length, trainer.lossHistory.length - (trainer.step_ - step)));
    trainer.step_ = step;
    const o = trainer.opts;                                  // fresh optimizer state for the old weights
    trainer.optim = o.optimizer === 'sgd' ? new NN.SGD(net.nParams, o.lr) : new NN.Adam(net.nParams, o.lr);
    trainer.done = false; trainer.diverged = false;
    liveTheta = null; viewIdx = null;
  }
  function pushHistory() {
    if (!net) return;
    const step = trainer ? trainer.step_ : 0;
    if (history.length && history[0].theta.length !== net.nParams) resetHistory();
    const last = history[history.length - 1];
    if (last && last.step === step && last.theta.every((v, k) => v === net.theta[k])) return;
    history.push({ step, theta: Float64Array.from(net.theta) });
    if (history.length > 400) history = history.filter((_, k) => k === 0 || k % 2 === 1 || k === history.length - 1);
  }

  /** Pre-activations z of layer l (1-based) for the points that are drawn (from the cached traces). */
  function layerZs(l) {
    if (!traces) return [];
    const out = [], s = 2 * l - 1;
    for (const d of traces.ds) {
      if (d.role === 'jac' || d.role === 'probe' || d.role === 'interf') continue;
      const k = Math.max(1, Math.floor(d.st.length / 400));
      for (let i = 0; i < d.st.length; i += k) for (const v of d.st[i][s]) out.push(v);
      if (out.length > 8000) break;
    }
    return out;
  }

  /*
   * "One point through this layer": a_in → [W | b] → z = W·a_in + b → σ → a_out, for the origin or a basis
   * vector (chosen above).  In layer 1 the input is e.g. e₁ = (1, 0) itself; in later layers it is where
   * that point has arrived.  Hovering an entry of z shows its whole sum.
   */
  let exSel = 1;
  function renderInspExample(l) {
    const box = $('inspExample'), d = S.dim, Lay = net.layout[l - 1], act = net.acts[l - 1], th = net.theta;
    const names = ['o', ...[...Array(d)].map((_, i) => `e${subN(i + 1)}`)], colors = ['#e6edf3', ...BASIS_COLORS];
    exSel = Math.min(exSel, d);
    const p = new Float64Array(d); if (exSel > 0) p[exSel - 1] = 1;
    const st = net.stages(p), aIn = st[2 * l - 2], z = st[2 * l - 1], aOut = st[2 * l];
    let mx = 1e-9;
    for (const v of th) mx = Math.max(mx, Math.abs(v));
    const n2 = (v) => (Math.abs(v) < 0.005 ? '0.00' : v.toFixed(2)).replace('-', '−');
    const cell = (v, scale = mx) => {
      const a = (0.1 + 0.6 * Math.min(1, Math.abs(v) / scale)).toFixed(3);
      return `<span style="background:${v >= 0 ? `rgba(227,179,65,${a})` : `rgba(188,140,255,${a})`}">${n2(v)}</span>`;
    };
    const vec = (v, label, cls = '', titles = null) => `<div class="ex-col"><span class="lbl">${label}</span><div class="ex-vec ${cls}">${
      Array.from(v).map((x, i) => (titles ? cell(x, Math.max(1, ...Array.from(v).map(Math.abs))).replace('<span', `<span title="${titles[i]}"`) : cell(x, Math.max(1, ...Array.from(v).map(Math.abs))))).join('')}</div></div>`;
    let W = '';
    for (let i = 0; i < Lay.nout; i++) for (let j = 0; j < Lay.nin; j++) W += cell(th[Lay.w + i * Lay.nin + j]);
    const bv = Array.from(th.subarray(Lay.b, Lay.b + Lay.nout));
    const sums = [...Array(Lay.nout)].map((_, i) => {
      const terms = [...Array(Lay.nin)].map((__, j) => `(${n2(th[Lay.w + i * Lay.nin + j])})(${n2(aIn[j])})`).join(' + ');
      return `z${subN(i + 1)} = ${terms} + (${n2(bv[i])}) = ${n2(z[i])}`;
    });
    const inName = l === 1 ? `x = ${names[exSel]}` : `a${subN(l - 1)}`, actName = act === 'identity' ? 'linear' : act;
    const pIn = `(${Array.from(p).map((v) => v.toFixed(0)).join(', ')})`;
    box.innerHTML = `<div class="ex-pick">Follow ${names.map((nm, i) => `<button data-ex="${i}" class="${i === exSel ? 'on' : ''}" style="color:${colors[i]}">${nm}</button>`).join('')}</div>
      <div class="ex-note">${names[exSel]} = ${pIn} enters the network${l === 1 ? ' here.' : `; layer ${l} receives it as a${subN(l - 1)}, after ${l - 1} layer${l > 2 ? 's' : ''}.`}
        Hover a value of z to see its sum.</div>
      <div class="ex-flow">
        ${vec(aIn, `in: ${inName}`)}
        <div class="ex-op">→<small>W·a + b</small></div>
        <div class="ex-col"><span class="lbl">W${subN(l)} (${Lay.nout}×${Lay.nin})</span><div class="ex-mat" style="grid-template-columns: repeat(${Lay.nin}, auto)">${W}</div></div>
        <div class="ex-col"><span class="lbl">b${subN(l)}</span><div class="ex-vec">${bv.map((v) => cell(v)).join('')}</div></div>
        <div class="ex-op">=</div>
        ${vec(z, 'z (linear)', '', sums)}
        <div class="ex-op">→<small>${actName}</small></div>
        ${vec(aOut, `out: a${subN(l)}${l === net.nLayers ? ' = f(x)' : ''}`, 'out')}
      </div>`;
    box.querySelectorAll('[data-ex]').forEach((b) => { b.onclick = () => { exSel = +b.dataset.ex; renderInspExample(l); }; });
  }

  function renderInspectorAndNet() {
    if (!net || !traces) return;
    const L = net.nLayers;
    inspect.layer = Math.min(Math.max(1, inspect.layer), L);
    const key = `${netVersion}|${trainer.step_}|${inspect.layer}|${Math.round(t * 100)}|${JSON.stringify(inspect.hover)}|${history.length}|${S.probe.join()}`;
    if (key === inspect.key) return;
    inspect.key = key;
    const current = t > 1e-6 ? Math.max(1, Math.ceil(t / 2 - 1e-6)) : 0;   // layer the view is showing (0 = input)
    if (!$('netStrip').classList.contains('hidden')) {
      const { as } = net.forward(Float64Array.from(S.probe));
      inspect.netHits = Inspector.drawNetwork($('netCanvas'), { net, values: as, selected: inspect.layer, current });
    }
    if (!isOpen('inspector')) return;
    const l = inspect.layer, Lay = net.layout[l - 1], act = net.acts[l - 1];
    const W = net.weightMatrix(l - 1), b = Array.from(net.theta.subarray(Lay.b, Lay.b + Lay.nout));
    const h0 = history.length && history[0].theta.length === net.nParams ? history[0].theta : null;
    const W0 = h0 ? W.map((_, i) => Array.from(h0.subarray(Lay.w + i * Lay.nin, Lay.w + (i + 1) * Lay.nin))) : null;
    $('inspTitle').innerHTML = `Layer ${l} of ${L}<span class="tag">${Lay.nin} → ${Lay.nout} · ${act}${l === L ? ' · output' : ''}</span>`;
    $('inspPrev').disabled = l <= 1; $('inspNext').disabled = l >= L;
    renderInspExample(l);
    inspect.cells = Inspector.drawWeights($('wCanvas'), W, b, { W0, hover: inspect.hover });
    // numbers
    const sv = NN.singularValues(W), rank = NN.rankOf(sv);
    let fro = 0, dfro = 0;
    W.forEach((r, i) => r.forEach((v, j) => { fro += v * v; if (W0) dfro += (v - W0[i][j]) ** 2; }));
    const rows = [['singular values', sv.slice(0, 4).map((v) => v.toFixed(3)).join(', ') + (sv.length > 4 ? ' …' : '')],
      ['rank', `${rank} of ${Math.min(Lay.nin, Lay.nout)}${rank < Math.min(Lay.nin, Lay.nout) ? ' — squashes a dimension' : ''}`]];
    if (Lay.nin === Lay.nout) { const dt = NN.det(W); rows.push(['det W', `${fmt(dt)}${dt < 0 ? ' (mirrors space)' : ''}`]); }
    rows.push(['‖W‖ (Frobenius)', fmt(Math.sqrt(fro))]);
    if (W0) rows.push(['change since start', `‖W − W₀‖ = ${fmt(Math.sqrt(dfro))} (bars under the cells)`]);
    $('wStats').innerHTML = rows.map(([k, v]) => `<span>${k}</span><span>${v}</span>`).join('');
    // activation + where the points are
    const zs = layerZs(l);
    Inspector.drawActivation($('actCanvas'), act, zs);
    let note = '';
    if (zs.length) {
      const frac = (f) => `${(100 * zs.filter(f).length / zs.length).toFixed(0)}%`;
      note = act === 'relu' ? `${frac((z) => z <= 0)} of the values are negative → set to 0 (that part of space is flattened).`
        : act === 'tanh' ? `${frac((z) => Math.abs(z) > 2.5)} of the values are in the flat tails (|z| > 2.5) where tanh barely changes.`
        : act === 'sigmoid' ? `${frac((z) => Math.abs(z) > 4)} of the values are in the flat tails (|z| > 4).`
        : act === 'sin' ? `${frac((z) => Math.abs(z) > Math.PI / 2)} of the values are beyond ±π/2, where sin turns back (folds).`
        : act === 'gelu' ? `${frac((z) => z < -0.75)} of the values are below −0.75, where GELU turns back up (folds).`
        : 'Identity: this step changes nothing — the layer is purely linear.';
    }
    $('actNote').textContent = (zs.length ? 'Now: ' : '') + note;
    Inspector.drawHistory($('histCanvas'), history, Lay);
    $('histNote').textContent = history.length > 1
      ? (history[0].step === history[history.length - 1].step
        ? `${history.length - 1} hand edit(s), no training yet. Solid = weights (colour = output neuron), dashed = biases.`
        : `${history.length} snapshots from step ${history[0].step} to ${history[history.length - 1].step}. Solid = weights (colour = output neuron), dashed = biases.`)
      : '';
  }

  function selectLayer(l, jump) {
    inspect.layer = Math.min(Math.max(1, l), net.nLayers);
    inspect.hover = null;
    document.dispatchEvent(new CustomEvent('nsd:layer', { detail: inspect.layer }));   // the shell opens the inspector
    if (jump) animateTo(2 * inspect.layer);
    inspect.key = '';
  }

  function setupInspector() {
    $('inspPrev').onclick = () => selectLayer(inspect.layer - 1, false);
    $('inspNext').onclick = () => selectLayer(inspect.layer + 1, false);
    $('inspShowLin').onclick = () => animateTo(2 * inspect.layer - 1);
    $('inspShowAct').onclick = () => animateTo(2 * inspect.layer);
    const wc = $('wCanvas'), tip = $('tip');
    const cellAt = (e) => {
      const r = wc.getBoundingClientRect(), x = e.clientX - r.left, y = e.clientY - r.top;
      return inspect.cells.find((c) => x >= c.x && x <= c.x + c.w && y >= c.y && y <= c.y + c.h) || null;
    };
    wc.addEventListener('mousemove', (e) => {
      const c = cellAt(e);
      const hv = c ? { kind: c.kind, i: c.i, j: c.j } : null;
      if (JSON.stringify(hv) !== JSON.stringify(inspect.hover)) { inspect.hover = hv; inspect.key = ''; }
      if (!c) { tip.classList.add('hidden'); return; }
      const Lay = net.layout[inspect.layer - 1];
      const v = c.kind === 'W' ? net.theta[Lay.w + c.i * Lay.nin + c.j] : net.theta[Lay.b + c.i];
      tip.textContent = c.kind === 'W'
        ? `W[${c.i + 1}][${c.j + 1}] = ${v.toFixed(4)} — how much input a${c.j + 1} pushes neuron z${c.i + 1}. Double-click to edit.`
        : `b[${c.i + 1}] = ${v.toFixed(4)} — bias (constant shift) of neuron z${c.i + 1}. Double-click to edit.`;
      tip.classList.remove('hidden');
      tip.style.left = Math.min(window.innerWidth - tip.offsetWidth - 8, e.clientX + 14) + 'px';
      tip.style.top = (e.clientY + 18) + 'px';
    });
    wc.addEventListener('mouseleave', () => { inspect.hover = null; inspect.key = ''; tip.classList.add('hidden'); });
    wc.addEventListener('dblclick', (e) => {
      const c = cellAt(e);
      if (!c) return;
      const Lay = net.layout[inspect.layer - 1], off = c.kind === 'W' ? Lay.w + c.i * Lay.nin + c.j : Lay.b + c.i;
      const r = wc.getBoundingClientRect();
      const inp = document.createElement('input');
      inp.className = 'cell-edit'; inp.type = 'text'; inp.value = net.theta[off].toFixed(4);
      inp.style.left = (r.left + c.x) + 'px'; inp.style.top = (r.top + c.y) + 'px';
      document.body.appendChild(inp); inp.focus(); inp.select();
      let done = false;
      const finish = (apply) => {
        if (done) return; done = true;
        const v = parseFloat(inp.value.replace(',', '.'));
        inp.remove();
        if (!apply || !Number.isFinite(v)) return;
        training = false;
        net.theta[off] = v;
        if (S.net.homeo) NN.clampSingularValues(net, HOMEO_FLOOR);
        netVersion++; invalidate(); pushHistory(); inspect.key = '';
        flashHint(`${c.kind === 'W' ? `W[${c.i + 1}][${c.j + 1}]` : `b[${c.i + 1}]`} of layer ${inspect.layer} set to ${v} — the deformation is redrawn.`);
      };
      inp.addEventListener('keydown', (k) => { if (k.key === 'Enter') finish(true); if (k.key === 'Escape') finish(false); });
      inp.addEventListener('blur', () => finish(true));
    });
    // network diagram
    $('netCanvas').addEventListener('click', (e) => {
      const r = $('netCanvas').getBoundingClientRect(), x = e.clientX - r.left;
      const h = inspect.netHits.find((z) => x >= z.x0 && x <= z.x1);
      if (h) selectLayer(h.layer, true);
    });
    $('netToggle').onclick = () => {
      const strip = $('netStrip'), show = strip.classList.contains('hidden');
      strip.classList.toggle('hidden', !show);
      $('netToggle').classList.toggle('on', show);
      store('nsd.netStrip', show ? '1' : '0');
      inspect.key = ''; onResize();
    };
    if (load('nsd.netStrip') === '1') { $('netStrip').classList.remove('hidden'); $('netToggle').classList.add('on'); }
  }

  // ===========================================================================
  // Collapsible sections, header summaries, keyboard shortcuts
  // ===========================================================================

  function store(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* storage may be unavailable */ } }
  function load(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }

  /** Remember which sections are open; redraw charts when a section opens (closed canvases have no size). */
  function setupSections() {
    for (const d of document.querySelectorAll('details[data-sec]')) {
      const saved = load(`nsd.sec.${d.dataset.sec}`);
      if (saved !== null) d.open = saved === '1';
      d.addEventListener('toggle', () => {
        store(`nsd.sec.${d.dataset.sec}`, d.open ? '1' : '0');
        if (d.open) { analysisDirty = true; inspect.key = ''; lastAnalysis = 0; }
      });
    }
  }

  const TASK_NAMES = { goals: 'my goals', none: 'no training', anchors: 'classes → points', classify: 'classify', transform: 'linear map', morph: 'shape → shape', pins: 'pins' };
  function setText(id, text) { const el = $(id); if (el && el.textContent !== text) el.textContent = text; }
  function renderSummaries() {
    if (!net) return;
    setText('sumNetwork', `${net.nLayers} layer${net.nLayers > 1 ? 's' : ''} · width ${S.net.width} · ${S.net.temperature > 0 ? 'mixed' : S.net.defaultAct}${S.net.homeo ? ' · invertible' : ''}`);
    setText('sumObjects', `${S.objects.length} object${S.objects.length === 1 ? '' : 's'}${S.pins.length ? ` · ${S.pins.length} pins` : ''}`);
    const D = S.display, ov = [D.jac && 'Jacobian', D.det && 'det', D.dist && 'distance', D.autoFit && 'auto-fit'].filter(Boolean);
    setText('sumDisplay', ov.length ? ov.join(' · ') : 'grid, circle, basis');
    const T = trainer, h = T ? T.lossHistory : [];
    setText('sumTraining', `${TASK_NAMES[S.train.target]}${training ? ' · training…' : T && T.step_ ? ` · step ${T.step_}` : ''}`);
    setText('sumLoss', h.length ? fmt(h[h.length - 1]) : '');
    setText('sumInspector', `layer ${inspect.layer} · ${net.acts[inspect.layer - 1] || ''}`);
  }

  function setupHelp() {
    $('helpBtn').onclick = (e) => { e.stopPropagation(); $('helpPanel').classList.toggle('hidden'); };
    $('tourBtn').onclick = (e) => { e.stopPropagation(); $('helpPanel').classList.add('hidden'); if (window.Tour) window.Tour.start(); };
    document.addEventListener('click', (e) => { if (!e.target.closest('#helpPanel')) $('helpPanel').classList.add('hidden'); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') $('helpPanel').classList.add('hidden'); });
  }

  function setupShortcuts() {
    document.addEventListener('keydown', (e) => {
      if (e.key === 'Escape' && tool !== 'pan' && !document.body.classList.contains('wizard-open')) {
        setTool('pan'); e.stopImmediatePropagation(); return;
      }
      const tag = (e.target.tagName || '').toLowerCase();
      if (tag === 'input' || tag === 'select' || tag === 'textarea' || e.ctrlKey || e.metaKey || e.altKey) return;
      if (document.body.classList.contains('wizard-open') || document.body.classList.contains('tour-open')) return;
      if (e.key === ' ') { e.preventDefault(); $('play').click(); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); $('stepFwd').click(); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); $('stepBack').click(); }
      else if (e.key === 't' || e.key === 'T') $('trainPlay').click();
    });
  }

  function onResize() {
    viz2.resize();
    if (viz3) viz3.resize();
    analysisDirty = true;
    inspect.key = '';
  }

  function init() {
    setupSections();
    setupControls();
    setupTooltips();
    setupInspector();
    setupShortcuts();
    setupHelp();
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
      serializeState, get training() { return training; }, set training(v) { training = v; }, viz2,
      selectLayer, get inspect() { return inspect; }, get history() { return history; }, applyGuided, recommendedArch, initFor,
      animateTo, stageName, scrubTo, get playing() { return playing; }, showHistory, backToLive,
      get viewIdx() { return viewIdx; }, get historySteps() { return history.map((h) => h.step); }, restOfNetwork, computeTraces, get traces() { return traces; } };
  }

  init();
})();
