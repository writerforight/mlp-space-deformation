/* Small looping previews for the guided start (cards on the dimension and problem steps).

   Every preview shows the same thing the workspace shows, in miniature: a set of points and grid lines
   in input space, and where a small network sends them.  The card animates input -> output -> input.

     Minis.scene(problem, dim)  ->  { items, boundary, ok }        (pure; uses NN, runs in node too)
     Minis.sceneAsync(problem, dim, done)                            (same, built in slices between frames)
     Minis.player(canvas, scene) ->  { draw(timeSeconds) }          (browser only)

   items: [{ kind: 'line' | 'dots', from: [Float64Array], to: [Float64Array], color, alpha, width }]
   For training problems a tiny network is trained here, with a fixed seed, so every visit looks the same.
*/
(function (root) {
  'use strict';
  const NN = root.NN || (typeof require !== 'undefined' ? require('./nn.js') : null);

  const CLASS = ['#58a6ff', '#ff6b6b'];        // the two classes (blue / red are reserved for them)
  const GRID = '#8b949e';
  const OBJ = '#e3b341';                       // a shape that is not a class
  const TARGET = '#e6edf3';

  // ---- geometry helpers ---------------------------------------------------------------------------
  /** Grid lines over [-e, e]^dim: 2D a 9 x 9 grid, 3D the edges of a 4 x 4 x 4 lattice. */
  function gridLines(dim, e = 1.5, n = dim === 2 ? 9 : 4, samples = 32) {
    const lines = [], ticks = [...Array(n)].map((_, i) => -e + (2 * e * i) / (n - 1));
    const along = (fix, axis) => [...Array(samples)].map((_, k) => {
      const p = new Float64Array(dim), t = -e + (2 * e * k) / (samples - 1);
      let j = 0;
      for (let a = 0; a < dim; a++) p[a] = a === axis ? t : fix[j++];
      return p;
    });
    if (dim === 2) for (const a of [0, 1]) for (const c of ticks) lines.push(along([c], a));
    else for (const a of [0, 1, 2]) for (const c1 of ticks) for (const c2 of ticks) lines.push(along([c1, c2], a));
    return lines;
  }

  /** Train in slices: yields every 50 steps so the browser can keep animating in between. */
  function* train(net, data, type, steps, seed, lr = 0.01) {
    const T = new NN.Trainer(net, { type, lr, batch: 32, seed });
    T.setTasks([data]);
    for (let i = 0; i < steps && !T.diverged; i++) {
      T.step();
      if (i % 50 === 49) yield;
    }
  }

  function buildNet(dim, layers, width, seed, outputLinear, dist = 'xavier', scale = 1) {
    const dims = [dim, ...Array(layers - 1).fill(width), dim];
    const acts = dims.slice(1).map((_, l) => (outputLinear && l === layers - 1 ? 'identity' : 'tanh'));
    return new NN.MLP(dims, acts).init(dist, scale, new NN.Rng(seed));
  }

  const lineItems = (lines, net, style) => lines.map((pts) => Object.assign({
    kind: 'line', from: pts, to: pts.map((p) => net.predict(p)) }, style));
  const gridItems = (lines, net, style) => lineItems(lines, net, style).map((it) => Object.assign(it, { grid: true }));

  // ---- the scenes -----------------------------------------------------------------------------------
  function* sceneSteps(problem, dim) {
    const rng = new NN.Rng(dim * 100 + 7);
    const grid = gridLines(dim);
    const gridStyle = { color: GRID, alpha: dim === 2 ? 0.45 : 0.3, width: 1 };

    if (problem === 'classify') {
      // 2D: two moons; 3D: a ball inside a shell (the 3D "disk inside a ring")
      const data = NN.makeDataset(dim === 2 ? 'moons' : 'circles', 240, dim, rng, 0.06);
      const net = buildNet(dim, 3, dim === 2 ? 4 : 6, 11, true);
      yield* train(net, data, 'ce', 1000, 3);
      const acc = data.filter((s) => { const y = net.predict(s.x); return (y[1] > y[0] ? 1 : 0) === s.y; }).length / data.length;
      return {
        items: [...gridItems(grid, net, gridStyle),
          ...[0, 1].map((c) => {
            const pts = data.filter((s) => s.y === c).map((s) => s.x);
            return { kind: 'dots', from: pts, to: pts.map((p) => net.predict(p)), color: CLASS[c], alpha: 1 };
          })],
        boundary: dim === 2,              // the decision line logit_1 = logit_2 (fades in at the output)
        ok: acc, fitTo: 'objects',
      };
    }

    if (problem === 'transform') {
      // a rotation by 50 degrees and a mild shear, learned from samples
      const A = NN.targetMatrix('rotation', dim, (50 * Math.PI) / 180);
      A[0][1] += 0.35;
      const data = NN.makeTransformData(A, 300, dim, rng);
      const net = buildNet(dim, 2, 4, 5, true);
      yield* train(net, data, 'mse', 1500, 2);
      const err = NN.meanLoss(net, data, 'mse');
      // an "F"-like marker so rotation and shear are visible (a circle would hide the rotation)
      const F = [[[-0.5, -0.8], [-0.5, 0.8], [0.5, 0.8]], [[-0.5, 0.05], [0.25, 0.05]]]
        .map((poly) => densify(poly.map((p) => (dim === 2 ? p : [p[0], 0, p[1]]))));   // 3D: upright, facing the camera
      return {
        items: [...lineItems(grid, net, gridStyle),
          ...lineItems(F, net, { color: OBJ, alpha: 1, width: 2.5 })],
        ok: 1 - err, fitTo: 'all',
      };
    }

    if (problem === 'morph') {
      const [from, to] = dim === 2 ? ['circle', 'star'] : ['sphere', 'cube'];
      const data = NN.morphData(from, to, 300);
      // a five-pointed star needs more room than the recommended width d + 1 to fit its points sharply
      const net = buildNet(dim, 4, dim === 2 ? 8 : dim + 1, 9, true);
      yield* train(net, data, 'mse', dim === 2 ? 2000 : 1500, 4, dim === 2 ? 0.02 : 0.01);
      const err = NN.meanLoss(net, data, 'mse');
      const src = NN.makeShape(from, dim === 2 ? 240 : 500).components;
      const tgt = NN.makeShape(to, dim === 2 ? 240 : 500).components;
      const shapeItem = (c, color, alpha, width) => (dim === 2
        ? { kind: 'line', closed: true, from: c.pts, to: c.pts.map((p) => net.predict(p)), color, alpha, width }
        : { kind: 'dots', from: c.pts, to: c.pts.map((p) => net.predict(p)), color, alpha, size: 1.3 });
      return {
        items: [...gridItems(grid, net, gridStyle),
          ...tgt.map((c) => ({ kind: dim === 2 ? 'line' : 'dots', closed: true, from: c.pts, to: c.pts, color: TARGET,
            alpha: 0.35, width: 1.2, size: 0.9, dash: [4, 4], showAt: 'out' })),
          ...src.map((c) => shapeItem(c, OBJ, 1, 2.5))],
        ok: 1 - err, fitTo: 'objects',
      };
    }

    // 'none' (just look): an untrained network with strong weights bends the grid.  The seeds are picked so
    // the image keeps some area (many random tanh nets squash the plane onto a thin curve).
    const net = buildNet(dim, 3, dim, dim === 2 ? 2 : 13, false, 'he', 1.6);
    const circle = [...Array(120)].map((_, i) => {
      const t = (2 * Math.PI * i) / 120;
      return Float64Array.from(dim === 2 ? [Math.cos(t), Math.sin(t)] : [Math.cos(t), Math.sin(t), 0]);
    });
    return {
      items: [...lineItems(grid, net, { ...gridStyle, alpha: dim === 2 ? 0.6 : 0.4 }),
        { kind: 'line', closed: true, from: circle, to: circle.map((p) => net.predict(p)), color: OBJ, alpha: 1, width: 2 }],
      ok: 1, fitTo: 'all',
    };
  }

  /** The whole scene at once (node, tests). */
  function scene(problem, dim) {
    const g = sceneSteps(problem, dim);
    for (;;) { const r = g.next(); if (r.done) return r.value; }
  }

  /** Browser: build the scene in slices of ~6 ms per animation frame; calls done(scene) at the end. */
  function sceneAsync(problem, dim, done) {
    const g = sceneSteps(problem, dim);
    let cancelled = false;
    (function pump() {
      if (cancelled) return;
      const t0 = performance.now();
      while (performance.now() - t0 < 6) {
        const r = g.next();
        if (r.done) { done(r.value); return; }
      }
      requestAnimationFrame(pump);
    })();
    return () => { cancelled = true; };
  }

  /** Polyline -> many points, so a curved image of a straight segment stays smooth. */
  function densify(poly, per = 24) {
    const out = [];
    for (let i = 0; i < poly.length - 1; i++) {
      for (let k = 0; k < per; k++) {
        const t = k / per;
        out.push(Float64Array.from(poly[i].map((v, j) => v + t * (poly[i + 1][j] - v))));
      }
    }
    out.push(Float64Array.from(poly[poly.length - 1]));
    return out;
  }


  // ---- data previews (no network): the data set itself, or the target map / shape pair ---------------
  /** Data sets offered on the data step, per problem.  `noise`: the generator uses the noise setting. */
  const DATASETS = [
    { id: 'blobs', name: 'Two blobs', group: 'Easy', dims: [2, 3], blurb: 'Two separate clouds. A straight line already splits them.' },
    { id: 'moons', name: 'Two moons', group: 'Easy', dims: [2, 3], noise: true, blurb: 'Two interleaving half-moons. One bend is enough.' },
    { id: 'circles', name: 'Disk in a ring', group: 'Easy', dims: [2, 3], noise: true, blurb: 'A disk inside a ring (3D: a ball inside a shell). No line separates them; the network has to squash space or use an extra dimension.' },
    { id: 'xor', name: 'XOR', group: 'Harder', dims: [2, 3], blurb: 'Opposite quadrants share a colour. No single line works: the network must fold space.' },
    { id: 'wave', name: 'Wavy boundary', group: 'Harder', dims: [2, 3], blurb: 'A smooth but curved boundary.' },
    { id: 'rings', name: 'Three rings', group: 'Harder', dims: [2, 3], noise: true, blurb: 'Blue, red, blue rings. Separating the middle ring takes two folds.' },
    { id: 'spirals', name: 'Spirals', group: 'Hard', dims: [2, 3], noise: true, blurb: 'Two interleaved spirals. Many folds are needed: use more layers or width.' },
    { id: 'checker', name: 'Checkerboard', group: 'Hard', dims: [2, 3], blurb: 'Many small regions: a test of capacity (width x depth).' },
    { id: 'linked', name: 'Linked rings', group: 'Hard', dims: [3], noise: true, blurb: 'Two linked rings. No smooth invertible deformation of space can unlink them; width 4 or more separates them.' },
    { id: 'shape:diskInRing', name: 'Exact disk in a ring', group: 'Exact shapes', dims: [2], exact: true, blurb: 'The disk and ring without noise. A homeomorphism of the plane cannot move the disk out.' },
    { id: 'shape:linkedRings', name: 'Exact linked rings', group: 'Exact shapes', dims: [3], exact: true, blurb: 'Two exact linked rings. No homeomorphism of 3D space unlinks them.' },
    { id: 'shape:nestedSpheres', name: 'Ball in a shell', group: 'Exact shapes', dims: [3], exact: true, blurb: 'An exact ball inside a spherical shell. No homeomorphism of 3D space gets the ball out.' },
  ];
  const MAPS = [
    { id: 'rotation', name: 'Rotation', amount: 0.9, min: -3.14, max: 3.14, unit: 'rad', blurb: 'Turn space around the origin by an angle.' },
    { id: 'shear', name: 'Shear', amount: 0.8, min: -2, max: 2, unit: '', blurb: 'Slide each row sideways in proportion to its height.' },
    { id: 'scaling', name: 'Scaling', amount: 1.6, min: 0.3, max: 3, unit: 'x', blurb: 'Stretch one axis and squeeze the other (area is kept).' },
  ];
  const PAIRS = {
    2: [['circle', 'square'], ['circle', 'star'], ['circle', 'figure8'], ['twoCircles', 'oneCircleTwice']],
    3: [['sphere', 'ellipsoid'], ['sphere', 'cube'], ['unknot', 'trefoil'], ['torus', 'torusOnSphere']],
  };
  const PAIR_BLURBS = {
    'circle>square': 'Easy: only corners have to appear.',
    'circle>star': 'Five points to push out: needs a few more neurons.',
    'circle>figure8': 'The curve must touch itself. An invertible map cannot do that; an extra dimension can.',
    'twoCircles>oneCircleTwice': 'Two circles must land on the same circle: two parts glued together.',
    'sphere>ellipsoid': 'A smooth stretch: the easy 3D case.',
    'sphere>cube': 'Corners and edges have to appear on a smooth surface.',
    'unknot>trefoil': 'Tie a knot into a ring. Space itself has to be twisted.',
    'torus>torusOnSphere': 'Close the hole of a donut: points must be glued together.',
  };

  /**
   * opts: classify { dataset, n, noise, seed (0, 1, 2, … = sample number) } | transform { map, amount } | morph { pair: 'a>b' }
   * The classify scene stands still (from = to); transform and morph animate identity -> target.
   */
  function dataScene(problem, dim, opts) {
    if (problem === 'classify') {
      const ds = opts.dataset;
      let samples;
      if (ds.startsWith('shape:')) {
        samples = NN.makeShape(ds.slice(6), opts.n || 400).components.flatMap((c) => c.pts.map((x) => ({ x, y: c.label })));
      } else {
        // same random stream as the workspace's buildData() with its default weight seed 2 (2·101 + 5 = 207),
        // so the sample shown here is the sample the workspace trains on; `seed` counts "new random sample" clicks
        samples = NN.makeDataset(ds, opts.n || 400, dim, new NN.Rng(207 + 7919 * (opts.seed || 0)), opts.noise ?? 0.08);
      }
      return {
        items: [0, 1].map((c) => {
          const pts = samples.filter((q) => q.y === c).map((q) => q.x);
          return { kind: 'dots', from: pts, to: pts, color: CLASS[c], alpha: 0.95, size: opts.dotSize || 1.8 };
        }),
        fitTo: 'all', still: dim === 2,
      };
    }
    if (problem === 'transform') {
      const A = NN.targetMatrix(opts.map, dim, opts.amount);
      const map = (p) => NN.matVec(A, p);
      const grid = gridLines(dim, 1.2, dim === 2 ? 7 : 3, 20);
      const F = [[[-0.5, -0.8], [-0.5, 0.8], [0.5, 0.8]], [[-0.5, 0.05], [0.25, 0.05]]]
        .map((poly) => densify(poly.map((p) => (dim === 2 ? p : [p[0], 0, p[1]]))));
      const mk = (lines, style) => lines.map((pts) => Object.assign({ kind: 'line', from: pts, to: pts.map(map) }, style));
      return {
        items: [...mk(grid, { color: GRID, alpha: dim === 2 ? 0.5 : 0.3, width: 1 }), ...mk(F, { color: OBJ, alpha: 1, width: 2.5 })],
        fitTo: 'all', stableView: true,
      };
    }
    // morph: point i of the first shape travels to point i of the second
    const [from, to] = opts.pair.split('>');
    const n = dim === 2 ? 240 : 500;
    const A = NN.makeShape(from, n).components, B = NN.makeShape(to, n).components;
    const kind = dim === 2 ? 'line' : 'dots';
    return {
      items: [
        ...B.map((c) => ({ kind, closed: c.closed, from: c.pts, to: c.pts, color: TARGET, alpha: 0.35, width: 1.2, size: 0.9, dash: [4, 4], showAt: 'out' })),
        // a shape in two labelled parts is drawn blue / red, a single shape gold
        ...A.map((c, k) => ({ kind, closed: c.closed, from: c.pts, to: B[k].pts, color: c.label === undefined ? OBJ : CLASS[c.label], alpha: 1, width: 2.2, size: 1.3 })),
      ],
      fitTo: 'all', stableView: true,
    };
  }

  // ---- drawing (browser) ------------------------------------------------------------------------------
  const HOLD_IN = 0.8, MOVE = 1.8, HOLD_OUT = 1.6, PERIOD = 2 * MOVE + HOLD_IN + HOLD_OUT;
  const ease = (u) => (u < 0.5 ? 4 * u * u * u : 1 - Math.pow(-2 * u + 2, 3) / 2);

  /** 0 = input, 1 = output, at time t (seconds) of the loop. */
  function phase(t) {
    const u = t % PERIOD;
    if (u < HOLD_IN) return 0;
    if (u < HOLD_IN + MOVE) return ease((u - HOLD_IN) / MOVE);
    if (u < HOLD_IN + MOVE + HOLD_OUT) return 1;
    return 1 - ease((u - HOLD_IN - MOVE - HOLD_OUT) / MOVE);
  }

  function bounds(sc, which, dim, project) {
    const lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
    for (const it of sc.items) {
      if (sc.fitTo === 'objects' && it.grid) continue;
      for (const p of it[which]) {
        const q = project(p);
        for (let k = 0; k < 2; k++) { lo[k] = Math.min(lo[k], q[k]); hi[k] = Math.max(hi[k], q[k]); }
      }
    }
    return { lo, hi };
  }

  function player(canvas, sc, dim) {
    const ctx = canvas.getContext('2d');
    let yaw = 0.6;
    const pitch = 0.45;
    const project = (p) => {
      if (dim === 2) return [p[0], p[1]];
      const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
      const x = cy * p[0] + sy * p[1], y = -sy * p[0] + cy * p[1];
      return [x, cp * p[2] - sp * y];
    };
    // the view box moves from the input's extent to the output's, so both ends fill the card
    let boxIn = null, boxOut = null;
    const refit = () => {
      boxIn = bounds(sc, 'from', dim, project); boxOut = bounds(sc, 'to', dim, project);
      if (sc.stableView) {                             // one box for the whole loop: the map is seen against fixed axes
        const u = { lo: [0, 1].map((k) => Math.min(boxIn.lo[k], boxOut.lo[k])), hi: [0, 1].map((k) => Math.max(boxIn.hi[k], boxOut.hi[k])) };
        boxIn = boxOut = u;
      }
    };
    if (dim === 2) refit();

    function draw(t, still) {
      const w = canvas.clientWidth, h = canvas.clientHeight, dpr = window.devicePixelRatio || 1;
      if (!w || !h) return;
      if (canvas.width !== Math.round(w * dpr)) { canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr); }
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      ctx.clearRect(0, 0, w, h);
      if (dim === 3) { yaw = 0.6 + (still ? 0 : 0.25 * t); refit(); }
      const s = still || sc.still ? 1 : phase(t);
      const lo = [0, 1].map((k) => boxIn.lo[k] + s * (boxOut.lo[k] - boxIn.lo[k]));
      const hi = [0, 1].map((k) => boxIn.hi[k] + s * (boxOut.hi[k] - boxIn.hi[k]));
      const pad = 12, span = Math.max(hi[0] - lo[0], hi[1] - lo[1], 1e-6);
      const k = (Math.min(w, h) - 2 * pad) / span;
      const cx = (lo[0] + hi[0]) / 2, cy = (lo[1] + hi[1]) / 2;
      const X = (q) => w / 2 + k * (q[0] - cx), Y = (q) => h / 2 - k * (q[1] - cy);
      const at = (it, i) => {
        const a = it.from[i], b = it.to[i];
        const p = a.length === 2 ? [a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1])]
          : [a[0] + s * (b[0] - a[0]), a[1] + s * (b[1] - a[1]), a[2] + s * (b[2] - a[2])];
        return project(p);
      };
      if (sc.boundary) {                               // logit_1 = logit_2 is the line y = x in output space
        ctx.save();
        ctx.globalAlpha = 0.6 * Math.max(0, (s - 0.7) / 0.3);
        ctx.strokeStyle = TARGET; ctx.setLineDash([5, 5]); ctx.lineWidth = 1.2;
        ctx.beginPath(); ctx.moveTo(X([lo[0] - span, lo[0] - span]), Y([lo[0] - span, lo[0] - span]));
        ctx.lineTo(X([hi[0] + span, hi[0] + span]), Y([hi[0] + span, hi[0] + span])); ctx.stroke();
        ctx.restore();
      }
      for (const it of sc.items) {
        let alpha = it.alpha;
        if (it.showAt === 'out') alpha *= Math.max(0, (s - 0.5) / 0.5);
        if (alpha <= 0) continue;
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = ctx.fillStyle = it.color;
        if (it.kind === 'line') {
          ctx.lineWidth = it.width || 1;
          ctx.setLineDash(it.dash || []);
          ctx.beginPath();
          for (let i = 0; i < it.from.length; i++) {
            const q = it.showAt === 'out' ? project(it.to[i]) : at(it, i);
            if (i) ctx.lineTo(X(q), Y(q)); else ctx.moveTo(X(q), Y(q));
          }
          if (it.closed) ctx.closePath();
          ctx.stroke();
        } else {
          const r = it.size || 1.8;
          for (let i = 0; i < it.from.length; i++) {
            const q = it.showAt === 'out' ? project(it.to[i]) : at(it, i);
            ctx.beginPath(); ctx.arc(X(q), Y(q), r, 0, 2 * Math.PI); ctx.fill();
          }
        }
      }
      ctx.globalAlpha = 1;
      ctx.setLineDash([]);
    }
    return { draw };
  }

  const Minis = { scene, sceneAsync, dataScene, player, phase, PERIOD, DATASETS, MAPS, PAIRS, PAIR_BLURBS };
  if (typeof module !== 'undefined' && module.exports) module.exports = Minis;
  else root.Minis = Minis;
})(typeof window !== 'undefined' ? window : globalThis);
