/*
 * inspector.js — the network diagram and the layer inspector.
 *
 *   drawNetwork    neurons as columns, weights as lines (gold +, violet −, thickness ∝ |w|), each neuron
 *                  coloured by its value at the probe point; returns hit regions so a click selects a layer
 *   drawWeights    the weight matrix W (rows = output neurons, columns = inputs) and bias b as a heatmap
 *                  with the numbers written in; returns cell rectangles for hover / double-click editing
 *   drawActivation σ(z) and σ'(z) with a histogram of where the drawn points actually fall
 *   drawHistory    every weight of a layer over the training steps
 *
 * Pure drawing: ui.js decides what to show and handles the interaction.
 */
(function (root) {
  'use strict';

  const C = {
    // sign colours: gold = positive, violet = negative (blue/red are reserved for the two classes)
    pos: [227, 179, 65], neg: [188, 140, 255], text: '#e6edf3', muted: '#8b949e', line: '#30363d',
    grid: 'rgba(255,255,255,0.06)', accent: '#58a6ff', panel: '#1c2128', sel: 'rgba(31,111,235,0.18)',
  };
  const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

  function prep(canvas) {
    const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(r.width * dpr)); canvas.height = Math.max(1, Math.round(r.height * dpr));
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, r.width, r.height);
    return { ctx, w: r.width, h: r.height };
  }

  /** Signed value → colour: blue (+) / red (−), opacity by magnitude relative to `scale`. */
  // one colour per output neuron in the history chart (no class blue / red)
  const ROWS = [[227, 179, 65], [188, 140, 255], [63, 185, 160], [247, 120, 186], [255, 166, 87], [165, 214, 167], [210, 168, 255], [121, 192, 180]];

  function signed(v, scale, minA = 0.12) {
    const a = Math.min(1, Math.abs(v) / (scale || 1));
    return rgba(v >= 0 ? C.pos : C.neg, minA + (1 - minA) * a);
  }

  const fmt = (v) => {
    const a = Math.abs(v);
    return a >= 100 ? v.toFixed(0) : a >= 10 ? v.toFixed(1) : v.toFixed(2);
  };

  // ---------------------------------------------------------------------------
  // Network diagram
  // ---------------------------------------------------------------------------

  /**
   * opt: { net, values: [[...] per column], selected: layer (1-based), current: layer the view is at,
   *        part: 'linear' | 'activation' | null }
   * Returns [{ layer, x0, x1 }] — the clickable band of each layer (between its input and output column).
   */
  function drawNetwork(canvas, opt) {
    const { ctx, w, h } = prep(canvas);
    const { net, values, selected, current } = opt;
    const dims = net.dims, cols = dims.length, MAXN = 16;
    const padX = 34, top = 14, bottom = 26, colX = (k) => padX + ((w - 2 * padX) * k) / Math.max(1, cols - 1);
    const shown = dims.map((d) => Math.min(d, MAXN));
    const nodeY = (k, i) => {
      const n = shown[k], span = Math.min(h - top - bottom, n * 16);
      return top + (h - top - bottom - span) / 2 + (n === 1 ? span / 2 : (span * i) / (n - 1));
    };
    const hits = [];
    // layer bands (selection / current position)
    for (let l = 1; l < cols; l++) {
      hits.push({ layer: l, x0: colX(l - 1) + 8, x1: colX(l) + 14 });
      if (l === selected) {
        ctx.fillStyle = C.sel;
        roundRect(ctx, colX(l - 1) + 8, 4, colX(l) - colX(l - 1) + 6, h - 8, 8); ctx.fill();
      }
      if (l === current) {
        ctx.strokeStyle = C.accent; ctx.lineWidth = 1; ctx.setLineDash([3, 3]);
        roundRect(ctx, colX(l - 1) + 8, 4, colX(l) - colX(l - 1) + 6, h - 8, 8); ctx.stroke(); ctx.setLineDash([]);
      }
    }
    // edges: W_l[i][j] connects neuron j of column l-1 to neuron i of column l
    for (let l = 1; l < cols; l++) {
      const W = net.weightMatrix(l - 1);
      let mx = 1e-9;
      for (const row of W) for (const v of row) mx = Math.max(mx, Math.abs(v));
      for (let i = 0; i < shown[l]; i++) {
        for (let j = 0; j < shown[l - 1]; j++) {
          const v = W[i][j];
          ctx.strokeStyle = signed(v, mx, 0.06);
          ctx.lineWidth = 0.4 + 2.6 * (Math.abs(v) / mx);
          ctx.beginPath(); ctx.moveTo(colX(l - 1), nodeY(l - 1, j)); ctx.lineTo(colX(l), nodeY(l, i)); ctx.stroke();
        }
      }
    }
    // neurons, coloured by their value at the probe point
    for (let k = 0; k < cols; k++) {
      const vals = values && values[k];
      let mx = 1e-9;
      if (vals) for (const v of vals) mx = Math.max(mx, Math.abs(v));
      for (let i = 0; i < shown[k]; i++) {
        const y = nodeY(k, i);
        ctx.beginPath(); ctx.arc(colX(k), y, 6, 0, 2 * Math.PI);
        ctx.fillStyle = C.panel; ctx.fill();
        if (vals) { ctx.fillStyle = signed(vals[i], Math.max(1, mx), 0.08); ctx.fill(); }
        ctx.strokeStyle = 'rgba(230,237,243,0.5)'; ctx.lineWidth = 1; ctx.stroke();
      }
      if (dims[k] > MAXN) { ctx.fillStyle = C.muted; ctx.font = '10px ui-sans-serif, system-ui'; ctx.fillText(`+${dims[k] - MAXN}`, colX(k) - 8, h - bottom + 4); }
      // label
      ctx.fillStyle = k === 0 ? C.muted : (k === selected ? C.text : C.muted);
      ctx.font = `${k === selected ? 600 : 400} 11px ui-sans-serif, system-ui`;
      const label = k === 0 ? 'input' : `L${k} ${net.acts[k - 1]}`;
      const tw = ctx.measureText(label).width;
      ctx.fillText(label, Math.min(w - tw - 2, Math.max(2, colX(k) - tw / 2)), h - 8);
    }
    return hits;
  }

  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }

  // ---------------------------------------------------------------------------
  // Weight matrix
  // ---------------------------------------------------------------------------

  /**
   * W: rows (out × in), b: bias (out).  Also draws the change since training started if W0 is given
   * (a thin bar under each cell).  Returns [{ i, j, x, y, w, h, kind: 'W' | 'b' }].
   */
  function drawWeights(canvas, W, b, opt = {}) {
    const { ctx, w, h } = prep(canvas);
    const nout = W.length, nin = W[0].length;
    const labW = 26, labH = 16, gap = 10;
    const cell = Math.max(8, Math.min(64, (w - labW - gap - 8) / (nin + 1), (h - labH - 6) / nout));
    const gridW = labW + (nin + 1) * cell + gap;
    const x0 = labW + Math.max(0, (w - gridW) / 2), y0 = labH;   // centred
    let mx = 1e-9;
    for (const r of W) for (const v of r) mx = Math.max(mx, Math.abs(v));
    for (const v of b) mx = Math.max(mx, Math.abs(v));
    const cells = [], showNum = cell >= 26;
    ctx.font = '10px ui-monospace, monospace';
    ctx.fillStyle = C.muted;
    for (let j = 0; j < nin; j++) if (nin <= 16) ctx.fillText(`a${j + 1}`, x0 + j * cell + cell / 2 - 7, y0 - 4);
    ctx.fillText('b', x0 + nin * cell + gap + cell / 2 - 3, y0 - 4);
    for (let i = 0; i < nout; i++) {
      if (nout <= 16) { ctx.fillStyle = C.muted; ctx.fillText(`z${i + 1}`, x0 - 22, y0 + i * cell + cell / 2 + 3); }
      const put = (v, x, y, kind, jj) => {
        ctx.fillStyle = C.panel; ctx.fillRect(x, y, cell - 2, cell - 2);
        ctx.fillStyle = signed(v, mx, 0.05); ctx.fillRect(x, y, cell - 2, cell - 2);
        if (opt.W0 && kind === 'W') {
          const d = v - opt.W0[i][jj];
          ctx.fillStyle = d >= 0 ? rgba(C.pos, 0.9) : rgba(C.neg, 0.9);
          const len = Math.min(1, Math.abs(d) / mx) * (cell - 4);
          ctx.fillRect(x + 1, y + cell - 5, Math.max(1, len), 2);
        }
        if (showNum) {
          ctx.fillStyle = Math.abs(v) / mx > 0.55 ? (v > 0 ? '#0d1117' : '#ffffff') : C.text;
          let t = fmt(v), tw = ctx.measureText(t).width;
          if (tw > cell - 6) { t = v.toFixed(1); tw = ctx.measureText(t).width; }
          ctx.fillText(t, x + (cell - 2 - tw) / 2, y + cell / 2 + 2);
        }
        if (opt.hover && opt.hover.kind === kind && opt.hover.i === i && opt.hover.j === jj) {
          ctx.strokeStyle = '#ffffff'; ctx.lineWidth = 1.5; ctx.strokeRect(x + 0.5, y + 0.5, cell - 3, cell - 3);
        }
        cells.push({ i, j: jj, x, y, w: cell - 2, h: cell - 2, kind });
      };
      for (let j = 0; j < nin; j++) put(W[i][j], x0 + j * cell, y0 + i * cell, 'W', j);
      put(b[i], x0 + nin * cell + gap, y0 + i * cell, 'b', 0);
    }
    return cells;
  }

  // ---------------------------------------------------------------------------
  // Activation function
  // ---------------------------------------------------------------------------

  /** σ(z) (solid), σ'(z) (dashed) and a histogram of the pre-activations z of the drawn points. */
  function drawActivation(canvas, act, zs) {
    const { ctx, w, h } = prep(canvas);
    const A = NN.ACTIVATIONS[act];
    let lo = -4, hi = 4;
    if (zs && zs.length) {
      const s = zs.slice().sort((a, b) => a - b), q = (p) => s[Math.floor(p * (s.length - 1))];
      lo = Math.min(-3, q(0.01) - 0.5); hi = Math.max(3, q(0.99) + 0.5);
    }
    const pad = { l: 30, r: 8, t: 8, b: 18 };
    const X = (z) => pad.l + ((w - pad.l - pad.r) * (z - lo)) / (hi - lo);
    let ymin = Infinity, ymax = -Infinity;
    const N = 160, fz = [], dz = [];
    for (let k = 0; k <= N; k++) {
      const z = lo + ((hi - lo) * k) / N, f = A.f(z), d = A.df(z);
      fz.push(f); dz.push(d);
      ymin = Math.min(ymin, f, d); ymax = Math.max(ymax, f, d);
    }
    if (ymax - ymin < 1e-6) { ymax += 1; ymin -= 1; }
    const m = 0.08 * (ymax - ymin); ymin -= m; ymax += m;
    const Y = (v) => h - pad.b - ((h - pad.t - pad.b) * (v - ymin)) / (ymax - ymin);
    // histogram of where the points are
    if (zs && zs.length) {
      const bins = 48, cnt = new Array(bins).fill(0);
      for (const z of zs) { const k = Math.floor(((z - lo) / (hi - lo)) * bins); if (k >= 0 && k < bins) cnt[k]++; }
      const cm = Math.max(...cnt);
      ctx.fillStyle = 'rgba(139,148,158,0.28)';
      for (let k = 0; k < bins; k++) {
        if (!cnt[k]) continue;
        const x = X(lo + ((hi - lo) * k) / bins), bw = (w - pad.l - pad.r) / bins, bh = (cnt[k] / cm) * (h - pad.t - pad.b) * 0.9;
        ctx.fillRect(x, h - pad.b - bh, bw - 1, bh);
      }
    }
    // axes
    ctx.strokeStyle = C.line; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(pad.l, Y(0)); ctx.lineTo(w - pad.r, Y(0)); ctx.moveTo(X(0), pad.t); ctx.lineTo(X(0), h - pad.b); ctx.stroke();
    ctx.fillStyle = C.muted; ctx.font = '10px ui-monospace, monospace';
    ctx.fillText(ymax.toFixed(1), 2, pad.t + 8); ctx.fillText(ymin.toFixed(1), 2, h - pad.b);
    ctx.fillText(lo.toFixed(1), pad.l, h - 4); ctx.fillText(hi.toFixed(1), w - pad.r - 22, h - 4); ctx.fillText('z', X(0) + 4, h - 4);
    const curve = (arr, color, dash) => {
      ctx.strokeStyle = color; ctx.lineWidth = 2; ctx.setLineDash(dash || []);
      ctx.beginPath();
      arr.forEach((v, k) => { const x = X(lo + ((hi - lo) * k) / N), y = Y(v); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke(); ctx.setLineDash([]);
    };
    curve(dz, 'rgba(201,209,217,0.8)', [4, 4]);
    curve(fz, C.accent);
    ctx.fillStyle = 'rgba(13,17,23,0.75)'; ctx.fillRect(pad.l + 4, pad.t, 92, 28);
    ctx.fillStyle = C.accent; ctx.fillText(`— ${act}(z)`, pad.l + 8, pad.t + 11);
    ctx.fillStyle = 'rgba(201,209,217,0.95)'; ctx.fillText(`- - ${act}′(z)`, pad.l + 8, pad.t + 23);
  }

  // ---------------------------------------------------------------------------
  // Weight history
  // ---------------------------------------------------------------------------

  /** history: [{ step, theta }]; layout: { w, b, nin, nout } offsets into theta.  One line per weight. */
  function drawHistory(canvas, history, L) {
    const { ctx, w, h } = prep(canvas);
    ctx.fillStyle = C.muted; ctx.font = '10px ui-monospace, monospace';
    if (!history || history.length < 2) { ctx.fillText('train to see how the weights move', 34, h / 2); return; }
    const pad = { l: 34, r: 8, t: 8, b: 18 };
    const idx = [];
    for (let k = 0; k < L.nout * L.nin; k++) idx.push({ off: L.w + k, row: Math.floor(k / L.nin), bias: false });
    for (let i = 0; i < L.nout; i++) idx.push({ off: L.b + i, row: i, bias: true });
    const lines = idx.slice(0, 96);
    let lo = Infinity, hi = -Infinity;
    for (const s of history) for (const e of lines) { const v = s.theta[e.off]; lo = Math.min(lo, v); hi = Math.max(hi, v); }
    if (hi - lo < 1e-9) { hi += 1; lo -= 1; }
    const s0 = history[0].step, s1 = history[history.length - 1].step;
    // x = training step; if every snapshot has the same step (hand edits only), space them evenly instead
    const byIndex = s1 === s0, n = history.length;
    const X = (st, k) => pad.l + ((w - pad.l - pad.r) * (byIndex ? k / Math.max(1, n - 1) : (st - s0) / Math.max(1, s1 - s0)));
    const Y = (v) => h - pad.b - ((h - pad.t - pad.b) * (v - lo)) / (hi - lo);
    ctx.strokeStyle = C.line; ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
    if (lo < 0 && hi > 0) { ctx.strokeStyle = C.grid; ctx.beginPath(); ctx.moveTo(pad.l, Y(0)); ctx.lineTo(w - pad.r, Y(0)); ctx.stroke(); }
    ctx.fillStyle = C.muted;
    ctx.fillText(hi.toFixed(2), 2, pad.t + 8); ctx.fillText(lo.toFixed(2), 2, h - pad.b);
    ctx.fillText(byIndex ? 'start' : `step ${s0}`, pad.l, h - 4); const e = byIndex ? `edit ${n - 1}` : `step ${s1}`; ctx.fillText(e, w - pad.r - ctx.measureText(e).width, h - 4);
    for (const ln of lines) {
      ctx.strokeStyle = rgba(ROWS[ln.row % ROWS.length], ln.bias ? 0.55 : 0.9);
      ctx.lineWidth = ln.bias ? 1 : 1.4; ctx.setLineDash(ln.bias ? [3, 3] : []);
      ctx.beginPath();
      history.forEach((s, k) => { const x = X(s.step, k), y = Y(s.theta[ln.off]); k ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  root.Inspector = { drawNetwork, drawWeights, drawActivation, drawHistory };
})(window);
