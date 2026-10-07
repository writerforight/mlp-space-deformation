/*
 * nn.js — a tiny neural-network library written from first principles.
 *
 * Everything the visualization needs lives here:
 *   - a seeded random number generator (reproducible "seeds")
 *   - an MLP whose parameters are stored in ONE flat Float64Array
 *     (makes optimizers, gradient projection and EWC simple vector maths)
 *   - forward pass, backpropagation, input Jacobians, parameter Jacobians
 *   - losses (MSE, softmax cross-entropy), SGD and Adam
 *   - small linear-algebra helpers (symmetric eigen-decomposition, PCA, singular values)
 *   - toy datasets and target transforms
 *   - a Trainer with joint / sequential training and anti-forgetting methods
 *     (replay buffer, orthogonal gradient projection, EWC)
 *   - analysis helpers (gradient interference, NTK, Lipschitz estimates, layer rank)
 *
 * The file works both as a browser <script> (defines window.NN) and as a Node module (for tests).
 */
(function (root) {
  'use strict';

  // ===========================================================================
  // Random numbers
  // ===========================================================================

  /** Mulberry32: a small, fast, seedable PRNG returning floats in [0, 1). */
  class Rng {
    constructor(seed) {
      this.s = (seed >>> 0) || 1;
      this._spare = null;
    }
    next() {
      let t = (this.s = (this.s + 0x6d2b79f5) >>> 0);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    uniform(a = 0, b = 1) { return a + (b - a) * this.next(); }
    int(n) { return Math.floor(this.next() * n); }
    /** Standard normal via the Box–Muller transform. */
    normal() {
      if (this._spare !== null) { const s = this._spare; this._spare = null; return s; }
      let u = 0, v = 0;
      while (u === 0) u = this.next();
      v = this.next();
      const r = Math.sqrt(-2 * Math.log(u));
      this._spare = r * Math.sin(2 * Math.PI * v);
      return r * Math.cos(2 * Math.PI * v);
    }
  }

  // ===========================================================================
  // Activations: value f(z) and derivative f'(z)
  // ===========================================================================

  const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
  const ACTIVATIONS = {
    identity: { f: (z) => z, df: () => 1, maxSlope: 1, injective: true },
    relu: { f: (z) => (z > 0 ? z : 0), df: (z) => (z > 0 ? 1 : 0), maxSlope: 1, injective: false },
    tanh: { f: Math.tanh, df: (z) => { const t = Math.tanh(z); return 1 - t * t; }, maxSlope: 1, injective: true },
    sigmoid: {
      f: (z) => 1 / (1 + Math.exp(-z)),
      df: (z) => { const s = 1 / (1 + Math.exp(-z)); return s * (1 - s); },
      maxSlope: 0.25, injective: true,
    },
    // GELU, tanh approximation: 0.5 z (1 + tanh(√(2/π)(z + 0.044715 z³)))
    gelu: {
      f: (z) => 0.5 * z * (1 + Math.tanh(SQRT_2_OVER_PI * (z + 0.044715 * z * z * z))),
      df: (z) => {
        const u = SQRT_2_OVER_PI * (z + 0.044715 * z * z * z);
        const t = Math.tanh(u);
        const du = SQRT_2_OVER_PI * (1 + 3 * 0.044715 * z * z);
        return 0.5 * (1 + t) + 0.5 * z * (1 - t * t) * du;
      },
      maxSlope: 1.13, injective: false, // GELU dips below zero, so it is not monotonic
    },
    sin: { f: Math.sin, df: Math.cos, maxSlope: 1, injective: false },
  };
  const ACTIVATION_NAMES = Object.keys(ACTIVATIONS);

  // ===========================================================================
  // The MLP
  // ===========================================================================

  /**
   * Multi-layer perceptron.  Layer l maps a_{l-1} (size dims[l]) to
   *   z_l = W_l a_{l-1} + b_l        (the "linear" sub-step)
   *   a_l = act_l(z_l)               (the "activation" sub-step)
   * All parameters live in `theta` (Float64Array).  For layer l, W_l is stored row-major
   * starting at layout[l].w (shape out×in), followed by b_l at layout[l].b.
   */
  class MLP {
    constructor(dims, acts) {
      if (acts.length !== dims.length - 1) throw new Error('need one activation per layer');
      this.dims = dims.slice();
      this.acts = acts.slice();
      this.layout = [];
      let off = 0;
      for (let l = 0; l < dims.length - 1; l++) {
        const nin = dims[l], nout = dims[l + 1];
        this.layout.push({ nin, nout, w: off, b: off + nin * nout });
        off += nin * nout + nout;
      }
      this.nParams = off;
      this.theta = new Float64Array(off);
    }

    get nLayers() { return this.dims.length - 1; }
    get inDim() { return this.dims[0]; }
    get outDim() { return this.dims[this.dims.length - 1]; }

    clone() {
      const m = new MLP(this.dims, this.acts);
      m.theta.set(this.theta);
      return m;
    }

    /**
     * Random initialisation.
     *   dist:  'normal' N(0, s²) | 'uniform' U(-s, s) | 'xavier' N(0, s²·2/(in+out)) | 'he' N(0, s²·2/in)
     *   scale: multiplies the standard deviation / range
     * Biases come from the same distribution, scaled by 0.25, so random networks are not
     * forced to keep the origin fixed.
     */
    init(dist, scale, rng) {
      for (const L of this.layout) {
        let std;
        switch (dist) {
          case 'uniform': std = null; break;
          case 'xavier': std = scale * Math.sqrt(2 / (L.nin + L.nout)); break;
          case 'he': std = scale * Math.sqrt(2 / L.nin); break;
          default: std = scale;
        }
        const draw = () => (std === null ? rng.uniform(-scale, scale) : std * rng.normal());
        for (let i = 0; i < L.nin * L.nout; i++) this.theta[L.w + i] = draw();
        for (let i = 0; i < L.nout; i++) this.theta[L.b + i] = 0.25 * draw();
      }
      return this;
    }

    /** Forward pass keeping every intermediate value (needed by backprop and the visualization). */
    forward(x) {
      const zs = [], as = [Float64Array.from(x)];
      let a = as[0];
      for (let l = 0; l < this.layout.length; l++) {
        const L = this.layout[l], th = this.theta, act = ACTIVATIONS[this.acts[l]];
        const z = new Float64Array(L.nout), out = new Float64Array(L.nout);
        for (let i = 0; i < L.nout; i++) {
          let s = th[L.b + i];
          const row = L.w + i * L.nin;
          for (let j = 0; j < L.nin; j++) s += th[row + j] * a[j];
          z[i] = s;
          out[i] = act.f(s);
        }
        zs.push(z);
        as.push(out);
        a = out;
      }
      return { zs, as };
    }

    predict(x) {
      const c = this.forward(x);
      return c.as[c.as.length - 1];
    }

    /** Every stage a point goes through: [x, z_1, a_1, z_2, a_2, …]. Stage 2l+1 is "after linear l". */
    stages(x) {
      const { zs, as } = this.forward(x);
      const out = [as[0]];
      for (let l = 0; l < zs.length; l++) out.push(zs[l], as[l + 1]);
      return out;
    }

    /**
     * Backpropagation.  Given the forward cache and gOut = ∂L/∂(output), ACCUMULATES ∂L/∂θ
     * into `grad` (a Float64Array of length nParams) and returns ∂L/∂x.
     */
    backward(cache, gOut, grad) {
      const th = this.theta;
      let gA = Float64Array.from(gOut);
      for (let l = this.layout.length - 1; l >= 0; l--) {
        const L = this.layout[l], act = ACTIVATIONS[this.acts[l]];
        const z = cache.zs[l], aPrev = cache.as[l];
        const gZ = new Float64Array(L.nout);
        for (let i = 0; i < L.nout; i++) gZ[i] = gA[i] * act.df(z[i]);   // through the activation
        const gPrev = new Float64Array(L.nin);
        for (let i = 0; i < L.nout; i++) {
          const row = L.w + i * L.nin, g = gZ[i];
          if (grad) {
            grad[L.b + i] += g;                                            // ∂L/∂b_i = gZ_i
            for (let j = 0; j < L.nin; j++) grad[row + j] += g * aPrev[j]; // ∂L/∂W_ij = gZ_i a_j
          }
          for (let j = 0; j < L.nin; j++) gPrev[j] += th[row + j] * g;     // ∂L/∂a_prev = Wᵀ gZ
        }
        gA = gPrev;
      }
      return gA;
    }

    /**
     * Jacobian of every stage with respect to the input (forward-mode, chain rule):
     *   J_0 = I,  after linear: J ← W J,  after activation: J ← diag(f'(z)) J.
     * Returns { stages, jacs } where jacs[s] is an array of rows (dim(stage s) × inDim).
     */
    stageJacobians(x) {
      const { zs, as } = this.forward(x);
      const d0 = this.inDim, th = this.theta;
      let J = [];
      for (let i = 0; i < d0; i++) { const r = new Float64Array(d0); r[i] = 1; J.push(r); }
      const stages = [as[0]], jacs = [J];
      for (let l = 0; l < this.layout.length; l++) {
        const L = this.layout[l], act = ACTIVATIONS[this.acts[l]];
        const WJ = [];
        for (let i = 0; i < L.nout; i++) {
          const r = new Float64Array(d0), row = L.w + i * L.nin;
          for (let j = 0; j < L.nin; j++) {
            const w = th[row + j];
            if (w !== 0) for (let k = 0; k < d0; k++) r[k] += w * J[j][k];
          }
          WJ.push(r);
        }
        const DJ = WJ.map((r, i) => r.map((v) => v * act.df(zs[l][i])));
        stages.push(zs[l], as[l + 1]);
        jacs.push(WJ, DJ);
        J = DJ;
      }
      return { stages, jacs };
    }

    /** Jacobian of the network output with respect to the input (outDim × inDim). */
    inputJacobian(x) {
      const r = this.stageJacobians(x);
      return r.jacs[r.jacs.length - 1];
    }

    /** Jacobian of the output with respect to the parameters: one gradient vector per output. */
    paramJacobian(x) {
      const cache = this.forward(x), rows = [];
      for (let k = 0; k < this.outDim; k++) {
        const e = new Float64Array(this.outDim); e[k] = 1;
        const g = new Float64Array(this.nParams);
        this.backward(cache, e, g);
        rows.push(g);
      }
      return rows;
    }

    /** Spectral norm (largest singular value) of each weight matrix. */
    weightMatrix(l) {
      const L = this.layout[l], rows = [];
      for (let i = 0; i < L.nout; i++) rows.push(this.theta.slice(L.w + i * L.nin, L.w + (i + 1) * L.nin));
      return rows;
    }
  }

  // ===========================================================================
  // Losses.  Each returns { loss, grad } where grad = ∂loss/∂output.
  // ===========================================================================

  /** Half squared error ½‖y − t‖². */
  function mse(y, t) {
    let loss = 0;
    const g = new Float64Array(y.length);
    for (let i = 0; i < y.length; i++) { const d = y[i] - t[i]; loss += 0.5 * d * d; g[i] = d; }
    return { loss, grad: g };
  }

  /** Softmax cross-entropy: outputs are logits, label is a class index. */
  function crossEntropy(y, label) {
    let m = -Infinity;
    for (const v of y) m = Math.max(m, v);
    let s = 0;
    const p = new Float64Array(y.length);
    for (let i = 0; i < y.length; i++) { p[i] = Math.exp(y[i] - m); s += p[i]; }
    for (let i = 0; i < y.length; i++) p[i] /= s;
    const loss = -Math.log(Math.max(p[label], 1e-12));
    p[label] -= 1; // ∂/∂logits = softmax − onehot
    return { loss, grad: p };
  }

  /** Loss of one sample.  sample = { x, y } where y is a target vector (mse) or class index (ce). */
  /*
   * A sample may carry its own loss type and weight: { x, y, type?: 'mse' | 'ce', w?: number }.  That lets
   * several goals with different losses train together (total loss = Σ w · L).  Without them the batch
   * type and weight 1 apply, as before.
   */
  function sampleLoss(net, sample, type) {
    const out = net.predict(sample.x), ty = sample.type || type, w = sample.w ?? 1;
    return w * (ty === 'ce' ? crossEntropy(out, sample.y) : mse(out, sample.y)).loss;
  }

  /** Mean loss and mean gradient ∂L/∂θ over a batch. */
  function batchLossGrad(net, batch, type) {
    const grad = new Float64Array(net.nParams);
    let loss = 0;
    for (const s of batch) {
      const cache = net.forward(s.x);
      const out = cache.as[cache.as.length - 1];
      const ty = s.type || type, w = s.w ?? 1;
      const r = ty === 'ce' ? crossEntropy(out, s.y) : mse(out, s.y);
      loss += w * r.loss;
      if (w !== 1) for (let i = 0; i < r.grad.length; i++) r.grad[i] *= w;
      net.backward(cache, r.grad, grad);
    }
    const n = Math.max(1, batch.length);
    for (let i = 0; i < grad.length; i++) grad[i] /= n;
    return { loss: loss / n, grad };
  }

  function meanLoss(net, samples, type) {
    if (!samples.length) return 0;
    let s = 0;
    for (const p of samples) s += sampleLoss(net, p, type);
    return s / samples.length;
  }

  // ===========================================================================
  // Optimizers (operate on flat parameter vectors)
  // ===========================================================================

  class SGD {
    constructor(n, lr) { this.lr = lr; }
    step(theta, grad) { for (let i = 0; i < theta.length; i++) theta[i] -= this.lr * grad[i]; }
  }

  /** Adam (Kingma & Ba): per-parameter step sizes from running gradient moments. */
  class Adam {
    constructor(n, lr, b1 = 0.9, b2 = 0.999, eps = 1e-8) {
      Object.assign(this, { lr, b1, b2, eps, t: 0 });
      this.m = new Float64Array(n);
      this.v = new Float64Array(n);
    }
    step(theta, grad) {
      this.t++;
      const { b1, b2 } = this, c1 = 1 - Math.pow(b1, this.t), c2 = 1 - Math.pow(b2, this.t);
      for (let i = 0; i < theta.length; i++) {
        this.m[i] = b1 * this.m[i] + (1 - b1) * grad[i];
        this.v[i] = b2 * this.v[i] + (1 - b2) * grad[i] * grad[i];
        theta[i] -= (this.lr * (this.m[i] / c1)) / (Math.sqrt(this.v[i] / c2) + this.eps);
      }
    }
  }

  // ===========================================================================
  // Linear algebra helpers
  // ===========================================================================

  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  const norm = (a) => Math.sqrt(dot(a, a));

  /**
   * Eigen-decomposition of a small symmetric matrix (array of rows) with the cyclic Jacobi method.
   * Returns { values, vectors } sorted by decreasing eigenvalue; vectors[k] is the k-th eigenvector.
   */
  function symEig(A) {
    const n = A.length;
    const a = A.map((r) => Float64Array.from(r));
    const V = [];
    for (let i = 0; i < n; i++) { const r = new Float64Array(n); r[i] = 1; V.push(r); }
    for (let sweep = 0; sweep < 60; sweep++) {
      let off = 0;
      for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p][q] * a[p][q];
      if (off < 1e-22) break;
      for (let p = 0; p < n; p++) {
        for (let q = p + 1; q < n; q++) {
          if (Math.abs(a[p][q]) < 1e-300) continue;
          const theta = (a[q][q] - a[p][p]) / (2 * a[p][q]);
          const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
          const c = 1 / Math.sqrt(t * t + 1), s = t * c;
          for (let k = 0; k < n; k++) {            // rotate columns p, q
            const akp = a[k][p], akq = a[k][q];
            a[k][p] = c * akp - s * akq; a[k][q] = s * akp + c * akq;
          }
          for (let k = 0; k < n; k++) {            // rotate rows p, q
            const apk = a[p][k], aqk = a[q][k];
            a[p][k] = c * apk - s * aqk; a[q][k] = s * apk + c * aqk;
          }
          for (let k = 0; k < n; k++) {            // accumulate eigenvectors (rows of V)
            const vp = V[p][k], vq = V[q][k];
            V[p][k] = c * vp - s * vq; V[q][k] = s * vp + c * vq;
          }
        }
      }
    }
    const idx = [...Array(n).keys()].sort((i, j) => a[j][j] - a[i][i]);
    return { values: idx.map((i) => a[i][i]), vectors: idx.map((i) => V[i]) };
  }

  /** Singular values of a matrix (rows) = √eig(MᵀM), sorted descending. */
  function singularValues(M) {
    const n = M[0].length, G = [];
    for (let i = 0; i < n; i++) {
      const r = new Float64Array(n);
      for (let j = 0; j < n; j++) { let s = 0; for (const row of M) s += row[i] * row[j]; r[j] = s; }
      G.push(r);
    }
    const sv = symEig(G).values.map((v) => Math.sqrt(Math.max(0, v)));
    return sv.slice(0, Math.min(M.length, n));
  }

  /** Numerical rank from singular values. */
  function rankOf(sv, rel = 1e-6) {
    const mx = sv.length ? sv[0] : 0;
    return sv.filter((s) => s > rel * Math.max(mx, 1e-12)).length;
  }

  /**
   * PCA basis (k × d) of a point cloud (array of d-vectors).  If d ≤ k the basis is the identity
   * (padded), so low-dimensional layers are drawn exactly, without projection.
   */
  function pcaBasis(points, k) {
    const d = points[0].length;
    if (d <= k) {
      const B = [];
      for (let i = 0; i < k; i++) { const r = new Float64Array(d); if (i < d) r[i] = 1; B.push(r); }
      return { basis: B, mean: new Float64Array(d), exact: true, explained: 1 };
    }
    const mean = new Float64Array(d);
    for (const p of points) for (let i = 0; i < d; i++) mean[i] += p[i] / points.length;
    const C = [];
    for (let i = 0; i < d; i++) C.push(new Float64Array(d));
    for (const p of points) {
      for (let i = 0; i < d; i++) {
        const di = p[i] - mean[i];
        for (let j = i; j < d; j++) C[i][j] += di * (p[j] - mean[j]);
      }
    }
    for (let i = 0; i < d; i++) for (let j = 0; j < i; j++) C[i][j] = C[j][i];
    const { values, vectors } = symEig(C);
    const total = values.reduce((s, v) => s + Math.max(0, v), 0) || 1;
    const top = values.slice(0, k).reduce((s, v) => s + Math.max(0, v), 0);
    return { basis: vectors.slice(0, k), mean, exact: false, explained: top / total };
  }

  // ===========================================================================
  // Datasets and target transforms
  // ===========================================================================

  /**
   * Two-class toy datasets.  Returns [{ x, y }] with y ∈ {0, 1} (classes alternate, so they are balanced).
   * dim = 2 or 3.  Kinds:
   *   blobs    two Gaussian clouds — already linearly separable (the network barely needs to bend space)
   *   moons    two interleaving half-circles
   *   circles  a disk inside a ring (3D: a ball inside a shell)
   *   rings    blue–red–blue nested rings: the middle ring needs two folds
   *   xor      opposite quadrants share a class (3D: octants by sign parity) — needs a fold
   *   checker  checkerboard (4×4 in 2D, 3×3×3 in 3D) — many regions, tests capacity
   *   wave     a wavy (sine) boundary
   *   spirals  two interleaved spiral arms
   *   linked   (3D) two interlocked rings — a Hopf link: no continuous invertible map of 3D space can
   *            pull them apart.  A width-3 network only gets close by making a weight matrix (nearly)
   *            singular — crushing a dimension — and still plateaus; with width ≥ 4 it separates them cleanly.
   */
  function makeDataset(kind, n, dim, rng, noise = 0.08) {
    const out = [];
    const jitter = () => noise * rng.normal();
    const box = (e) => [...Array(dim)].map(() => rng.uniform(-e, e));
    // rejection sampling for region-defined datasets: a point of class `want`, at least `gap` from the boundary
    const region = (want, rule, e = 1.6, gap = 0.06) => {
      for (let k = 0; k < 20000; k++) {
        const p = box(e), r = rule(p);
        if (r.label === want && r.margin >= gap) return p;
      }
      return box(e);
    };
    if (kind === 'linked' && dim === 2) kind = 'rings';   // the link only exists in 3D
    for (let i = 0; i < n; i++) {
      const c = i % 2;
      let p;
      if (kind === 'blobs') {
        const m = c === 0 ? [0.8, 0.5, 0.4] : [-0.8, -0.5, -0.4];
        p = [...Array(dim)].map((_, k) => m[k] + 0.35 * rng.normal());
      } else if (kind === 'moons') {
        const t = Math.PI * rng.next();
        p = c === 0 ? [Math.cos(t) - 0.5, Math.sin(t) - 0.25] : [0.5 - Math.cos(t), 0.25 - Math.sin(t)];
        p = [p[0] * 1.2 + jitter(), p[1] * 1.2 + jitter()];
        if (dim === 3) p.push(0.3 * jitter());
      } else if (kind === 'circles' || kind === 'rings') {
        // circles: disk (blue) inside ring (red); rings: blue / red / blue
        const r = kind === 'circles' ? (c === 0 ? 0.5 : 1.3) : (c === 1 ? 0.95 : (rng.next() < 0.4 ? 0.4 : 1.5));
        if (dim === 3) {                         // spherical shells
          const u = 2 * rng.next() - 1, phi = 2 * Math.PI * rng.next(), s = Math.sqrt(1 - u * u);
          p = [r * s * Math.cos(phi) + jitter(), r * s * Math.sin(phi) + jitter(), r * u + jitter()];
        } else {
          const t = 2 * Math.PI * rng.next();
          p = [r * Math.cos(t) + jitter(), r * Math.sin(t) + jitter()];
        }
      } else if (kind === 'xor') {
        p = region(c, (q) => ({ label: q.reduce((s, v) => s * Math.sign(v), 1) > 0 ? 0 : 1, margin: Math.min(...q.map(Math.abs)) }), 1.5, 0.1);
      } else if (kind === 'checker') {
        const cells = dim === 2 ? 4 : 3, e = dim === 2 ? 1.6 : 1.5, w = (2 * e) / cells;
        p = region(c, (q) => {
          let sum = 0, margin = Infinity;
          for (const v of q) {
            const u = (v + e) / w, k = Math.min(cells - 1, Math.floor(u));
            sum += k; margin = Math.min(margin, w * Math.min(u - k, k + 1 - u));
          }
          return { label: sum % 2, margin };
        }, e, 0.05);
      } else if (kind === 'wave') {
        const f = (q) => (dim === 2 ? q[1] - 0.6 * Math.sin(2.4 * q[0]) : q[2] - 0.6 * Math.sin(2 * q[0]) * Math.cos(2 * q[1]));
        p = region(c, (q) => ({ label: f(q) > 0 ? 0 : 1, margin: Math.abs(f(q)) }), 1.6, 0.08);
      } else if (kind === 'linked') {
        // ring 0 in the xy-plane around (−0.5, 0, 0); ring 1 in the xz-plane around (0.5, 0, 0):
        // each ring passes through the other's centre, so they are linked
        const t = 2 * Math.PI * rng.next();
        p = c === 0 ? [-0.5 + Math.cos(t), Math.sin(t), 0] : [0.5 - Math.cos(t), 0, Math.sin(t)];
        p = p.map((v) => v + jitter());
      } else { // spirals: two interleaved arms
        const t = 0.25 + 2.6 * rng.next();
        const r = 0.25 + 0.55 * t;
        const a = t * 1.9 + c * Math.PI;
        p = [r * Math.cos(a) * 0.8 + 0.6 * jitter(), r * Math.sin(a) * 0.8 + 0.6 * jitter()];
        if (dim === 3) p.push(0.25 * (t - 1.5) + 0.5 * jitter());
      }
      out.push({ x: Float64Array.from(p), y: c });
    }
    return out;
  }

  /** Linear target maps A (d×d) for "match a fixed transform". */
  function targetMatrix(kind, dim, amount) {
    const I = (d) => [...Array(d)].map((_, i) => { const r = new Float64Array(d); r[i] = 1; return r; });
    const A = I(dim);
    if (kind === 'rotation') {
      const c = Math.cos(amount), s = Math.sin(amount);
      A[0][0] = c; A[0][1] = -s; A[1][0] = s; A[1][1] = c;           // about the z axis in 3D
    } else if (kind === 'shear') {
      A[0][1] = amount;
      if (dim === 3) A[1][2] = 0.5 * amount;
    } else if (kind === 'scaling') {
      A[0][0] = amount; A[1][1] = 1 / amount;
      if (dim === 3) A[2][2] = Math.sqrt(amount);
    }
    return A;
  }

  const matVec = (A, x) => Float64Array.from(A.map((r) => dot(r, x)));

  /** Samples for the transform task: points in a box with targets A·x. */
  function makeTransformData(A, n, dim, rng, extent = 1.5) {
    const out = [];
    for (let i = 0; i < n; i++) {
      const x = Float64Array.from([...Array(dim)].map(() => rng.uniform(-extent, extent)));
      out.push({ x, y: matVec(A, x) });
    }
    return out;
  }

  /** Fibonacci (golden-spiral) points on a sphere — nearly uniform, in a continuous order. */
  function fibonacciSphere(n, center = [0, 0, 0], radius = 1) {
    const pts = [], ga = Math.PI * (3 - Math.sqrt(5));
    for (let i = 0; i < n; i++) {
      const y = 1 - (2 * (i + 0.5)) / n, r = Math.sqrt(1 - y * y), t = ga * i;
      pts.push(Float64Array.from([center[0] + radius * r * Math.cos(t), center[1] + radius * y,
        center[2] + radius * r * Math.sin(t)]));
    }
    return pts;
  }

  // ===========================================================================
  // Shapes for topology experiments
  // ===========================================================================

  /*
   * Every shape is a list of components; a component is an ordered list of points (a closed or open
   * curve, or a surface drawn as one long spiral) with a class label.  Shapes that are meant to be
   * morphed into each other use the SAME parameter for the same index i (e.g. the angle t), so point i
   * of the source corresponds to point i of the target.
   */
  const TAU = 2 * Math.PI;
  const ring = (n, f) => [...Array(n)].map((_, i) => Float64Array.from(f((TAU * i) / n, i)));
  const toCube = (p, s) => { const m = Math.max(...p.map(Math.abs)) || 1; return p.map((v) => (s * v) / m); };

  const SHAPES = {
    // ---- 2D
    circle: { dim: 2, label: 'circle', make: (n) => [{ closed: true, pts: ring(n, (t) => [0.8 * Math.cos(t), 0.8 * Math.sin(t)]) }] },
    square: { dim: 2, label: 'square', make: (n) => [{ closed: true, pts: ring(n, (t) => toCube([Math.cos(t), Math.sin(t)], 0.8)) }] },
    star: { dim: 2, label: 'star', make: (n) => [{ closed: true, pts: ring(n, (t) => { const r = 0.62 * (1 + 0.35 * Math.cos(5 * t)); return [r * Math.cos(t), r * Math.sin(t)]; }) }] },
    // lemniscate of Gerono: the curve passes through the origin twice (t = π/2 and 3π/2)
    figure8: { dim: 2, label: 'figure eight', make: (n) => [{ closed: true, pts: ring(n, (t) => [0.9 * Math.cos(t), 0.9 * Math.sin(t) * Math.cos(t)]) }] },
    twoCircles: { dim: 2, label: 'two circles', make: (n) => [-1, 1].map((sx, c) => ({ closed: true, label: c,
      pts: ring(Math.floor(n / 2), (t) => [0.75 * sx + 0.45 * Math.cos(t), 0.45 * Math.sin(t)]) })) },
    // two copies of one circle on top of each other: the target of "merge two circles into one"
    oneCircleTwice: { dim: 2, label: 'one circle (both parts)', make: (n) => [0, 1].map((c) => ({ closed: true, label: c,
      pts: ring(Math.floor(n / 2), (t) => [0.8 * Math.cos(t), 0.8 * Math.sin(t)]) })) },
    diskInRing: { dim: 2, label: 'disk inside a ring', make: (n) => {
      const disk = [], k = Math.floor(n / 2), ga = Math.PI * (3 - Math.sqrt(5));
      for (let i = 0; i < k; i++) { const r = 0.45 * Math.sqrt((i + 0.5) / k); disk.push(Float64Array.from([r * Math.cos(ga * i), r * Math.sin(ga * i)])); }
      return [{ closed: false, filled: true, label: 0, pts: disk }, { closed: true, label: 1, pts: ring(n - k, (t) => [1.1 * Math.cos(t), 1.1 * Math.sin(t)]) }];
    } },
    // ---- 3D
    sphere: { dim: 3, label: 'sphere', make: (n) => [{ closed: false, pts: fibonacciSphere(n, [0, 0, 0], 0.8) }] },
    // a stretched, sheared sphere: the clean 3D control (homeomorphic, smooth, no corners)
    ellipsoid: { dim: 3, label: 'ellipsoid', make: (n) => [{ closed: false, pts: fibonacciSphere(n, [0, 0, 0], 1).map((p) =>
      Float64Array.from([1.1 * p[0] + 0.3 * p[1], 0.55 * p[1], 0.75 * p[2] + 0.2 * p[0]])) }] },
    cube: { dim: 3, label: 'cube surface', make: (n) => [{ closed: false, pts: fibonacciSphere(n, [0, 0, 0], 1).map((p) => Float64Array.from(toCube(Array.from(p), 0.75))) }] },
    torus: { dim: 3, label: 'torus (donut)', make: (n) => [{ closed: true, pts: torusPts(n) }] },
    // the torus mapped onto a sphere (longitude = u, latitude from v): what "close the hole" asks for
    torusOnSphere: { dim: 3, label: 'torus squeezed onto a sphere', make: (n) => [{ closed: true, pts: torusPts(n, true) }] },
    unknot: { dim: 3, label: 'unknotted ring', make: (n) => [{ closed: true, pts: ring(n, (t) => [0.9 * Math.cos(t), 0.9 * Math.sin(t), 0]) }] },
    trefoil: { dim: 3, label: 'trefoil knot', make: (n) => [{ closed: true, pts: ring(n, (t) => [
      0.3 * (Math.sin(t) + 2 * Math.sin(2 * t)), 0.3 * (Math.cos(t) - 2 * Math.cos(2 * t)), -0.3 * Math.sin(3 * t)]) }] },
    linkedRings: { dim: 3, label: 'two linked rings', make: (n) => [0, 1].map((c) => ({ closed: true, label: c,
      pts: ring(Math.floor(n / 2), (t) => (c === 0 ? [-0.5 + Math.cos(t), Math.sin(t), 0] : [0.5 - Math.cos(t), 0, Math.sin(t)])) })) },
    nestedSpheres: { dim: 3, label: 'ball inside a shell', make: (n) => [
      { closed: false, label: 0, pts: fibonacciSphere(Math.floor(n / 2), [0, 0, 0], 0.4) },
      { closed: false, label: 1, pts: fibonacciSphere(n - Math.floor(n / 2), [0, 0, 0], 1.1) }] },
  };

  /** Torus (R = 0.75, r = 0.3) drawn as one spiral that winds 23 times around the tube. */
  function torusPts(n, ontoSphere = false) {
    const R = 0.75, r = 0.3, W = 23;
    return [...Array(n)].map((_, i) => {
      const u = (TAU * i) / n, v = (TAU * W * i) / n;
      if (!ontoSphere) return Float64Array.from([(R + r * Math.cos(v)) * Math.cos(u), (R + r * Math.cos(v)) * Math.sin(u), r * Math.sin(v)]);
      const lat = (Math.PI / 2) * Math.cos(v);           // v and −v land on the same point: the hole is closed
      return Float64Array.from([0.8 * Math.cos(lat) * Math.cos(u), 0.8 * Math.cos(lat) * Math.sin(u), 0.8 * Math.sin(lat)]);
    });
  }

  function makeShape(name, n = 400) {
    const S = SHAPES[name];
    return { name, dim: S.dim, label: S.label, components: S.make(n) };
  }

  /** Training pairs that send point i of shape `from` to point i of shape `to`. */
  function morphData(from, to, n = 400) {
    const A = makeShape(from, n).components, B = makeShape(to, n).components;
    const out = [];
    A.forEach((ca, k) => ca.pts.forEach((p, i) => out.push({ x: p, y: B[k].pts[i], label: ca.label ?? 0 })));
    return out;
  }

  // ===========================================================================
  // Square-matrix SVD and the "homeomorphism mode" projection
  // ===========================================================================

  /** SVD of a square matrix M (rows): M = U diag(s) Vᵀ.  Returns { U, s, V } with U, V as arrays of columns. */
  function squareSvd(M) {
    const n = M.length;
    const G = [...Array(n)].map((_, i) => Float64Array.from([...Array(n)].map((_, j) => M.reduce((acc, row) => acc + row[i] * row[j], 0))));
    const { values, vectors } = symEig(G);
    const s = values.map((v) => Math.sqrt(Math.max(0, v)));
    const V = vectors.map((v) => Float64Array.from(v));
    const U = [];
    for (let k = 0; k < n; k++) {
      let u;
      if (s[k] > 1e-10 * Math.max(1, s[0])) u = Float64Array.from(M.map((row) => dot(row, V[k]) / s[k]));
      else {                                               // complete the basis (Gram–Schmidt on e_i)
        for (let e = 0; e < n && !u; e++) {
          const c = new Float64Array(n); c[e] = 1;
          for (const b of U) { const d = dot(c, b); for (let i = 0; i < n; i++) c[i] -= d * b[i]; }
          const nc = norm(c);
          if (nc > 1e-6) u = c.map((v) => v / nc);
        }
      }
      U.push(u);
    }
    return { U, s, V };
  }

  /** Determinant of a small square matrix (rows), by Gaussian elimination with partial pivoting. */
  function det(M) {
    const a = M.map((r) => Float64Array.from(r)), n = a.length;
    let d = 1;
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(a[r][c]) > Math.abs(a[p][c])) p = r;
      if (Math.abs(a[p][c]) < 1e-300) return 0;
      if (p !== c) { [a[p], a[c]] = [a[c], a[p]]; d = -d; }
      d *= a[c][c];
      for (let r = c + 1; r < n; r++) { const f = a[r][c] / a[c][c]; for (let k = c; k < n; k++) a[r][k] -= f * a[c][k]; }
    }
    return d;
  }

  /**
   * Make every square weight matrix orientation-preserving (det > 0) by flipping the sign of one row
   * (and its bias) where needed.  Invertible matrices come in two pieces, det > 0 and det < 0, and a
   * homeomorphism-mode network can never cross from one to the other while training — so it should
   * start on the orientation-preserving side when the target keeps orientation.
   */
  function orientPositive(net) {
    for (let l = 0; l < net.nLayers; l++) {
      const L = net.layout[l];
      if (L.nin !== L.nout || det(net.weightMatrix(l)) >= 0) continue;
      for (let j = 0; j < L.nin; j++) net.theta[L.w + j] *= -1;
      net.theta[L.b] *= -1;
    }
  }

  /**
   * Homeomorphism mode: raise every singular value of every square weight matrix to at least `floor`.
   * With width = input dimension and injective activations (tanh, sigmoid, identity) the network then
   * stays a homeomorphism onto its image: it can bend and stretch space but never crush a dimension —
   * so it cannot tear, glue, unlink or unknot.  Returns how many singular values were raised.
   */
  function clampSingularValues(net, floor) {
    let raised = 0;
    for (let l = 0; l < net.nLayers; l++) {
      const L = net.layout[l];
      if (L.nin !== L.nout) continue;
      const W = net.weightMatrix(l), { U, s, V } = squareSvd(W);
      if (s.every((v) => v >= floor)) continue;
      const s2 = s.map((v) => { if (v < floor) { raised++; return floor; } return v; });
      for (let i = 0; i < L.nout; i++) {
        for (let j = 0; j < L.nin; j++) {
          let v = 0;
          for (let k = 0; k < L.nin; k++) v += U[k][i] * s2[k] * V[k][j];
          net.theta[L.w + i * L.nin + j] = v;
        }
      }
    }
    return raised;
  }

  // ===========================================================================
  // Training with joint / sequential modes and anti-forgetting methods
  // ===========================================================================

  /**
   * Trainer.  `tasks` is a list of sample lists.  Joint mode mixes all samples in one loss.
   * Sequential mode trains on task 0 for `stepsPerTask` steps, then task 1, … (continual learning);
   * after finishing a task it records that task's loss (for the forgetting metric) and updates the
   * chosen anti-forgetting memory:
   *   replay — keep samples of finished tasks and mix them into later batches;
   *   ogd    — orthogonal gradient descent: remember ∂f(x)/∂θ of finished tasks and remove those
   *            directions from future gradients, so outputs on old points (to first order) do not move;
   *   ewc    — elastic weight consolidation: quadratic penalty λ/2 Σ F_i (θ_i − θ*_i)² with the
   *            diagonal Fisher information F of each finished task.
   */
  class Trainer {
    constructor(net, opts) {
      this.net = net;
      this.opts = Object.assign({
        type: 'mse', optimizer: 'adam', lr: 0.01, batch: 32, mode: 'joint', stepsPerTask: 200,
        method: 'none', ewcLambda: 50, replayPerTask: 32, seed: 1,
        clip: 10, // gradient-norm clipping: keeps stiff losses (e.g. strong EWC + SGD) from blowing up
        invertibleFloor: 0, // > 0: homeomorphism mode (minimum singular value of square weight matrices)
      }, opts);
      this.rng = new Rng(this.opts.seed * 7919 + 13);
      this.tasks = [];
      this.reset();
    }

    reset() {
      const n = this.net.nParams, o = this.opts;
      this.optim = o.optimizer === 'sgd' ? new SGD(n, o.lr) : new Adam(n, o.lr);
      this.step_ = 0;
      this.task = 0;
      this.taskStep = 0;
      this.done = false;
      this.diverged = false;
      this.lossHistory = [];
      this.lossAfterOwn = [];   // L_i measured right after finishing task i
      this.replay = [];
      this.ogdBasis = [];       // orthonormal directions in parameter space
      this.ewc = [];            // [{ fisher, anchor }]
    }

    setOpts(o) {
      const optChanged = o.optimizer !== undefined && o.optimizer !== this.opts.optimizer;
      Object.assign(this.opts, o);
      if (optChanged) this.optim = this.opts.optimizer === 'sgd' ? new SGD(this.net.nParams, this.opts.lr)
        : new Adam(this.net.nParams, this.opts.lr);
      this.optim.lr = this.opts.lr;
    }

    setTasks(tasks) { this.tasks = tasks; }

    allSamples() { return this.tasks.flat(); }

    sampleBatch(pool, k) {
      if (pool.length <= k) return pool.slice();
      const out = [];
      for (let i = 0; i < k; i++) out.push(pool[this.rng.int(pool.length)]);
      return out;
    }

    /** One optimisation step.  Returns the batch loss (or null if sequential training finished). */
    step() {
      const o = this.opts;
      if (!this.tasks.length || this.done || this.diverged) return null;
      let batch;
      if (o.mode === 'sequential') {
        batch = this.sampleBatch(this.tasks[this.task], o.batch);
        if (o.method === 'replay' && this.replay.length) batch = batch.concat(this.sampleBatch(this.replay, o.batch));
      } else {
        batch = this.sampleBatch(this.allSamples(), o.batch);
      }
      const { loss, grad } = batchLossGrad(this.net, batch, o.type);
      if (o.mode === 'sequential') {
        if (o.method === 'ewc') this.addEwcGrad(grad);
        if (o.method === 'ogd') this.projectOut(grad);
      }
      const gn = norm(grad);
      if (o.clip && gn > o.clip) for (let k = 0; k < grad.length; k++) grad[k] *= o.clip / gn;
      if (!Number.isFinite(loss) || !Number.isFinite(gn)) { this.diverged = true; return null; }
      this.optim.step(this.net.theta, grad);
      if (o.invertibleFloor > 0) clampSingularValues(this.net, o.invertibleFloor);
      this.step_++;
      this.lossHistory.push(this.totalLoss());
      if (o.mode === 'sequential' && ++this.taskStep >= o.stepsPerTask) this.finishTask();
      return loss;
    }

    /** Loss over every task (what the loss curve shows). */
    totalLoss() {
      const all = this.allSamples();
      if (all.length > 256) {              // estimate on a fixed-size subset for speed
        let s = 0;
        const stride = all.length / 256;
        for (let i = 0; i < 256; i++) s += sampleLoss(this.net, all[Math.floor(i * stride)], this.opts.type);
        return s / 256;
      }
      return meanLoss(this.net, all, this.opts.type);
    }

    taskLoss(i) { return meanLoss(this.net, this.tasks[i], this.opts.type); }

    finishTask() {
      const i = this.task, o = this.opts, samples = this.tasks[i];
      this.lossAfterOwn[i] = this.taskLoss(i);
      if (o.method === 'replay') this.replay.push(...this.sampleBatch(samples, o.replayPerTask));
      if (o.method === 'ogd') {
        // remember the output gradients ∂f_k(x)/∂θ on (up to 16) points of this task
        for (const s of this.sampleBatch(samples, 16)) {
          for (const g of this.net.paramJacobian(s.x)) this.addOgdDirection(g);
        }
      }
      if (o.method === 'ewc') this.ewc.push({ fisher: this.fisherDiag(samples), anchor: Float64Array.from(this.net.theta) });
      this.taskStep = 0;
      if (++this.task >= this.tasks.length) { this.task = this.tasks.length - 1; this.done = true; }
    }

    /**
     * Diagonal Fisher information of the model on `samples`.  Uses the model's own output
     * distribution (not the observed loss gradient, which is ~0 once a task is learned):
     *   MSE (Gaussian output):  F = mean_x Σ_k (∂f_k(x)/∂θ)²
     *   cross-entropy:          F = mean_x Σ_c p_c(x) (∂ log p_c(x)/∂θ)²
     */
    fisherDiag(samples) {
      const net = this.net, F = new Float64Array(net.nParams);
      const subset = this.sampleBatch(samples, 64);
      for (const s of subset) {
        if (this.opts.type === 'ce') {
          const cache = net.forward(s.x), out = cache.as[cache.as.length - 1];
          const p = crossEntropy(out, 0).grad; p[0] += 1;           // softmax probabilities
          for (let c = 0; c < out.length; c++) {
            const g = new Float64Array(net.nParams);
            net.backward(cache, crossEntropy(out, c).grad, g);
            for (let k = 0; k < F.length; k++) F[k] += (p[c] * g[k] * g[k]) / subset.length;
          }
        } else {
          for (const g of net.paramJacobian(s.x)) for (let k = 0; k < F.length; k++) F[k] += (g[k] * g[k]) / subset.length;
        }
      }
      return F;
    }

    addOgdDirection(g) {
      const v = Float64Array.from(g);
      for (let pass = 0; pass < 2; pass++) {       // Gram–Schmidt, twice for numerical stability
        for (const b of this.ogdBasis) { const c = dot(v, b); for (let k = 0; k < v.length; k++) v[k] -= c * b[k]; }
      }
      const n = norm(v);
      if (n > 1e-8 * Math.max(1, norm(g)) && this.ogdBasis.length < this.net.nParams) {
        for (let k = 0; k < v.length; k++) v[k] /= n;
        this.ogdBasis.push(v);
      }
    }

    /** Remove the components of `grad` along the stored OGD basis (in place). */
    projectOut(grad) {
      for (const b of this.ogdBasis) { const c = dot(grad, b); for (let k = 0; k < grad.length; k++) grad[k] -= c * b[k]; }
    }

    /** Adds ∂/∂θ of λ/2 Σ_t Σ_i F_t,i (θ_i − θ*_t,i)² to grad; returns the penalty value. */
    addEwcGrad(grad) {
      const lam = this.opts.ewcLambda, th = this.net.theta;
      let pen = 0;
      for (const { fisher, anchor } of this.ewc) {
        for (let k = 0; k < grad.length; k++) {
          const d = th[k] - anchor[k];
          grad[k] += lam * fisher[k] * d;
          pen += 0.5 * lam * fisher[k] * d * d;
        }
      }
      return pen;
    }

    /** Forgetting F_i = L_i(now) − L_i(right after training on task i), for finished tasks. */
    forgetting() {
      return this.tasks.map((_, i) => (this.lossAfterOwn[i] === undefined ? null : {
        after: this.lossAfterOwn[i], now: this.taskLoss(i), F: this.taskLoss(i) - this.lossAfterOwn[i],
      }));
    }
  }

  // ===========================================================================
  // Analysis
  // ===========================================================================

  /** Cosine similarity of per-sample loss gradients: how much training on i moves the loss on j. */
  function gradientCosine(net, samples, type) {
    const gs = samples.map((s) => batchLossGrad(net, [s], type).grad);
    const ns = gs.map(norm);
    return gs.map((gi, i) => gs.map((gj, j) => (ns[i] * ns[j] > 1e-20 ? dot(gi, gj) / (ns[i] * ns[j]) : 0)));
  }

  /** Neural tangent kernel K(x_i, x_j) = J_i J_jᵀ (outDim × outDim); returned as its trace. */
  function ntkMatrix(net, xs) {
    const Js = xs.map((x) => net.paramJacobian(x));
    return Js.map((Ji) => Js.map((Jj) => Ji.reduce((s, row, k) => s + dot(row, Jj[k]), 0)));
  }

  /** Largest singular value of a matrix (rows). */
  const spectralNorm = (M) => singularValues(M)[0] || 0;

  /**
   * Lipschitz estimates of x ↦ f(x) on the box [-extent, extent]^d:
   *   empirical — max ‖f(x)−f(y)‖/‖x−y‖ over random nearby pairs (a lower bound);
   *   bound     — Π_l ‖W_l‖₂ · max|act_l'| (an upper bound).
   */
  function lipschitz(net, rng, extent = 2, pairs = 1500, delta = 0.02) {
    const d = net.inDim;
    let emp = 0;
    for (let i = 0; i < pairs; i++) {
      const x = Float64Array.from([...Array(d)].map(() => rng.uniform(-extent, extent)));
      const u = Float64Array.from([...Array(d)].map(() => rng.normal()));
      const un = norm(u);
      const y = x.map((v, k) => v + (delta * u[k]) / un);
      const fx = net.predict(x), fy = net.predict(y);
      let s = 0;
      for (let k = 0; k < fx.length; k++) s += (fx[k] - fy[k]) ** 2;
      emp = Math.max(emp, Math.sqrt(s) / delta);
    }
    let bound = 1;
    for (let l = 0; l < net.nLayers; l++) bound *= spectralNorm(net.weightMatrix(l)) * ACTIVATIONS[net.acts[l]].maxSlope;
    return { empirical: emp, bound };
  }

  /**
   * Per-layer invertibility report on a set of input points:
   *   rank of W, its singular values, the fraction of points whose activation is in a flat /
   *   clipped region (ReLU: some z_i < 0 → dimension collapse; sigmoid/tanh: saturated, |z| > 4),
   *   and whether the activation is globally injective.
   */
  function layerReport(net, xs) {
    const rep = [];
    const caches = xs.map((x) => net.forward(x));
    for (let l = 0; l < net.nLayers; l++) {
      const L = net.layout[l], act = net.acts[l];
      const sv = singularValues(net.weightMatrix(l));
      const rank = rankOf(sv);
      let flat = 0;
      for (const c of caches) {
        const z = c.zs[l];
        let hit = false;
        for (let i = 0; i < z.length; i++) {
          if ((act === 'relu' && z[i] <= 0) || ((act === 'sigmoid' || act === 'tanh') && Math.abs(z[i]) > 4)) hit = true;
        }
        if (hit) flat++;
      }
      const warnings = [];
      if (L.nout < net.inDim) warnings.push(`width ${L.nout} < input dim ${net.inDim}: dimensions are lost`);
      if (rank < Math.min(L.nin, L.nout)) warnings.push('W is rank-deficient: it squashes space onto a lower-dimensional subspace');
      if (act === 'relu' && flat > 0) warnings.push(`ReLU clips ${(100 * flat / xs.length).toFixed(0)}% of points onto faces/axes (not invertible there)`);
      if ((act === 'sigmoid' || act === 'tanh') && flat > 0) warnings.push(`${(100 * flat / xs.length).toFixed(0)}% of points are saturated (nearly flat)`);
      if (act === 'sin' || act === 'gelu') warnings.push(`${act} is not monotonic: it can fold space onto itself`);
      rep.push({ layer: l + 1, nin: L.nin, nout: L.nout, act, sv, rank, flatFrac: flat / Math.max(1, xs.length), warnings });
    }
    return rep;
  }

  const NN = {
    Rng, ACTIVATIONS, ACTIVATION_NAMES, MLP, mse, crossEntropy, sampleLoss, batchLossGrad, meanLoss,
    SGD, Adam, dot, norm, symEig, singularValues, rankOf, pcaBasis, makeDataset, targetMatrix, matVec,
    makeTransformData, fibonacciSphere, SHAPES, makeShape, morphData, squareSvd, clampSingularValues, det, orientPositive, Trainer, gradientCosine, ntkMatrix, spectralNorm, lipschitz, layerReport,
  };
  if (typeof module !== 'undefined' && module.exports) module.exports = NN;
  else root.NN = NN;
})(typeof window !== 'undefined' ? window : globalThis);
