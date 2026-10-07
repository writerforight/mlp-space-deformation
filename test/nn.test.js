// Run: node test/nn.test.js   — checks the math in src/nn.js against finite differences.
const NN = require('../src/nn.js');
let failed = 0;
const check = (name, ok, extra = '') => { console.log(`${ok ? 'ok  ' : 'FAIL'} ${name} ${extra}`); if (!ok) failed++; };
const relErr = (a, b) => Math.abs(a - b) / Math.max(1e-6, Math.abs(a) + Math.abs(b));

// 1) backprop gradient vs. central finite differences, every activation, both losses
for (const act of NN.ACTIVATION_NAMES) {
  for (const type of ['mse', 'ce']) {
    const rng = new NN.Rng(3);
    const net = new NN.MLP([2, 4, 3, 2], [act, act, 'identity']).init('normal', 0.8, rng);
    const batch = [{ x: [0.3, -0.7], y: type === 'ce' ? 1 : [0.2, 0.5] }, { x: [-1.1, 0.4], y: type === 'ce' ? 0 : [-0.3, 0.9] }];
    const { grad } = NN.batchLossGrad(net, batch, type);
    let worst = 0;
    for (let k = 0; k < net.nParams; k++) {
      const h = 1e-6, t0 = net.theta[k];
      net.theta[k] = t0 + h; const lp = NN.batchLossGrad(net, batch, type).loss;
      net.theta[k] = t0 - h; const lm = NN.batchLossGrad(net, batch, type).loss;
      net.theta[k] = t0;
      const fd = (lp - lm) / (2 * h);
      if (Math.abs(fd) + Math.abs(grad[k]) > 1e-7) worst = Math.max(worst, relErr(fd, grad[k]));
    }
    check(`backprop ${act}/${type}`, worst < 1e-5, `max rel err ${worst.toExponential(1)}`);
  }
}

// 1b) mixed batch: per-sample loss type and weight (several goals trained together)
{
  const net = new NN.MLP([2, 5, 2], ['tanh', 'identity']).init('normal', 0.8, new NN.Rng(9));
  const batch = [{ x: [0.3, -0.7], y: 1, type: 'ce', w: 2 }, { x: [-1.1, 0.4], y: [-0.3, 0.9], w: 0.5 }, { x: [0.5, 0.5], y: [0.1, 0.1] }];
  const { grad, loss } = NN.batchLossGrad(net, batch, 'mse');
  let worst = 0;
  for (let k = 0; k < net.nParams; k++) {
    const h = 1e-6, t0 = net.theta[k];
    net.theta[k] = t0 + h; const lp = NN.batchLossGrad(net, batch, 'mse').loss;
    net.theta[k] = t0 - h; const lm = NN.batchLossGrad(net, batch, 'mse').loss;
    net.theta[k] = t0;
    const fd = (lp - lm) / (2 * h);
    if (Math.abs(fd) + Math.abs(grad[k]) > 1e-7) worst = Math.max(worst, relErr(fd, grad[k]));
  }
  check('backprop mixed ce/mse with weights', worst < 1e-5, `max rel err ${worst.toExponential(1)}`);
  const byHand = (2 * NN.sampleLoss(net, { ...batch[0], w: 1 }, 'ce') + 0.5 * NN.sampleLoss(net, { ...batch[1], w: 1 }, 'mse')
    + NN.sampleLoss(net, batch[2], 'mse')) / 3;
  check('mixed batch loss = mean of weighted sample losses', Math.abs(byHand - loss) < 1e-12);
}

// 2) input Jacobians of every stage vs. finite differences (3D, wide layer)
{
  const rng = new NN.Rng(5);
  const net = new NN.MLP([3, 5, 3], ['tanh', 'gelu']).init('he', 1, rng);
  const x = [0.2, -0.4, 0.9];
  const { stages, jacs } = net.stageJacobians(x);
  let worst = 0;
  for (let s = 0; s < stages.length; s++) {
    for (let j = 0; j < 3; j++) {
      const h = 1e-6, xp = x.slice(), xm = x.slice(); xp[j] += h; xm[j] -= h;
      const sp = net.stages(xp)[s], sm = net.stages(xm)[s];
      for (let i = 0; i < sp.length; i++) worst = Math.max(worst, Math.abs((sp[i] - sm[i]) / (2 * h) - jacs[s][i][j]));
    }
  }
  check('stage Jacobians', worst < 1e-6, `max abs err ${worst.toExponential(1)}`);

  // parameter Jacobian ∂f_k/∂θ
  const PJ = net.paramJacobian(x);
  let w2 = 0;
  for (let k = 0; k < net.nParams; k += 3) {
    const h = 1e-6, t0 = net.theta[k];
    net.theta[k] = t0 + h; const fp = net.predict(x); net.theta[k] = t0 - h; const fm = net.predict(x); net.theta[k] = t0;
    for (let o = 0; o < 3; o++) w2 = Math.max(w2, Math.abs((fp[o] - fm[o]) / (2 * h) - PJ[o][k]));
  }
  check('parameter Jacobian', w2 < 1e-6, `max abs err ${w2.toExponential(1)}`);
}

// 3) linear algebra
{
  const A = [[4, 1, 2], [1, 3, 0.5], [2, 0.5, 5]];
  const { values, vectors } = NN.symEig(A);
  let worst = 0;
  values.forEach((lam, k) => {
    const v = vectors[k], Av = NN.matVec(A, v);
    for (let i = 0; i < 3; i++) worst = Math.max(worst, Math.abs(Av[i] - lam * v[i]));
  });
  check('symmetric eigen-decomposition', worst < 1e-10 && values[0] >= values[1] && values[1] >= values[2]);
  const sv = NN.singularValues([[3, 0], [0, -2], [0, 0]]);
  check('singular values', Math.abs(sv[0] - 3) < 1e-10 && Math.abs(sv[1] - 2) < 1e-10);
  check('rank of a rank-1 matrix', NN.rankOf(NN.singularValues([[1, 2], [2, 4]])) === 1);
  const pts = []; const r = new NN.Rng(1);
  for (let i = 0; i < 300; i++) { const t = r.normal(); pts.push([t, 2 * t + 0.01 * r.normal(), -t]); }
  const P = NN.pcaBasis(pts, 2);
  const dir = [1, 2, -1].map((v) => v / Math.sqrt(6));
  check('PCA finds the main direction', Math.abs(NN.dot(P.basis[0], dir)) > 0.999 && P.explained > 0.99);
}

// 4) training actually learns
{
  const rng = new NN.Rng(2);
  const net = new NN.MLP([2, 8, 8, 2], ['tanh', 'tanh', 'identity']).init('xavier', 1, rng);
  const data = NN.makeDataset('moons', 300, 2, new NN.Rng(9));
  const tr = new NN.Trainer(net, { type: 'ce', lr: 0.03, batch: 32 });
  tr.setTasks([data]);
  for (let i = 0; i < 1500; i++) tr.step();
  const acc = data.filter((s) => { const o = net.predict(s.x); return (o[1] > o[0] ? 1 : 0) === s.y; }).length / data.length;
  check('learns two moons', acc > 0.95, `accuracy ${(100 * acc).toFixed(1)}%`);

  const net2 = new NN.MLP([2, 2], ['identity']).init('normal', 1, new NN.Rng(4));
  const A = NN.targetMatrix('rotation', 2, 0.8);
  const tr2 = new NN.Trainer(net2, { type: 'mse', lr: 0.05, batch: 32 });
  tr2.setTasks([NN.makeTransformData(A, 200, 2, new NN.Rng(1))]);
  for (let i = 0; i < 800; i++) tr2.step();
  const W = net2.weightMatrix(0);
  const err = Math.max(...[0, 1].flatMap((i) => [0, 1].map((j) => Math.abs(W[i][j] - A[i][j]))));
  check('linear layer recovers a rotation', err < 1e-2, `max |W − A| ${err.toExponential(1)}`);
}

// 5) sequential training and anti-forgetting on pinned points
{
  // well-separated pins (for close, conflicting pins OGD's first-order guarantee breaks down — by design)
  const pins = [[[1, 0], [0, 1]], [[-1, 0.2], [0.5, -1]], [[0, -1], [-1, -0.3]], [[0.3, 1], [1, 0.4]]];
  const tasks = pins.map(([x, y]) => [{ x, y }]);
  const run = (method) => {
    const net = new NN.MLP([2, 6, 6, 2], ['tanh', 'tanh', 'identity']).init('xavier', 1, new NN.Rng(11));
    const tr = new NN.Trainer(net, { type: 'mse', lr: method === 'ogd' ? 0.05 : 0.02, optimizer: method === 'ogd' ? 'sgd' : 'adam',
      batch: 1, mode: 'sequential', stepsPerTask: 400, method, ewcLambda: 500, seed: 3 });
    tr.setTasks(tasks);
    while (tr.step() !== null);
    const F = tr.forgetting();
    return F.slice(0, -1).reduce((s, f) => s + Math.max(0, f.F), 0);
  };
  const base = run('none');
  for (const m of ['replay', 'ogd', 'ewc']) {
    const f = run(m);
    check(`${m} reduces forgetting`, f < 0.25 * base, `forgetting ${f.toFixed(4)} vs none ${base.toFixed(4)}`);
  }
  // strong EWC with plain SGD must not blow up (gradient clipping)
  {
    const net = new NN.MLP([2, 6, 6, 2], ['tanh', 'tanh', 'identity']).init('xavier', 1, new NN.Rng(11));
    const tr = new NN.Trainer(net, { type: 'mse', lr: 0.05, optimizer: 'sgd', batch: 1, mode: 'sequential',
      stepsPerTask: 300, method: 'ewc', ewcLambda: 500, seed: 3 });
    tr.setTasks(tasks);
    while (tr.step() !== null);
    check('EWC + SGD stays finite', !tr.diverged && net.theta.every(Number.isFinite));
  }
  // OGD: a projected step does not change old outputs to first order
  const net = new NN.MLP([2, 4, 2], ['tanh', 'identity']).init('normal', 1, new NN.Rng(1));
  const tr = new NN.Trainer(net, { method: 'ogd', mode: 'sequential' });
  for (const g of net.paramJacobian([0.5, 0.5])) tr.addOgdDirection(g);
  const g = NN.batchLossGrad(net, [{ x: [-0.5, 0.2], y: [1, 1] }], 'mse').grad;
  tr.projectOut(g);
  const J = net.paramJacobian([0.5, 0.5]);
  check('OGD-projected gradient is orthogonal to old output gradients', Math.max(...J.map((r) => Math.abs(NN.dot(r, g)))) < 1e-10);
}

// 5b) datasets: balanced, finite, and the rule-based ones obey their rule
{
  for (const dim of [2, 3]) {
    for (const k of ['blobs', 'moons', 'circles', 'rings', 'xor', 'checker', 'wave', 'spirals', 'linked']) {
      const d = NN.makeDataset(k, 200, dim, new NN.Rng(4));
      const ok = d.length === 200 && d.every((s) => s.x.length === dim && s.x.every(Number.isFinite))
        && d.filter((s) => s.y === 0).length === 100;
      if (!ok) check(`dataset ${k} ${dim}D`, false);
    }
  }
  const xor = NN.makeDataset('xor', 200, 2, new NN.Rng(1));
  check('xor labels follow the sign rule', xor.every((s) => (s.x[0] * s.x[1] > 0 ? 0 : 1) === s.y));
  const x3 = NN.makeDataset('xor', 200, 3, new NN.Rng(1));
  check('3D xor labels follow sign parity', x3.every((s) => (s.x[0] * s.x[1] * s.x[2] > 0 ? 0 : 1) === s.y));
  // linked rings: ring 0 pierces the disk of ring 1 and vice versa (that's what makes them a link)
  const L = NN.makeDataset('linked', 400, 3, new NN.Rng(2), 0);
  const r0 = L.filter((s) => s.y === 0), r1 = L.filter((s) => s.y === 1);
  const crosses = (ring, plane, centre) => ring.some((s) => Math.abs(s.x[plane]) < 0.02 && Math.hypot(...[0, 1, 2].filter((k) => k !== plane).map((k) => s.x[k] - centre[k])) < 1);
  check('linked rings pierce each other', crosses(r0, 1, [0.5, 0, 0]) && crosses(r1, 2, [-0.5, 0, 0]));
  check('all datasets generated', true);
}

// 5c) square SVD and homeomorphism mode
{
  const M = [[2, 1, 0], [0, 1e-4, 0], [1, 0, 3]];
  const { U, s, V } = NN.squareSvd(M);
  let err = 0;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) { let v = 0; for (let k = 0; k < 3; k++) v += U[k][i] * s[k] * V[k][j]; err = Math.max(err, Math.abs(v - M[i][j])); }
  check('square SVD reconstructs the matrix', err < 1e-10, `err ${err.toExponential(1)}`);
  const net = new NN.MLP([2, 2, 2, 2], ['tanh', 'tanh', 'identity']).init('normal', 1, new NN.Rng(3));
  net.theta.set([1, 2, 2, 4], net.layout[0].w);                   // a singular first layer
  NN.clampSingularValues(net, 0.2);
  const ok = [0, 1, 2].every((l) => NN.singularValues(net.weightMatrix(l)).every((v) => v >= 0.2 - 1e-9));
  check('homeomorphism mode keeps every square layer invertible', ok);
  const tr = new NN.Trainer(net, { type: 'mse', lr: 0.05, batch: 32, invertibleFloor: 0.2 });
  tr.setTasks([NN.morphData('circle', 'figure8')]);
  for (let i = 0; i < 300; i++) tr.step();
  check('…and stays so during training', [0, 1, 2].every((l) => NN.singularValues(net.weightMatrix(l)).every((v) => v >= 0.2 - 1e-9)));
  const md = NN.morphData('twoCircles', 'oneCircleTwice');
  check('morph data pairs every source point with a target point', md.length === 400 && md.every((s) => s.x.length === 2 && s.y.length === 2));
}

// 6) analysis helpers
{
  const net = new NN.MLP([2, 2, 2], ['relu', 'identity']).init('normal', 1, new NN.Rng(7));
  const L = NN.lipschitz(net, new NN.Rng(1));
  check('Lipschitz: empirical ≤ bound', L.empirical <= L.bound + 1e-9, `${L.empirical.toFixed(3)} ≤ ${L.bound.toFixed(3)}`);
  const C = NN.gradientCosine(net, [{ x: [1, 0], y: [0, 1] }, { x: [0, 1], y: [1, 0] }], 'mse');
  check('gradient cosine diagonal is 1', Math.abs(C[0][0] - 1) < 1e-12 && Math.abs(C[1][1] - 1) < 1e-12);
  const K = NN.ntkMatrix(net, [[1, 0], [0, 1]]);
  check('NTK is symmetric', Math.abs(K[0][1] - K[1][0]) < 1e-12);
  const rep = NN.layerReport(net, [[1, 1], [-1, -1], [1, -1], [-1, 1]]);
  check('layer report flags ReLU collapse', rep[0].warnings.some((w) => w.includes('ReLU')));
  const sph = NN.fibonacciSphere(400);
  check('Fibonacci sphere points have radius 1', sph.every((p) => Math.abs(NN.norm(p) - 1) < 1e-12));
}

console.log(failed ? `\n${failed} test(s) FAILED` : '\nall tests passed');
process.exit(failed ? 1 : 0);
