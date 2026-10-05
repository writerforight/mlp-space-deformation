// Reproduces the numbers of the "Topology experiments" panel:  node test/topology_experiments.js
// Every experiment uses the app's settings (tanh, linear output, Xavier init, homeomorphism mode =
// smallest singular value ≥ 0.2 with an orientation-preserving start, Adam, batch 64) and is run at
// width d and width d + 1 for several seeds.  "worst" is the largest distance from a target on a 50×
// finer copy of the shape than the 400 training points: a network can pass a curve *between* two
// training points, a tear the training points never see.
const NN = require('../src/nn.js');
const anchor = (c, d) => { const v = new Float64Array(d); v[0] = c === 0 ? 1 : -1; return v; };
const sep = (shape, d) => (n) => NN.makeShape(shape, n).components.flatMap((c) => c.pts.map((x) => ({ x, y: anchor(c.label, d) })));
const morph = (a, b) => (n) => NN.morphData(a, b, n);
//   name, d, data, layers, lr, steps, seeds
const EXPS = [
  ['2D control: circle → square', 2, morph('circle', 'square'), 4, 0.01, 8000, [1, 2, 3, 4]],
  ['2D: disk out of the ring', 2, sep('diskInRing', 2), 4, 0.01, 8000, [1, 2, 3, 4]],
  ['2D: two circles → one', 2, morph('twoCircles', 'oneCircleTwice'), 4, 0.01, 8000, [1, 2, 3, 4]],
  ['2D: circle → figure eight', 2, morph('circle', 'figure8'), 4, 0.01, 8000, [1, 2, 3, 4]],
  ['3D control: sphere → ellipsoid', 3, morph('sphere', 'ellipsoid'), 4, 0.01, 8000, [1, 2, 3, 4]],
  ['3D: unlink the rings', 3, sep('linkedRings', 3), 8, 0.003, 16000, [1, 2, 3, 4, 5, 6, 7, 8]],
  ['3D: ball out of the shell', 3, sep('nestedSpheres', 3), 6, 0.003, 16000, [1, 2, 3, 4]],
  ['3D: unknot → trefoil', 3, morph('unknot', 'trefoil'), 4, 0.01, 8000, [1, 2, 3, 4]],
  ['3D: torus → sphere', 3, morph('torus', 'torusOnSphere'), 4, 0.01, 8000, [1, 2, 3, 4]],
];
const worstOn = (net, set) => Math.max(...set.map((s) => { const o = net.predict(s.x); return Math.hypot(...o.map((v, k) => v - s.y[k])); }));
function run(train, dense, d, width, layers, lr, steps, seed) {
  const net = new NN.MLP([d, ...Array(layers - 1).fill(width), d], [...Array(layers - 1).fill('tanh'), 'identity']).init('xavier', 1, new NN.Rng(seed));
  NN.orientPositive(net);
  NN.clampSingularValues(net, 0.2);
  const tr = new NN.Trainer(net, { type: 'mse', lr, batch: 64, seed, invertibleFloor: 0.2 });
  tr.setTasks([train]);
  for (let i = 0; i < steps; i++) tr.step();
  return worstOn(net, dense);
}
const only = process.argv[2];
for (const [name, d, make, layers, lr, steps, seeds] of EXPS) {
  if (only && !name.includes(only)) continue;
  const train = make(400), dense = make(20000), cells = [];
  for (const width of [d, d + 1]) {
    const r = (width === d ? seeds.slice(0, 4) : seeds).map((seed) => run(train, dense, d, width, layers, lr, steps, seed));
    cells.push(`width ${width}: worst ${r.map((v) => v.toFixed(2)).join(' / ')} → solved ${r.filter((v) => v < 0.3).length}/${r.length}`);
  }
  console.log(name.padEnd(32), '|', cells.join(' | '));
}
