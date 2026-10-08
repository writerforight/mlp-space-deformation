/* The opening scene.

   A real network — six layers, three neurons wide, so every step of it is honest 3D space — is trained
   right here in a few seconds on three spiral arms.  Then it plays: space bends layer by layer until
   each class sits in its own corner, while a few lines of text appear.  A tap (or any key) skips to the
   end, where the buttons lead to the workspace or to the guided start.  Not shown at #workspace.
   Sound (src/sound.js) follows the picture once the visitor allows it (🔊, or any tap): a low drone
   whose filter opens as space untwists, a bell at the end of every layer, a chord when the classes land.
*/
(function () {
  'use strict';
  const root = document.getElementById('intro');
  if (!root) return;
  if (location.hash === '#workspace' || !window.THREE) { root.remove(); return; }

  const reduce = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const COLORS = ['#58a6ff', '#ff6b6b', '#56d364'];
  const K = 3, L = 6, STEPS = 2000;
  const S = 2 * L;                                   // stages: input, then (linear, activation) per layer

  // ---- the data and the network (same generator and seeds as the workspace) ------------------------
  const data = NN.makeDataset('spirals', 450, 3, new NN.Rng(207), 0.08, K);
  const goal = NN.classTargets(K, 3);
  const net = new NN.MLP([3, 3, 3, 3, 3, 3, 3], [...Array(L - 1).fill('tanh'), 'identity']).init('xavier', 1, new NN.Rng(2));
  const trainer = new NN.Trainer(net, { type: 'mse', lr: 0.01, batch: 32 });
  trainer.setTasks([data.map((s) => ({ x: s.x, y: goal[s.y] }))]);

  // a faint lattice of space itself, bent along with the points
  const ticks = [-1.4, -0.7, 0, 0.7, 1.4], lattice = [];
  for (const a of ticks) for (const b of ticks) for (let ax = 0; ax < 3; ax++) {
    const line = [];
    for (let k = 0; k <= 16; k++) {
      const p = [0, 0, 0], u = -1.4 + (2.8 * k) / 16;
      p[ax] = u; p[(ax + 1) % 3] = a; p[(ax + 2) % 3] = b;
      line.push(p);
    }
    lattice.push(line);
  }

  // positions at every stage, each stage scaled to the same size (a uniform zoom: shapes are untouched)
  const sizeOf = (pts) => 1.25 / Math.sqrt(pts.reduce((a, p) => a + p[0] ** 2 + p[1] ** 2 + p[2] ** 2, 0) / pts.length);
  const inputScale = sizeOf(data.map((s) => s.x));   // the input needs no network: shown like this while it trains
  let stagePts = null, stageGrid = null;
  function computeStages() {
    const raw = data.map((s) => net.stages(s.x));
    const scale = [...Array(S + 1)].map((_, k) => sizeOf(raw.map((st) => st[k])));
    stagePts = [...Array(S + 1)].map((_, k) => raw.map((st) => Array.from(st[k], (v) => v * scale[k])));
    stageGrid = [...Array(S + 1)].map((_, k) => lattice.map((line) => line.map((p) => Array.from(net.stages(p)[k], (v) => v * scale[k]))));
  }

  // ---- the scene -----------------------------------------------------------------------------------
  const holder = document.getElementById('introGL');
  let renderer;
  try { renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true }); } catch (err) { root.remove(); return; }
  renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
  holder.appendChild(renderer.domElement);
  const scene = new THREE.Scene(), camera = new THREE.PerspectiveCamera(40, 1, 0.1, 100);

  const dot = (() => {                               // a soft round dot
    const c = document.createElement('canvas'); c.width = c.height = 64;
    const g = c.getContext('2d'), grad = g.createRadialGradient(32, 32, 0, 32, 32, 32);
    grad.addColorStop(0, 'rgba(255,255,255,1)'); grad.addColorStop(0.45, 'rgba(255,255,255,0.9)'); grad.addColorStop(1, 'rgba(255,255,255,0)');
    g.fillStyle = grad; g.fillRect(0, 0, 64, 64);
    return new THREE.CanvasTexture(c);
  })();
  const pointsPos = new Float32Array(data.length * 3), pointsCol = new Float32Array(data.length * 3);
  data.forEach((s, i) => { const c = new THREE.Color(COLORS[s.y]); pointsCol.set([c.r, c.g, c.b], 3 * i); });
  const pg = new THREE.BufferGeometry();
  pg.setAttribute('position', new THREE.BufferAttribute(pointsPos, 3));
  pg.setAttribute('color', new THREE.BufferAttribute(pointsCol, 3));
  scene.add(new THREE.Points(pg, new THREE.PointsMaterial({ size: 0.09, map: dot, vertexColors: true, transparent: true,
    depthWrite: false, blending: THREE.AdditiveBlending })));

  const segs = lattice.length * 16, gridPos = new Float32Array(segs * 6);
  const gg = new THREE.BufferGeometry();
  gg.setAttribute('position', new THREE.BufferAttribute(gridPos, 3));
  const gridMat = new THREE.LineBasicMaterial({ color: 0xc9d1d9, transparent: true, opacity: 0.0 });
  scene.add(new THREE.LineSegments(gg, gridMat));

  // the raw input, shown (and turning) while the network trains
  function setStage(u) {
    const P = stagePts, G = stageGrid;
    const s0 = Math.min(S - 1, Math.floor(u)), f = u - s0, e = f * f * (3 - 2 * f);   // smooth within each step
    const lerp = (a, b, k) => a[k] + e * (b[k] - a[k]);
    for (let i = 0; i < data.length; i++) for (let k = 0; k < 3; k++) pointsPos[3 * i + k] = P ? lerp(P[s0][i], P[s0 + 1][i], k) : data[i].x[k] * inputScale;
    let o = 0;
    lattice.forEach((line, j) => {
      for (let q = 0; q < 16; q++) for (const r of [q, q + 1]) for (let k = 0; k < 3; k++) {
        gridPos[o++] = G ? lerp(G[s0][j][r], G[s0 + 1][j][r], k) : line[r][k] * inputScale;
      }
    });
    pg.attributes.position.needsUpdate = true;
    gg.attributes.position.needsUpdate = true;
  }

  function resize() {
    const w = root.clientWidth, h = root.clientHeight;
    renderer.setSize(w, h, false);
    renderer.domElement.style.width = '100%'; renderer.domElement.style.height = '100%';
    camera.aspect = w / h;
    dist = w < h ? 8.6 : 7.4;                        // portrait screens step back a little
    camera.updateProjectionMatrix();
  }
  let dist = 7.4;
  window.addEventListener('resize', resize);
  resize();

  // ---- the script: lines of text, then the network plays --------------------------------------------
  const line = (n, on) => root.querySelector(`[data-l="${n}"]`).classList.toggle('show', on);
  const at = (ms, f) => { const id = setTimeout(f, ms); timers.push(id); };
  const timers = [];
  let t0 = performance.now(), morphFrom = null, ended = false, trained = false, raf = 0;
  const MORPH = 7000, HOLD = 2500, BACK = 5000;

  function startMorph() {
    if (!trained) { at(120, startMorph); return; }
    morphFrom = performance.now();
  }
  function finish() {                                // the end state: title, buttons, the scene keeps playing
    if (ended) return;
    ended = true;
    timers.forEach(clearTimeout);
    [1, 2, 3, 4].forEach((n) => line(n, false));
    root.classList.add('ended');
    if (trained && morphFrom === null) morphFrom = performance.now() - MORPH;   // skipped: jump to the bent space
  }
  if (reduce) { root.classList.add('still'); }
  at(500, () => line(1, true));
  at(2500, () => line(2, true));
  at(4200, () => { line(1, false); line(2, false); startMorph(); });
  at(4600, () => line(3, true));
  at(8600, () => { line(3, false); line(4, true); });
  at(11800, finish);

  // train in small slices, so the opening keeps turning smoothly
  (function slice() {
    const end = performance.now() + 12;
    while (trainer.step_ < STEPS && performance.now() < end) trainer.step();
    if (trainer.step_ < STEPS) { setTimeout(slice, 0); return; }
    computeStages();
    trained = true;
    if (reduce) { morphFrom = performance.now() - MORPH; finish(); }
  })();

  // ---- sound: browsers allow it only after a tap -----------------------------------------------------
  const chip = root.querySelector('.intro-sound');
  let prevU = 0, passes = 0;
  const showChip = () => {
    const playing = Sound.on && Sound.ready;
    chip.classList.toggle('on', playing);
    chip.textContent = playing ? '🔊 On' : Sound.on ? '🔊 Sound' : '🔇 Off';
  };
  function startSound() { if (Sound.on) { Sound.unlock(); Sound.droneStart(); } showChip(); }
  chip.onclick = (e) => {
    e.stopPropagation();
    if (Sound.on && Sound.ready) { Sound.on = false; } else { Sound.on = true; startSound(); }
    showChip();
  };
  showChip();
  function soundFrame(u) {
    if (!Sound.ready) { prevU = u; return; }
    Sound.droneOpen(u / S);
    if (u > prevU) {                                 // going forward: a bell at the end of each layer, a chord at the end
      const s2 = 2 * Math.floor(u / 2 + 1e-9);
      if (s2 >= 2 && s2 > prevU + 1e-9) {
        if (s2 >= S) { Sound.resolve(passes ? 0.05 : 0.09); passes++; } else Sound.layer(s2 / 2 - 1, passes ? 0.04 : 0.07);
      }
    }
    prevU = u;
  }

  function frame(now) {
    raf = requestAnimationFrame(frame);
    const age = (now - t0) / 1000;
    gridMat.opacity = Math.min(0.13, age * 0.05);
    // stage: 0 until the play starts, then input → output, a pause, back to the input, and again
    let u = 0;
    if (morphFrom !== null && stagePts) {
      const m = (now - morphFrom) % (MORPH + HOLD + BACK + HOLD);
      const ease = (x) => 0.5 - 0.5 * Math.cos(Math.PI * Math.min(1, Math.max(0, x)));
      u = m < MORPH ? S * ease(m / MORPH) : m < MORPH + HOLD ? S : m < MORPH + HOLD + BACK ? S * (1 - ease((m - MORPH - HOLD) / BACK)) : 0;
    }
    setStage(u);
    soundFrame(u);
    const a = reduce ? 0.7 : 0.7 + age * 0.12;      // a slow turn around the vertical, seen a little from above
    camera.position.set(dist * Math.sin(a) * 0.98, dist * 0.2, dist * Math.cos(a) * 0.98);
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
  }
  raf = requestAnimationFrame(frame);

  // ---- leaving -------------------------------------------------------------------------------------
  function close(then) {
    root.classList.add('leaving');
    Sound.droneStop(1.2);
    setTimeout(() => {
      cancelAnimationFrame(raf);
      timers.forEach(clearTimeout);
      window.removeEventListener('resize', resize);
      renderer.dispose();
      root.remove();
      window.dispatchEvent(new Event('resize'));
    }, 650);
    then();
  }
  const click = (id) => { const el = document.getElementById(id); if (el) el.click(); };
  document.getElementById('introTry').onclick = (e) => { e.stopPropagation(); close(() => click('wzSkip')); };
  document.getElementById('introGuide').onclick = (e) => { e.stopPropagation(); close(() => click('wzStart')); };
  root.addEventListener('click', (e) => { if (!e.target.closest('button')) { startSound(); finish(); } });
  root.querySelector('.intro-skip').onclick = (e) => { e.stopPropagation(); startSound(); finish(); };
  window.addEventListener('keydown', function key(e) {
    if (!document.body.contains(root)) { window.removeEventListener('keydown', key); return; }
    if (e.key === 'Escape' || e.key === ' ' || e.key === 'Enter') { e.preventDefault(); finish(); }
  });
})();
