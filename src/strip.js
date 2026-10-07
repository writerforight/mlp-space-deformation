/* The model strip: the network drawn as the path a point travels.

     Input ─▶ [ W₁ | tanh ] ─▶ [ W₂ | tanh ] ─▶ [ W₃ | linear ] ─▶ Output   − ＋

   Every half of a block is one stage of the animation (W = the linear step, the right half = the
   activation).  Click a stage to go there, drag along the strip to scrub, ⓘ opens the layer inspector.
   A marker shows where the view is right now, also between stages.  The loss sparkline next to ▶ Train
   shows the last few hundred training steps (log scale).
*/
(function () {
  'use strict';
  const App = window.__app;
  const $ = (id) => document.getElementById(id);
  const SUB = '₀₁₂₃₄₅₆₇₈₉';
  const sub = (k) => String(k).split('').map((c) => SUB[+c]).join('');
  const pipe = $('pipe');
  let sig = '';
  let centers = [];             // x of every stage's slot, relative to the pipe

  function build() {
    const net = App.net;
    sig = net.acts.join() + '|' + net.dims.join();
    let h = '<button class="pn io" data-stage="0">Input</button>';
    net.acts.forEach((a, l) => {
      const last = l === net.nLayers - 1;
      h += `<span class="pa">▸</span><span class="pb" data-layer="${l + 1}">
        <button class="pn w" data-stage="${2 * l + 1}" data-tip="Layer ${l + 1}, linear step: z = W${l + 1} a + b${l + 1} (${net.dims[l]}→${net.dims[l + 1]})">W${sub(l + 1)}</button><button class="pn s" data-stage="${2 * l + 2}" data-tip="Layer ${l + 1}, activation: a = ${a}(z)">${a === 'identity' ? 'linear' : a}</button><button class="pi" data-layer="${l + 1}" data-tip="Inspect layer ${l + 1}: weights, activation, history">ⓘ</button></span>`;
      if (last) h += '<span class="pa">▸</span><button class="pn io" data-stage="out">Output</button>';
    });
    h += `<span class="pm"><button id="layerMinus" data-tip="Remove the last layer">−</button><button id="layerPlus" data-tip="Add a layer">＋</button></span><div class="marker"></div>`;
    pipe.innerHTML = h;
    $('layerMinus').disabled = net.nLayers <= 1;
    $('layerPlus').disabled = net.nLayers >= 8;
    measure();
  }

  /** x centre of every stage slot: 0 = Input, 2l-1 = W_l, 2l = σ_l; the last stage also sits on Output. */
  function measure() {
    const box = pipe.getBoundingClientRect();
    const mid = (el) => { const r = el.getBoundingClientRect(); return r.left + r.width / 2 - box.left + pipe.scrollLeft; };
    centers = [...pipe.querySelectorAll('.pn[data-stage]')].filter((b) => b.dataset.stage !== 'out').map(mid);
  }

  function stageAtX(x) {
    if (x <= centers[0]) return 0;
    for (let i = 0; i < centers.length - 1; i++) {
      if (x <= centers[i + 1]) return i + (x - centers[i]) / (centers[i + 1] - centers[i]);
    }
    return centers.length - 1;
  }

  function setLayers(n) {
    const el = $('layers');
    el.value = n;
    el.dispatchEvent(new Event('input', { bubbles: true }));
  }

  // click a slot: go there; ⓘ: inspector; − / ＋: network size
  pipe.addEventListener('click', (e) => {
    const info = e.target.closest('.pi');
    if (info) { App.selectLayer(+info.dataset.layer, false); return; }
    if (e.target.id === 'layerPlus') { setLayers(App.net.nLayers + 1); return; }
    if (e.target.id === 'layerMinus') { setLayers(App.net.nLayers - 1); return; }
    const slot = e.target.closest('.pn[data-stage]');
    if (!slot || dragged) return;
    const s = slot.dataset.stage === 'out' ? 2 * App.net.nLayers : +slot.dataset.stage;
    App.animateTo(s);
    if (App.inspect && s > 0) App.inspect.layer = Math.ceil(s / 2);
  });

  // drag along the strip to scrub through the stages
  let drag = null, dragged = false;
  pipe.addEventListener('pointerdown', (e) => {
    if (e.target.closest('.pi, .pm')) return;
    drag = { x0: e.clientX };
    dragged = false;
  });
  window.addEventListener('pointermove', (e) => {
    if (!drag) return;
    if (!dragged && Math.abs(e.clientX - drag.x0) < 4) return;
    dragged = true;
    const box = pipe.getBoundingClientRect();
    App.scrubTo(stageAtX(e.clientX - box.left + pipe.scrollLeft));
  });
  window.addEventListener('pointerup', () => { drag = null; setTimeout(() => { dragged = false; }, 0); });
  window.addEventListener('resize', () => requestAnimationFrame(measure));

  // ---- loss sparkline -----------------------------------------------------------------------------------
  const spark = $('lossSpark');
  let sparkKey = '';
  function drawSpark() {
    const T = App.trainer, hist = T ? T.lossHistory : [];
    const key = `${hist.length}|${spark.clientWidth}`;
    if (key === sparkKey) return;
    sparkKey = key;
    const w = spark.clientWidth, h = spark.clientHeight, dpr = window.devicePixelRatio || 1;
    if (!w) return;
    spark.width = Math.round(w * dpr); spark.height = Math.round(h * dpr);
    const ctx = spark.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const ys = hist.slice(-400).filter((v) => v > 0 && Number.isFinite(v)).map(Math.log10);
    $('lossVal').textContent = hist.length ? `loss ${hist[hist.length - 1].toPrecision(2)}` : 'loss –';
    if (ys.length < 2) return;
    const lo = Math.min(...ys), hi = Math.max(...ys), span = hi - lo || 1;
    ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 1.4;
    ctx.beginPath();
    ys.forEach((v, i) => {
      const x = (i / (ys.length - 1)) * (w - 2) + 1, y = h - 2 - ((v - lo) / span) * (h - 4);
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    });
    ctx.stroke();
  }

  // ---- every frame: rebuild if the network changed, move the marker, light up the current slot ----------
  function frame() {
    const net = App.net;
    if (net && net.acts.join() + '|' + net.dims.join() !== sig) build();
    if (centers.length) {
      const t = App.t, n = centers.length - 1;
      const i = Math.min(n - 1, Math.floor(t)), u = t - i;
      const x = n === 0 ? centers[0] : centers[i] + u * (centers[Math.min(n, i + 1)] - centers[i]);
      pipe.querySelector('.marker').style.transform = `translateX(${x}px)`;
      const s = Math.round(t), near = Math.abs(t - s) < 0.02;
      pipe.querySelectorAll('.pn[data-stage]').forEach((b) => {
        const k = b.dataset.stage === 'out' ? n : +b.dataset.stage;
        b.classList.toggle('now', near && k === s);
      });
    }
    drawSpark();
    requestAnimationFrame(frame);
  }
  build();
  requestAnimationFrame(frame);
})();
