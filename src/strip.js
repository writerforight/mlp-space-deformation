/* The model strip: the network drawn as the path a point travels.

     Input ─▶ [ W₁ | tanh ] ─▶ [ W₂ | tanh ] ─▶ [ W₃ | linear ] ─▶ Output   − ＋

   Every half of a block is one stage of the animation (W = the linear step, the right half = the
   activation).  Click a stage to go there, drag along the strip to scrub, ⓘ opens the layer inspector.
   A line under the strip grows to where the view is right now, also between stages.  On a phone-wide screen the strip is
   compact — In · 1 · 2 · … · Out, one button per layer (its output); tapping the current layer again opens
   its inspector, and dragging along the strip still passes through every stage.  Once there is training, a video-style
   bar right of Train shows the loss of the whole run: drag on it to see the network at an earlier step,
   ⏵ replays it (at the layer in view), ● live returns.  The big ▶ on the view plays the trained network's layers.
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
  let lastT = -1, lastSigDrawn = '';  // the marker only moves when the stage or the network changes
  const phone = window.matchMedia('(max-width: 600px)');
  let compact = phone.matches;

  function build() {
    const net = App.net;
    sig = net.acts.join() + '|' + net.dims.join();
    if (compact) {
      let h = '<button class="pn io" data-stage="0">In</button>';
      net.acts.forEach((a, l) => {
        h += `<span class="pa">·</span><button class="pn c" data-stage="${2 * l + 2}" data-layer="${l + 1}" data-tip="Layer ${l + 1}: W${l + 1} a + b${l + 1}, then ${a === 'identity' ? 'linear' : a}. Tap again to inspect it.">${l + 1}</button>`;
      });
      pipe.innerHTML = h + '<span class="pa">·</span><button class="pn io" data-stage="out">Out</button><div class="marker"></div>';
      measure();
      return;
    }
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

  /**
   * x centre of every stage slot: 0 = Input, 2l-1 = W_l, 2l = σ_l; the last stage also sits on Output.
   * Stages without a button of their own (the linear steps of the compact strip) sit halfway between.
   */
  function measure() {
    const box = pipe.getBoundingClientRect();
    const mid = (el) => { const r = el.getBoundingClientRect(); return r.left + r.width / 2 - box.left + pipe.scrollLeft; };
    const known = new Map([...pipe.querySelectorAll('.pn[data-stage]')].filter((b) => b.dataset.stage !== 'out').map((b) => [+b.dataset.stage, mid(b)]));
    centers = [...Array(2 * App.net.nLayers + 1)].map((_, k) => known.get(k) ?? (known.get(k - 1) + known.get(k + 1)) / 2);
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
    if (slot.dataset.layer && slot.classList.contains('now')) { App.selectLayer(+slot.dataset.layer, false); return; }
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
  window.addEventListener('resize', () => requestAnimationFrame(() => { measure(); lastT = -1; }));
  phone.addEventListener('change', () => { compact = phone.matches; build(); lastT = -1; });

  // ---- training timeline: the loss over all steps; drag on it to watch the network at an earlier step -----
  const spark = $('lossSpark');
  let sparkKey = '';
  function drawSpark() {
    const T = App.trainer, hist = T ? T.lossHistory : [], steps = App.historySteps, vi = App.viewIdx;
    const key = `${hist.length}|${spark.clientWidth}|${vi}|${steps.length}`;
    if (key === sparkKey) return;
    sparkKey = key;
    $('liveBtn').classList.toggle('hidden', vi === null);
    const show = steps.length >= 2 && hist.length >= 2;            // the bar appears once there is training to watch
    $('timeline').classList.toggle('hidden', !show);
    document.body.classList.toggle('has-timeline', show);
    const w = spark.clientWidth, h = spark.clientHeight, dpr = window.devicePixelRatio || 1;
    if (!w) return;
    spark.width = Math.round(w * dpr); spark.height = Math.round(h * dpr);
    const ctx = spark.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const last = T ? T.step_ : 0, first = last - hist.length;     // lossHistory[k] is the loss of step first + k + 1
    $('lossVal').textContent = vi !== null ? `step ${steps[vi]} / ${steps[steps.length - 1]}`
      : hist.length ? `loss ${hist[hist.length - 1].toPrecision(2)}` : 'loss –';
    if (hist.length < 2) return;
    // one value per pixel column (the mean of the log-loss in that column), over the whole run
    const cols = Math.max(2, Math.floor(w)), ys = new Array(cols).fill(null), cnt = new Array(cols).fill(0);
    for (let k = 0; k < hist.length; k++) {
      const v = hist[k];
      if (!(v > 0) || !Number.isFinite(v)) continue;
      const c = Math.min(cols - 1, Math.floor((k / (hist.length - 1)) * (cols - 1)));
      ys[c] = (ys[c] || 0) + Math.log10(v); cnt[c]++;
    }
    const pts = ys.map((v, c) => (cnt[c] ? [c, v / cnt[c]] : null)).filter(Boolean);
    const lo = Math.min(...pts.map((p) => p[1])), hi = Math.max(...pts.map((p) => p[1])), span = hi - lo || 1;
    ctx.strokeStyle = '#58a6ff'; ctx.lineWidth = 1.4;
    ctx.beginPath();
    pts.forEach(([c, v], i) => {
      const x = (c / (cols - 1)) * (w - 2) + 1, y = h - 2 - ((v - lo) / span) * (h - 4);
      if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y);
    });
    ctx.stroke();
    if (vi !== null && last > first) {                    // where we are in the video
      const x = 1 + ((steps[vi] - first) / (last - first)) * (w - 2);
      ctx.fillStyle = 'rgba(13,17,23,0.55)'; ctx.fillRect(x, 0, w - x, h);   // the "future" is dimmed
      ctx.strokeStyle = '#ff6b6b'; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
  }

  /** x on the timeline -> the nearest weight snapshot. */
  function frameAtX(clientX) {
    const steps = App.historySteps, T = App.trainer;
    if (steps.length < 2 || !T) return null;
    const r = spark.getBoundingClientRect(), f = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
    const first = T.step_ - T.lossHistory.length, target = first + f * (T.step_ - first);
    let best = 0;
    for (let i = 1; i < steps.length; i++) if (Math.abs(steps[i] - target) < Math.abs(steps[best] - target)) best = i;
    return best;
  }
  let scrubbing = false, replay = null;
  spark.addEventListener('pointerdown', (e) => {
    const i = frameAtX(e.clientX);
    if (i === null) return;
    stopReplay();
    scrubbing = true;
    App.showHistory(i);
  });
  window.addEventListener('pointermove', (e) => { if (scrubbing) { const i = frameAtX(e.clientX); if (i !== null && i !== App.viewIdx) App.showHistory(i); } });
  window.addEventListener('pointerup', () => { scrubbing = false; });
  $('liveBtn').onclick = () => { stopReplay(); App.backToLive(); };

  // replay: walk through the snapshots in ~5 s, then return to the live weights
  function stopReplay() { if (replay) { cancelAnimationFrame(replay.raf); replay = null; $('replayBtn').textContent = '⏵'; } }
  $('replayBtn').onclick = () => {
    if (replay) { stopReplay(); return; }
    const n = App.historySteps.length;
    if (n < 2) return;
    const t0 = performance.now(), dur = Math.min(8000, Math.max(3000, n * 25));
    $('replayBtn').textContent = '⏸';
    replay = {};
    (function tick(now) {
      const u = Math.min(1, (now - t0) / dur);
      App.showHistory(u * (n - 1));
      if (u < 1) replay.raf = requestAnimationFrame(tick);
      else { replay = null; $('replayBtn').textContent = '⏵'; App.backToLive(); }
    })(t0);
  };
  $('trainPlay').addEventListener('click', stopReplay, true);
  $('bendBtn').onclick = () => { stopReplay(); if (!App.playing) $('play').click(); };

  // ---- every frame: rebuild if the network changed, move the marker, light up the current slot ----------
  function frame() {
    const net = App.net;
    if (net && net.acts.join() + '|' + net.dims.join() !== sig) build();
    if (centers.length && (App.t !== lastT || sig !== lastSigDrawn)) {
      lastT = App.t; lastSigDrawn = sig;
      const t = App.t, n = centers.length - 1;
      const i = Math.min(n - 1, Math.floor(t)), u = t - i;
      const x = n === 0 ? centers[0] : centers[i] + u * (centers[Math.min(n, i + 1)] - centers[i]);
      const line = pipe.querySelector('.marker');      // the line grows from Input as the space moves through the network
      line.style.left = `${centers[0]}px`; line.style.width = `${Math.max(0, x - centers[0])}px`;
      const s = Math.round(t), near = Math.abs(t - s) < 0.02;
      pipe.querySelectorAll('.pn[data-stage]').forEach((b) => {
        const k = b.dataset.stage === 'out' ? n : +b.dataset.stage;
        // compact: a layer's button stands for both its stages (linear step and activation)
        b.classList.toggle('now', near && (b.dataset.layer ? s > 0 && Math.ceil(s / 2) === +b.dataset.layer : k === s));
      });
    }
    drawSpark();
    // after training, the big ▶ on the view: hidden while the layers play or training runs
    $('bendBtn').classList.toggle('hidden', $('timeline').classList.contains('hidden') || App.playing || App.training || !!replay);
    requestAnimationFrame(frame);
  }
  build();
  requestAnimationFrame(frame);
})();
