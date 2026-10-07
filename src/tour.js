/* An optional tour of the workspace: one element at a time, the rest of the page dimmed.

     Tour.start()          from the "?" panel, or after the guided start when its checkbox is ticked
     STEPS                 { sel, title, text, before?(), side? } — sel is the element to point at

   A step whose element is missing or hidden (e.g. the training timeline before any training) is skipped.
   The choice "show the tour after the guided start" is remembered in localStorage (nsd.tour = on | off).
*/
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const store = (k, v) => { try { localStorage.setItem(k, v); } catch (e) { /* storage may be blocked */ } };
  const load = (k) => { try { return localStorage.getItem(k); } catch (e) { return null; } };

  const STEPS = [
    { sel: '#view', side: 'center', title: 'This is space',
      text: 'The grid, the unit circle, the basis vectors and your objects live here. The network moves every one of their points; what you see is where they are at the current step.' },
    { sel: '#pipe', title: 'The network, as a path',
      text: 'Input ▸ W₁ | tanh ▸ … ▸ Output. Each layer is a linear step W·a + b and an activation. Click a step to see space there, or drag along the strip to move through the layers smoothly.' },
    { sel: '#play', title: 'Play the layers',
      text: 'Plays the whole path: first the linear map of a layer, then its activation, layer after layer (key: Space; ⏮ ⏭ step one at a time).' },
    { sel: '#basisBox', title: 'Where the basis vectors go',
      text: 'For the current layer: where o, e₁, e₂ are before it, after W·a + b and after the activation — in the layer’s real coordinates. The coloured arrows in the view end at these points.' },
    { sel: '.pipe .pi', title: 'Look inside a layer',
      text: 'ⓘ opens the layer inspector: one point followed through the layer (in → W, b → z → σ → out), the weight matrix (double-click to edit a weight), the activation and how the weights moved in training.' },
    { sel: '#objBtn', title: 'Put things into space',
      text: 'Add ready-made shapes, draw curves, circles and regions by hand, and edit each object: colour, size, position, number of points. ⓘ How it changes logs what every layer does to it.' },
    { sel: '#viewBtn', side: 'left', title: 'How to look',
      text: 'Overlays (Jacobian ellipses, det J, distance moved), the class background, and for wide layers a third direction as depth: right-drag the view to tilt it.' },
    { sel: '#drawerBtn', side: 'top', title: 'Network, goal, training',
      text: 'Model ▲ opens three columns: the network (layers, width, activation, the weight matrices as numbers), the goal (a task, or your own mix of goals) and the training settings.' },
    { sel: '#trainPlay', side: 'top', title: 'Train',
      text: 'Backpropagation reshapes the whole deformation step by step (key: T). After training, a timeline appears on the view: drag it like a video to see any earlier step.' },
    { sel: '#timeline', side: 'top', title: 'The training timeline',
      text: 'The loss of the whole run. Drag to see the network at an earlier step, ⏵ replays the training, ● live returns. Training from an old step continues from there.' },
    { sel: '#analysisBtn', side: 'bottom', title: 'Analysis and experiments',
      text: 'Analysis: loss, forgetting, gradient interference, sensitivity, invertibility. Experiments: ready-made topology tests of what a network can and cannot do.' },
    { sel: '#helpBtn', side: 'bottom', title: 'That’s it',
      text: 'The quick guide and this tour are always behind “?”. “Guided start” runs the step-by-step setup again.' },
  ];

  let i = -1, steps = [], root = null;

  const visible = (el) => !!el && el.getClientRects().length > 0 && !el.closest('.hidden');

  function build() {
    root = document.createElement('div');
    root.id = 'tour';
    root.innerHTML = `<div class="tour-spot"></div>
      <div class="tour-card" role="dialog" aria-live="polite">
        <div class="tour-count"></div><div class="tour-title"></div><div class="tour-text"></div>
        <div class="tour-btns"><button data-t="skip" class="tour-skip">Skip tour</button><span class="sp"></span>
          <button data-t="back">← Back</button><button data-t="next" class="primary">Next →</button></div>
      </div>`;
    document.body.appendChild(root);
    root.addEventListener('click', (e) => {
      const b = e.target.closest('[data-t]');
      if (!b) return;
      if (b.dataset.t === 'next') go(i + 1);
      else if (b.dataset.t === 'back') go(i - 1);
      else end();
    });
  }

  function place() {
    if (i < 0) return;
    const st = steps[i], el = document.querySelector(st.sel), spot = root.querySelector('.tour-spot'), card = root.querySelector('.tour-card');
    const r = el.getBoundingClientRect(), pad = st.side === 'center' ? -12 : 6;
    Object.assign(spot.style, { left: `${r.left - pad}px`, top: `${r.top - pad}px`, width: `${r.width + 2 * pad}px`, height: `${r.height + 2 * pad}px` });
    const W = window.innerWidth, H = window.innerHeight, cw = Math.min(340, W - 24), ch = card.offsetHeight || 170, gap = 14;
    let x, y;
    const side = st.side || (r.top > H * 0.55 ? 'top' : 'bottom');
    if (side === 'center') { x = r.left + r.width / 2 - cw / 2; y = r.top + r.height / 2 - ch / 2; }
    else if (side === 'top') { x = r.left + r.width / 2 - cw / 2; y = r.top - pad - gap - ch; }
    else if (side === 'left') { x = r.left - pad - gap - cw; y = r.top; }
    else { x = r.left + r.width / 2 - cw / 2; y = r.bottom + pad + gap; }
    x = Math.max(12, Math.min(W - cw - 12, x)); y = Math.max(12, Math.min(H - ch - 12, y));
    Object.assign(card.style, { left: `${x}px`, top: `${y}px`, width: `${cw}px` });
  }

  function go(k) {
    if (k < 0) return;
    if (k >= steps.length) { end(); return; }
    i = k;
    const st = steps[i];
    root.querySelector('.tour-count').textContent = `${i + 1} / ${steps.length}`;
    root.querySelector('.tour-title').textContent = st.title;
    root.querySelector('.tour-text').textContent = st.text;
    root.querySelector('[data-t="back"]').disabled = i === 0;
    root.querySelector('[data-t="next"]').textContent = i === steps.length - 1 ? 'Done' : 'Next →';
    root.querySelector('.tour-card').classList.remove('in'); void root.offsetWidth; root.querySelector('.tour-card').classList.add('in');
    place();
    requestAnimationFrame(place);                         // once the card has its real height
  }

  function start() {
    if (window.Shell) window.Shell.closeAll();
    if ($('helpPanel')) $('helpPanel').classList.add('hidden');
    steps = STEPS.filter((st) => visible(document.querySelector(st.sel)));
    if (!steps.length) return;
    if (!root) build();
    root.classList.remove('hidden');
    document.body.classList.add('tour-open');
    go(0);
  }

  function end() {
    i = -1;
    if (root) root.classList.add('hidden');
    document.body.classList.remove('tour-open');
    store('nsd.tourSeen', '1');
  }

  window.addEventListener('resize', () => { if (i >= 0) place(); });
  document.addEventListener('keydown', (e) => {
    if (i < 0) return;
    if (e.key === 'Escape') { end(); e.stopImmediatePropagation(); }
    else if (e.key === 'ArrowRight' || e.key === 'Enter') { go(i + 1); e.preventDefault(); e.stopImmediatePropagation(); }
    else if (e.key === 'ArrowLeft') { go(i - 1); e.preventDefault(); e.stopImmediatePropagation(); }
  }, true);

  /** The guided start asks this before it opens the workspace: show the tour afterwards? (default: until seen once) */
  const wanted = () => (load('nsd.tour') ? load('nsd.tour') === 'on' : load('nsd.tourSeen') !== '1');
  window.Tour = { start, end, wanted, setWanted: (on) => store('nsd.tour', on ? 'on' : 'off'), get active() { return i >= 0; } };
})();
