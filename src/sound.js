/* Sound — "deep space": everything is synthesised here with Web Audio, no sound files.

     drone     a low, warm pad while the opening scene turns; open(x) opens its filter as space untwists
     layer(k)  a soft bell for layer k (one step of a rising line)
     resolve() the classes have landed: one bell per class, together a chord
     click()   a button          train()  training starts          done()  training reached its goal

   Browsers only allow sound after the user has touched the page, so nothing plays before that.  The
   on / off choice is remembered (nsd.sound); sound is on unless the user turned it off.
*/
(function () {
  'use strict';
  const KEY = 'nsd.sound';
  let on = true;
  try { on = localStorage.getItem(KEY) !== '0'; } catch (e) { /* storage may be unavailable */ }
  let ctx = null, master = null, verb = null, drone = null;
  const listeners = [];

  const hz = (midi) => 440 * Math.pow(2, (midi - 69) / 12);
  const LINE = [57, 60, 64, 67, 69, 72, 74, 76];            // the rising line of the layers
  const CHORD = [69, 73, 76];                                 // A major: one note per class

  /** The audio graph, made on the first use after a user gesture. */
  function audio() {
    if (ctx) { if (ctx.state === 'suspended') ctx.resume(); return ctx; }
    const AC = window.AudioContext || window.webkitAudioContext;
    if (!AC) return null;
    ctx = new AC();
    master = ctx.createGain(); master.gain.value = 0.7;
    const comp = ctx.createDynamicsCompressor(); comp.threshold.value = -18; comp.ratio.value = 3;
    master.connect(comp).connect(ctx.destination);
    verb = ctx.createConvolver();                             // a soft reverb from decaying noise
    const len = ctx.sampleRate * 3.2, ir = ctx.createBuffer(2, len, ctx.sampleRate);
    for (let c = 0; c < 2; c++) { const d = ir.getChannelData(c); for (let i = 0; i < len; i++) d[i] = (Math.random() * 2 - 1) * Math.pow(1 - i / len, 2.6); }
    verb.buffer = ir;
    const vg = ctx.createGain(); vg.gain.value = 0.55; verb.connect(vg).connect(master);
    return ctx;
  }

  /** One voice: oscillator (with a touch of FM for bells) → low-pass → envelope → dry + reverb. */
  function voice({ f, type = 'sine', t = 0, a = 0.01, d = 0.4, s = 0, hold = 0, r = 0.3, g = 0.2, cut = 8000, q = 0.5, wet = 0.4, detune = 0, fm = 0, fmRatio = 2 }) {
    const c = audio();
    if (!c) return null;
    const t0 = c.currentTime + t, end = t0 + a + d + hold;
    const o = c.createOscillator(); o.type = type; o.frequency.value = f; o.detune.value = detune;
    if (fm) {
      const m = c.createOscillator(), mg = c.createGain(); m.frequency.value = f * fmRatio;
      mg.gain.setValueAtTime(f * fm, t0); mg.gain.exponentialRampToValueAtTime(Math.max(1e-3, f * fm * 0.05), end);
      m.connect(mg).connect(o.frequency); m.start(t0); m.stop(end + r + 0.1);
    }
    const fl = c.createBiquadFilter(); fl.type = 'lowpass'; fl.frequency.value = cut; fl.Q.value = q;
    const e = c.createGain();
    e.gain.setValueAtTime(0, t0); e.gain.linearRampToValueAtTime(g, t0 + a);
    e.gain.setTargetAtTime(g * s, t0 + a, d / 3);
    e.gain.setTargetAtTime(0, end, r / 4);
    o.connect(fl).connect(e);
    e.connect(master); const w = c.createGain(); w.gain.value = wet; e.connect(w).connect(verb);
    o.start(t0); o.stop(end + r + 0.3);
    return { o, fl, e };
  }
  const bell = (m, t = 0, g = 0.07, d = 1.2) => voice({ f: hz(m), t, a: 0.005, d, r: d * 1.3, g, fm: 1.2, fmRatio: 3.5, wet: 0.7 });

  /** The drone: detuned saws through a slowly opening low-pass, and a whisper of filtered noise. */
  function droneStart() {
    if (!on || drone || !audio()) return;
    const c = ctx, t0 = c.currentTime;
    const out = c.createGain(); out.gain.setValueAtTime(0, t0); out.gain.linearRampToValueAtTime(1, t0 + 2.5);
    const fl = c.createBiquadFilter(); fl.type = 'lowpass'; fl.frequency.value = 140; fl.Q.value = 1.2;
    fl.connect(out); out.connect(master); const w = c.createGain(); w.gain.value = 0.6; out.connect(w).connect(verb);
    const oscs = [];
    [33, 40, 45].forEach((m, i) => [-8, 8].forEach((dt) => {
      const o = c.createOscillator(), g = c.createGain(); o.type = 'sawtooth'; o.frequency.value = hz(m); o.detune.value = dt + i;
      g.gain.value = 0.04; o.connect(g).connect(fl); o.start(); oscs.push(o);
    }));
    const n = c.createBufferSource(), b = c.createBuffer(1, c.sampleRate * 2, c.sampleRate), x = b.getChannelData(0);
    for (let i = 0; i < x.length; i++) x[i] = Math.random() * 2 - 1;
    n.buffer = b; n.loop = true;
    const bp = c.createBiquadFilter(); bp.type = 'bandpass'; bp.Q.value = 2; bp.frequency.value = 200;
    const ng = c.createGain(); ng.gain.value = 0; n.connect(bp).connect(ng).connect(out); n.start(); oscs.push(n);
    drone = { out, fl, bp, ng, oscs };
  }
  /** x in 0…1: how far space has untwisted — the filter opens and the noise rises with it. */
  function droneOpen(x) {
    if (!drone) return;
    const t = ctx.currentTime, k = Math.max(0, Math.min(1, x));
    drone.fl.frequency.setTargetAtTime(140 * Math.pow(10, k), t, 0.15);       // 140 Hz … 1.4 kHz
    drone.bp.frequency.setTargetAtTime(200 * Math.pow(12, k), t, 0.15);
    drone.ng.gain.setTargetAtTime(0.03 * Math.sin(Math.PI * k), t, 0.2);
  }
  function droneStop(fade = 1.5) {
    if (!drone) return;
    const d = drone, t = ctx.currentTime;
    drone = null;
    d.out.gain.cancelScheduledValues(t); d.out.gain.setValueAtTime(d.out.gain.value, t); d.out.gain.linearRampToValueAtTime(0, t + fade);
    d.oscs.forEach((o) => o.stop(t + fade + 0.1));
  }

  const Sound = {
    get on() { return on; },
    set on(v) {
      on = !!v;
      try { localStorage.setItem(KEY, on ? '1' : '0'); } catch (e) { /* storage may be unavailable */ }
      if (!on) droneStop(0.4);
      listeners.forEach((f) => f(on));
    },
    onChange(f) { listeners.push(f); },
    /** Call from a user gesture: browsers start audio only then. */
    unlock() { if (on) audio(); },
    get ready() { return !!ctx && ctx.state === 'running'; },
    droneStart, droneOpen, droneStop,
    layer(k, g = 0.07) { if (on && audio()) bell(LINE[Math.min(LINE.length - 1, Math.max(0, k))], 0, g); },
    resolve(g = 0.09) {
      if (!on || !audio()) return;
      CHORD.forEach((m, i) => bell(m, i * 0.18, g, 3));
      voice({ f: hz(45), t: 0, a: 0.4, d: 3, r: 3, g: 2 * g, wet: 0.5 });
    },
    click() { if (on && audio()) voice({ f: 1200, a: 0.001, d: 0.05, r: 0.05, g: 0.06, cut: 3000, wet: 0.15 }); },
    train() {
      if (!on || !audio()) return;
      [45, 52].forEach((m, i) => [-8, 8].forEach((dt) => {
        const v = voice({ f: hz(m), type: 'sawtooth', a: 0.3, d: 0.1, s: 1, hold: 1.6, r: 1.6, g: 0.03, cut: 200, q: 1.2, wet: 0.6, detune: dt + i });
        if (v) v.fl.frequency.exponentialRampToValueAtTime(1600, ctx.currentTime + 1.8);
      }));
    },
    done() { if (on && audio()) [69, 73, 76, 81].forEach((m, k) => bell(m, k * 0.09, 0.06, 1.8)); },
  };
  window.Sound = Sound;
})();
