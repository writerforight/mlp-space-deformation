/*
 * viz2d.js — Canvas renderer for the 2D view.
 *
 * The renderer knows nothing about neural networks: ui.js hands it a list of "items"
 * (lines, point clouds, ellipses, arrows, an optional background image) in world coordinates,
 * and Viz2D maps them to the screen with a pan/zoom camera.
 *
 * Item kinds (all coordinates are flat arrays [x0, y0, x1, y1, …]):
 *   { kind: 'line',     pts, closed, color, width, alpha, dash, colors }   colors: per-point CSS colors (optional)
 *   { kind: 'points',   pts, color, colors, size, alpha }
 *   { kind: 'ellipses', centers, mats, r, color }    mats[i] = [a, b, c, d] for J = [[a, b], [c, d]]
 *   { kind: 'arrow',    from, to, color, width, dash }
 *   { kind: 'marker',   p, color, size, ring }
 *   { kind: 'label',    p, text, color }
 *   { kind: 'image',    bbox: [x0, y0, x1, y1], w, h, rgba }   rgba: Uint8ClampedArray (w·h·4)
 */
(function (root) {
  'use strict';

  class Viz2D {
    constructor(canvas) {
      this.canvas = canvas;
      this.ctx = canvas.getContext('2d');
      this.cx = 0; this.cy = 0;    // world point at the canvas centre
      this.scale = 90;             // pixels per world unit
      this.items = [];
      this._img = document.createElement('canvas');
      this.resize();
    }

    resize() {
      const r = this.canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
      // a hidden canvas (3D mode) measures 0×0: keep the last real size instead of shrinking to nothing
      if (r.width < 1 || r.height < 1) return;
      this.w = Math.max(10, r.width); this.h = Math.max(10, r.height); this.dpr = dpr;
      this.canvas.width = Math.round(this.w * dpr);
      this.canvas.height = Math.round(this.h * dpr);
    }

    toScreen(x, y) { return [this.w / 2 + (x - this.cx) * this.scale, this.h / 2 - (y - this.cy) * this.scale]; }
    toWorld(px, py) { return [this.cx + (px - this.w / 2) / this.scale, this.cy - (py - this.h / 2) / this.scale]; }

    /** Visible world rectangle [x0, y0, x1, y1]. */
    viewBox() {
      const [x0, y1] = this.toWorld(0, 0), [x1, y0] = this.toWorld(this.w, this.h);
      return [x0, y0, x1, y1];
    }

    /** Zoom so that the world box [x0, y0, x1, y1] fills the view with a margin. */
    fit(box, margin = 0.12) {
      const [x0, y0, x1, y1] = box;
      const bw = Math.max(1e-3, x1 - x0), bh = Math.max(1e-3, y1 - y0);
      this.cx = (x0 + x1) / 2; this.cy = (y0 + y1) / 2;
      this.scale = Math.min(this.w / (bw * (1 + 2 * margin)), this.h / (bh * (1 + 2 * margin)));
    }

    zoomAt(px, py, factor) {
      const [wx, wy] = this.toWorld(px, py);
      this.scale = Math.min(1e5, Math.max(2, this.scale * factor));
      const [nx, ny] = this.toWorld(px, py);
      this.cx += wx - nx; this.cy += wy - ny;
    }

    panBy(dpx, dpy) { this.cx -= dpx / this.scale; this.cy += dpy / this.scale; }

    render(items) {
      if (items) this.items = items;
      const ctx = this.ctx;
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      ctx.clearRect(0, 0, this.w, this.h);
      this.drawFixedAxes();
      for (const it of this.items) {
        switch (it.kind) {
          case 'image': this.drawImage(it); break;
          case 'line': this.drawLine(it); break;
          case 'points': this.drawPoints(it); break;
          case 'ellipses': this.drawEllipses(it); break;
          case 'arrow': this.drawArrow(it); break;
          case 'marker': this.drawMarker(it); break;
          case 'label': this.drawLabel(it); break;
        }
      }
    }

    /** Faint, fixed coordinate axes with tick labels — the "ruler" the moving grid is compared to. */
    drawFixedAxes() {
      const ctx = this.ctx, [x0, y0, x1, y1] = this.viewBox();
      const step = niceStep((x1 - x0) / 8);
      ctx.save();
      ctx.font = '10px ui-monospace, monospace';
      ctx.fillStyle = 'rgba(255,255,255,0.28)';
      ctx.strokeStyle = 'rgba(255,255,255,0.05)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let x = Math.ceil(x0 / step) * step; x <= x1; x += step) {
        const [px] = this.toScreen(x, 0); ctx.moveTo(px, 0); ctx.lineTo(px, this.h);
        ctx.fillText(fmtTick(x), px + 2, this.h - 4);
      }
      for (let y = Math.ceil(y0 / step) * step; y <= y1; y += step) {
        const [, py] = this.toScreen(0, y); ctx.moveTo(0, py); ctx.lineTo(this.w, py);
        ctx.fillText(fmtTick(y), 3, py - 2);
      }
      ctx.stroke();
      ctx.restore();
    }

    drawImage(it) {
      const c = this._img;
      if (c.width !== it.w || c.height !== it.h) { c.width = it.w; c.height = it.h; }
      c.getContext('2d').putImageData(new ImageData(it.rgba, it.w, it.h), 0, 0);
      const [px0, py0] = this.toScreen(it.bbox[0], it.bbox[3]), [px1, py1] = this.toScreen(it.bbox[2], it.bbox[1]);
      this.ctx.save();
      this.ctx.imageSmoothingEnabled = true;
      this.ctx.globalAlpha = it.alpha ?? 1;
      this.ctx.drawImage(c, px0, py0, px1 - px0, py1 - py0);
      this.ctx.restore();
    }

    drawLine(it) {
      const ctx = this.ctx, p = it.pts, n = p.length / 2;
      if (n < 2) return;
      ctx.save();
      ctx.globalAlpha = it.alpha ?? 1;
      ctx.lineWidth = it.width ?? 1.5;
      ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      if (it.dash) ctx.setLineDash(it.dash);
      if (it.colors) {                    // per-point colours: draw segment by segment
        const m = it.closed ? n : n - 1;
        for (let i = 0; i < m; i++) {
          const j = (i + 1) % n;
          const [ax, ay] = this.toScreen(p[2 * i], p[2 * i + 1]), [bx, by] = this.toScreen(p[2 * j], p[2 * j + 1]);
          ctx.strokeStyle = it.colors[i];
          ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
        }
      } else {
        ctx.strokeStyle = it.color;
        ctx.beginPath();
        for (let i = 0; i < n; i++) {
          const [x, y] = this.toScreen(p[2 * i], p[2 * i + 1]);
          i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        if (it.closed) ctx.closePath();
        ctx.stroke();
      }
      ctx.restore();
    }

    drawPoints(it) {
      const ctx = this.ctx, p = it.pts, n = p.length / 2, r = it.size ?? 2;
      ctx.save();
      ctx.globalAlpha = it.alpha ?? 1;
      ctx.fillStyle = it.color;
      for (let i = 0; i < n; i++) {
        const [x, y] = this.toScreen(p[2 * i], p[2 * i + 1]);
        if (it.colors) ctx.fillStyle = it.colors[i];
        ctx.beginPath(); ctx.arc(x, y, r, 0, 2 * Math.PI); ctx.fill();
      }
      ctx.restore();
    }

    /** Image of a small input circle of radius r under the local Jacobian: c + r·J·(cos t, sin t). */
    drawEllipses(it) {
      const ctx = this.ctx, r = it.r ?? 0.1;
      ctx.save();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = it.color ?? 'rgba(255,255,255,0.7)';
      for (let i = 0; i < it.mats.length; i++) {
        const [a, b, c, d] = it.mats[i], cx = it.centers[2 * i], cy = it.centers[2 * i + 1];
        if (it.colors) ctx.strokeStyle = it.colors[i];
        ctx.beginPath();
        for (let k = 0; k <= 24; k++) {
          const t = (2 * Math.PI * k) / 24, u = r * Math.cos(t), v = r * Math.sin(t);
          const [x, y] = this.toScreen(cx + a * u + b * v, cy + c * u + d * v);
          k ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
        }
        ctx.stroke();
      }
      ctx.restore();
    }

    drawArrow(it) {
      const ctx = this.ctx;
      const [ax, ay] = this.toScreen(it.from[0], it.from[1]), [bx, by] = this.toScreen(it.to[0], it.to[1]);
      const len = Math.hypot(bx - ax, by - ay);
      ctx.save();
      ctx.strokeStyle = ctx.fillStyle = it.color;
      ctx.lineWidth = it.width ?? 1.5;
      if (it.dash) ctx.setLineDash(it.dash);
      ctx.beginPath(); ctx.moveTo(ax, ay); ctx.lineTo(bx, by); ctx.stroke();
      if (len > 6) {
        ctx.setLineDash([]);
        const ux = (bx - ax) / len, uy = (by - ay) / len, h = Math.min(10, len / 2);
        ctx.beginPath();
        ctx.moveTo(bx, by);
        ctx.lineTo(bx - h * ux + 0.5 * h * uy, by - h * uy - 0.5 * h * ux);
        ctx.lineTo(bx - h * ux - 0.5 * h * uy, by - h * uy + 0.5 * h * ux);
        ctx.closePath(); ctx.fill();
      }
      ctx.restore();
    }

    drawMarker(it) {
      const ctx = this.ctx, [x, y] = this.toScreen(it.p[0], it.p[1]), s = it.size ?? 4;
      ctx.save();
      ctx.fillStyle = ctx.strokeStyle = it.color;
      ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.arc(x, y, s, 0, 2 * Math.PI);
      it.ring ? ctx.stroke() : ctx.fill();
      ctx.restore();
    }

    drawLabel(it) {
      const ctx = this.ctx, [x, y] = this.toScreen(it.p[0], it.p[1]);
      ctx.save();
      ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
      ctx.fillStyle = it.color ?? '#ddd';
      ctx.fillText(it.text, x + 6, y - 6);
      ctx.restore();
    }
  }

  function niceStep(raw) {
    const p = Math.pow(10, Math.floor(Math.log10(raw))), m = raw / p;
    return (m < 1.5 ? 1 : m < 3.5 ? 2 : m < 7.5 ? 5 : 10) * p;
  }
  function fmtTick(v) {
    if (Math.abs(v) < 1e-9) return '0';
    return Math.abs(v) >= 1000 || Math.abs(v) < 0.01 ? v.toExponential(0) : +v.toFixed(2) + '';
  }

  /**
   * Small line chart / heatmap helpers for the side panels (loss curve, sensitivity, matrices).
   */
  const Charts = {
    /** Line chart. series: [{ ys, color, dash }], options: { logY, xs, yLabel, xLabel }. */
    line(canvas, series, opt = {}) {
      const { ctx, w, h } = prep(canvas);
      const pad = { l: 44, r: 8, t: 8, b: 20 };
      const tf = (v) => (opt.logY ? Math.log10(Math.max(v, 1e-12)) : v);
      let ymin = Infinity, ymax = -Infinity, n = 0;
      for (const s of series) for (const v of s.ys) if (Number.isFinite(v)) { const t = tf(v); ymin = Math.min(ymin, t); ymax = Math.max(ymax, t); }
      for (const s of series) n = Math.max(n, s.ys.length);
      ctx.fillStyle = '#7d8590'; ctx.font = '10px ui-monospace, monospace';
      if (!Number.isFinite(ymin) || n < 2) { ctx.fillText(opt.empty ?? 'no data yet', pad.l, h / 2); return; }
      if (ymax - ymin < 1e-9) { ymax += 0.5; ymin -= 0.5; }
      const X = (i, m) => pad.l + ((w - pad.l - pad.r) * i) / Math.max(1, m - 1);
      const Y = (v) => h - pad.b - ((h - pad.t - pad.b) * (tf(v) - ymin)) / (ymax - ymin);
      ctx.strokeStyle = '#30363d'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(pad.l, pad.t); ctx.lineTo(pad.l, h - pad.b); ctx.lineTo(w - pad.r, h - pad.b); ctx.stroke();
      const lab = (t) => (opt.logY ? (Math.pow(10, t)).toExponential(0) : (+t.toPrecision(3)).toString());
      ctx.fillText(lab(ymax), 2, pad.t + 8); ctx.fillText(lab(ymin), 2, h - pad.b);
      if (opt.xLabel) ctx.fillText(opt.xLabel, w - pad.r - ctx.measureText(opt.xLabel).width, h - 5);
      for (const s of series) {
        ctx.strokeStyle = s.color; ctx.lineWidth = 1.5; ctx.setLineDash(s.dash ?? []);
        ctx.beginPath();
        // draw at most ~w points (min-max decimation would be nicer; striding is enough here)
        const m = s.ys.length, stride = Math.max(1, Math.floor(m / (w - pad.l)));
        let first = true;
        for (let i = 0; i < m; i += stride) {
          const v = s.ys[i]; if (!Number.isFinite(v)) continue;
          const x = X(i, m), y = Y(v);
          first ? ctx.moveTo(x, y) : ctx.lineTo(x, y); first = false;
        }
        ctx.stroke();
      }
      ctx.setLineDash([]);
    },

    /** Matrix heatmap. mode 'diverging' maps [-1, 1] violet→black→gold; 'sequential' maps [0, max]. */
    heatmap(canvas, M, opt = {}) {
      const { ctx, w, h } = prep(canvas);
      const n = M.length;
      ctx.fillStyle = '#7d8590'; ctx.font = '10px ui-monospace, monospace';
      if (!n) { ctx.fillText(opt.empty ?? 'nothing to show', 8, h / 2); return; }
      const size = Math.min(w - 24, h - 24), cell = size / n, x0 = 20, y0 = 4;
      let mx = 0;
      for (const r of M) for (const v of r) mx = Math.max(mx, Math.abs(v));
      for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
          const v = M[i][j];
          ctx.fillStyle = opt.mode === 'sequential' ? seqColor(mx ? v / mx : 0) : divColor(v);
          ctx.fillRect(x0 + j * cell, y0 + i * cell, Math.ceil(cell), Math.ceil(cell));
        }
      }
      if (n <= 16) {
        ctx.fillStyle = '#7d8590';
        for (let i = 0; i < n; i++) {
          ctx.fillText(String(i + 1), 4, y0 + (i + 0.6) * cell + 3);
          ctx.fillText(String(i + 1), x0 + (i + 0.35) * cell, y0 + size + 12);
        }
      }
    },
  };

  function prep(canvas) {
    const r = canvas.getBoundingClientRect(), dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(r.width * dpr); canvas.height = Math.round(r.height * dpr);
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, r.width, r.height);
    return { ctx, w: r.width, h: r.height };
  }

  /** Diverging colour for v ∈ [-1, 1]: blue (−) → dark (0) → red (+). */
  function divColor(v) {
    const t = Math.max(-1, Math.min(1, v));
    const a = Math.abs(t);
    // gold for +, violet for − (blue / red are reserved for the two classes)
    return t >= 0 ? `rgb(${Math.round(30 + 197 * a)},${Math.round(30 + 149 * a)},${Math.round(36 + 29 * a)})`
      : `rgb(${Math.round(30 + 158 * a)},${Math.round(30 + 110 * a)},${Math.round(36 + 219 * a)})`;
  }
  /** Sequential colour for t ∈ [0, 1] (dark → teal → yellow, a viridis-like ramp). */
  function seqColor(t) {
    t = Math.max(0, Math.min(1, t));
    const stops = [[68, 1, 84], [59, 82, 139], [33, 145, 140], [94, 201, 98], [253, 231, 37]];
    const x = t * (stops.length - 1), i = Math.min(stops.length - 2, Math.floor(x)), f = x - i;
    const c = stops[i].map((v, k) => Math.round(v + f * (stops[i + 1][k] - v)));
    return `rgb(${c[0]},${c[1]},${c[2]})`;
  }

  /**
   * Wheel distance in pixels.  Firefox reports wheel steps in lines (deltaMode 1, ≈3 per notch) or
   * pages (deltaMode 2); treating those as pixels would make zooming almost invisible.
   */
  function wheelPixels(e) {
    return e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? 400 : 1);
  }

  /**
   * Two-finger touch gestures on an element: pinch to zoom, move both fingers to pan.  Register it before the
   * element's other pointer handlers: while two fingers are (or were, until all lift) down, they see nothing.
   *   onGesture(factor, cx, cy, dx, dy): zoom by factor about (cx, cy) (element pixels), then pan by (dx, dy)
   *   onStart(): the second finger came down — cancel whatever the first one started
   */
  function touchGestures(el, onGesture, onStart) {
    const pts = new Map();
    let prev = null, active = false;
    const geo = () => {
      const [a, b] = [...pts.values()], r = el.getBoundingClientRect();
      return { d: Math.hypot(a.x - b.x, a.y - b.y) || 1, cx: (a.x + b.x) / 2 - r.left, cy: (a.y + b.y) / 2 - r.top };
    };
    el.addEventListener('pointerdown', (e) => {
      if (e.pointerType !== 'touch') return;
      // the first finger of a new touch: forget everything (a lifted finger whose pointerup went astray
      // must never leave the view stuck in "two fingers")
      if (e.isPrimary) { pts.clear(); active = false; }
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (pts.size === 2) { active = true; prev = geo(); onStart(); }
      if (active) e.stopImmediatePropagation();
    });
    el.addEventListener('pointermove', (e) => {
      if (!pts.has(e.pointerId)) return;
      pts.set(e.pointerId, { x: e.clientX, y: e.clientY });
      if (!active) return;
      e.stopImmediatePropagation();
      if (pts.size < 2) return;
      const g = geo();
      onGesture(g.d / prev.d, g.cx, g.cy, g.cx - prev.cx, g.cy - prev.cy);
      prev = g;
    });
    const end = (e) => {
      if (!pts.delete(e.pointerId)) return;
      if (active) e.stopImmediatePropagation();
      if (pts.size === 0) active = false;
      else if (pts.size === 1) prev = null;
    };
    el.addEventListener('pointerup', end);
    el.addEventListener('pointercancel', end);
    // a finger lifted outside the element: its pointerup lands elsewhere
    const away = (e) => { if (e.target !== el && pts.delete(e.pointerId) && pts.size === 0) active = false; };
    window.addEventListener('pointerup', away, true);
    window.addEventListener('pointercancel', away, true);
  }

  root.Viz2D = Viz2D;
  root.touchGestures = touchGestures;
  root.wheelPixels = wheelPixels;
  root.Charts = Charts;
  root.Colors = { divColor, seqColor };
})(window);
