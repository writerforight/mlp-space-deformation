/*
 * viz3d.js — Three.js renderer for the 3D view.
 *
 * Same idea as viz2d.js: ui.js passes "items" in world coordinates (flat [x, y, z, …] arrays) and
 * Viz3D keeps one Three.js object per item id, updating vertex positions in place every frame.
 * A small hand-written orbit camera (drag = rotate, wheel = zoom, right-drag = pan) avoids extra
 * dependencies; `ray(px, py)` lets ui.js pick points on a sphere or a plane.
 *
 * Item kinds:
 *   { id, kind: 'line',     pts, closed, color, colors, alpha }   colors: per-point [r, g, b] in 0..1
 *   { id, kind: 'points',   pts, color, colors, size }
 *   { id, kind: 'segments', pts, color, alpha }                   pairs of points
 *   { id, kind: 'ellipses', centers, mats, r, color }             mats[i]: 3×3 row-major (9 numbers)
 */
(function (root) {
  'use strict';

  class Viz3D {
    constructor(container) {
      this.container = container;
      this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true });
      this.renderer.setPixelRatio(window.devicePixelRatio || 1);
      this.renderer.setClearColor(0x0d1117, 1);
      container.appendChild(this.renderer.domElement);
      this.scene = new THREE.Scene();
      this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
      this.target = new THREE.Vector3(0, 0, 0);
      this.orbit = { theta: 0.8, phi: 1.1, radius: 7 };
      this.objects = new Map();
      this.allowRotate = () => true;
      this.addFixedAxes();
      this.bindControls();
      this.resize();
    }

    /** Faint fixed axes (the static reference the deforming grid is compared to). */
    addFixedAxes() {
      const g = new THREE.BufferGeometry(), L = 3;
      g.setAttribute('position', new THREE.Float32BufferAttribute([-L, 0, 0, L, 0, 0, 0, -L, 0, 0, L, 0, 0, 0, -L, 0, 0, L], 3));
      const m = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.08 });
      this.scene.add(new THREE.LineSegments(g, m));
    }

    resize() {
      const r = this.container.getBoundingClientRect();
      this.w = Math.max(10, r.width); this.h = Math.max(10, r.height);
      this.renderer.setSize(this.w, this.h, false);
      this.renderer.domElement.style.width = '100%';
      this.renderer.domElement.style.height = '100%';
      this.camera.aspect = this.w / this.h;
      this.camera.updateProjectionMatrix();
    }

    updateCamera() {
      const { theta, phi, radius } = this.orbit;
      this.camera.position.set(
        this.target.x + radius * Math.sin(phi) * Math.cos(theta),
        this.target.y + radius * Math.cos(phi),
        this.target.z + radius * Math.sin(phi) * Math.sin(theta));
      this.camera.lookAt(this.target);
    }

    bindControls() {
      const el = this.renderer.domElement;
      let drag = null;
      el.addEventListener('contextmenu', (e) => e.preventDefault());
      el.addEventListener('pointerdown', (e) => {
        if (!this.allowRotate(e)) return;
        drag = { x: e.clientX, y: e.clientY, pan: e.button === 2 || e.shiftKey };
        el.setPointerCapture(e.pointerId);
      });
      el.addEventListener('pointermove', (e) => {
        if (!drag) return;
        const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
        drag.x = e.clientX; drag.y = e.clientY;
        if (drag.pan) {
          const s = this.orbit.radius / this.h;
          const right = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 0);
          const up = new THREE.Vector3().setFromMatrixColumn(this.camera.matrix, 1);
          this.target.addScaledVector(right, -dx * s).addScaledVector(up, dy * s);
        } else {
          this.orbit.theta += dx * 0.008;
          this.orbit.phi = Math.min(Math.PI - 0.05, Math.max(0.05, this.orbit.phi - dy * 0.008));
        }
      });
      const end = () => { drag = null; };
      el.addEventListener('pointerup', end);
      el.addEventListener('pointercancel', end);
      el.addEventListener('wheel', (e) => {
        e.preventDefault();
        this.orbit.radius = Math.min(200, Math.max(0.5, this.orbit.radius * Math.exp(e.deltaY * 0.001)));
      }, { passive: false });
    }

    /** Fit the camera distance to a bounding radius. */
    fitRadius(r) { this.orbit.radius = Math.max(1.5, r * 2.5); this.target.set(0, 0, 0); }

    /** World-space ray through a canvas pixel (client coordinates relative to the canvas). */
    ray(px, py) {
      const ndc = new THREE.Vector3((px / this.w) * 2 - 1, -(py / this.h) * 2 + 1, 0.5);
      ndc.unproject(this.camera);
      const origin = this.camera.position.clone();
      return { origin, dir: ndc.sub(origin).normalize() };
    }

    /** Intersection of a pixel ray with a sphere (nearest hit) or null. */
    pickSphere(px, py, center, radius) {
      const { origin, dir } = this.ray(px, py);
      const oc = origin.clone().sub(new THREE.Vector3(...center));
      const b = oc.dot(dir), c = oc.dot(oc) - radius * radius, disc = b * b - c;
      if (disc < 0) return null;
      const t = -b - Math.sqrt(disc);
      if (t < 0) return null;
      return origin.addScaledVector(dir, t).toArray();
    }

    /** Intersection of a pixel ray with the plane z = 0 (the "floor" pins are placed on). */
    pickPlaneZ0(px, py) {
      const { origin, dir } = this.ray(px, py);
      if (Math.abs(dir.z) < 1e-9) return null;
      const t = -origin.z / dir.z;
      return t > 0 ? origin.addScaledVector(dir, t).toArray() : null;
    }

    /** Replace the drawn items; objects are created once per id and updated in place afterwards. */
    setItems(items) {
      const seen = new Set();
      for (const it of items) {
        seen.add(it.id);
        const verts = it.kind === 'ellipses' ? ellipseVerts(it) : it.pts;
        const nVerts = verts.length / 3;
        let rec = this.objects.get(it.id);
        if (!rec || rec.kind !== it.kind || rec.n !== nVerts || rec.closed !== !!it.closed || rec.hasColors !== !!it.colors) {
          if (rec) { this.scene.remove(rec.obj); rec.obj.geometry.dispose(); rec.obj.material.dispose(); }
          rec = this.makeObject(it, nVerts);
          this.objects.set(it.id, rec);
          this.scene.add(rec.obj);
        }
        const pos = rec.obj.geometry.attributes.position;
        for (let i = 0; i < verts.length; i++) pos.array[i] = Number.isFinite(verts[i]) ? verts[i] : 0;
        pos.needsUpdate = true;
        if (it.colors) {
          const col = rec.obj.geometry.attributes.color;
          const cols = it.kind === 'ellipses' ? repeatColors(it.colors, ELLIPSE_VERTS) : it.colors;
          for (let i = 0; i < nVerts; i++) { const c = cols[i] || [1, 1, 1]; col.array[3 * i] = c[0]; col.array[3 * i + 1] = c[1]; col.array[3 * i + 2] = c[2]; }
          col.needsUpdate = true;
        } else if (it.color) {
          rec.obj.material.color.set(it.color);
        }
        rec.obj.material.opacity = it.alpha ?? 1;
        rec.obj.geometry.computeBoundingSphere();
      }
      for (const [id, rec] of this.objects) {
        if (!seen.has(id)) { this.scene.remove(rec.obj); rec.obj.geometry.dispose(); rec.obj.material.dispose(); this.objects.delete(id); }
      }
    }

    makeObject(it, n) {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
      if (it.colors) g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
      const common = { transparent: true, opacity: it.alpha ?? 1, vertexColors: !!it.colors };
      let obj;
      if (it.kind === 'points') {
        obj = new THREE.Points(g, new THREE.PointsMaterial({ ...common, size: it.size ?? 0.05, sizeAttenuation: true, color: it.colors ? 0xffffff : it.color }));
      } else if (it.kind === 'segments' || it.kind === 'ellipses') {
        obj = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ ...common, color: it.colors ? 0xffffff : it.color }));
      } else {
        const M = it.closed ? THREE.LineLoop : THREE.Line;
        obj = new M(g, new THREE.LineBasicMaterial({ ...common, color: it.colors ? 0xffffff : it.color }));
      }
      obj.frustumCulled = false;
      return { obj, kind: it.kind, n, closed: !!it.closed, hasColors: !!it.colors };
    }

    render() {
      this.updateCamera();
      this.renderer.render(this.scene, this.camera);
    }
  }

  /**
   * Jacobian "ellipsoids": the images of three small orthogonal circles (in the xy, yz and zx planes)
   * around each centre under its 3×3 Jacobian — drawn as line segments (24 segments per circle).
   */
  const RING_SEGMENTS = 24, ELLIPSE_VERTS = 3 * RING_SEGMENTS * 2; // 3 rings × 24 segments × 2 vertices
  function ellipseVerts(it) {
    const out = [], r = it.r ?? 0.1, K = RING_SEGMENTS;
    for (let e = 0; e < it.mats.length; e++) {
      const M = it.mats[e], cx = it.centers[3 * e], cy = it.centers[3 * e + 1], cz = it.centers[3 * e + 2];
      for (const [a, b] of [[0, 1], [1, 2], [2, 0]]) {
        let prev = null;
        for (let k = 0; k <= K; k++) {
          const t = (2 * Math.PI * k) / K, u = [0, 0, 0];
          u[a] = r * Math.cos(t); u[b] = r * Math.sin(t);
          const p = [0, 1, 2].map((i) => [cx, cy, cz][i] + M[3 * i] * u[0] + M[3 * i + 1] * u[1] + M[3 * i + 2] * u[2]);
          if (prev) out.push(...prev, ...p);
          prev = p;
        }
      }
    }
    return out;
  }
  /** One colour per ellipsoid → one colour per vertex. */
  function repeatColors(colors, per) {
    const out = [];
    for (const c of colors) for (let i = 0; i < per; i++) out.push(c);
    return out;
  }

  root.Viz3D = Viz3D;
})(window);
