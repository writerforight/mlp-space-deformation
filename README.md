# Neural Space Deformation

An interactive, browser-only visualization of **what a small neural network does geometrically**:
every layer bends, stretches, folds and squashes space — and backpropagation reshapes those
deformations while it trains.

**Live demo:** https://writerforight.github.io/mlp-space-deformation/

No backend, no build step, no ML libraries. The MLP, backpropagation, SGD, Adam, PCA and every
analysis are written from first principles in plain JavaScript (`src/nn.js`, with comments).
Three.js (from a CDN) is used only to draw the 3D view.

![Forward view: a tanh network half-way through layer 2, with Jacobian ellipses](docs/screenshot-2d.png)
![Training on two moons: learned decision regions in input space, loss curve and gradient interference](docs/screenshot-train.png)
![3D view: a sphere and the unit sphere after the first tanh layer](docs/screenshot-3d.png)

## What it shows

- **Forward view.** Draw curves, circles, filled regions (2D) or spheres and curves on spheres (3D).
  Each object is sampled into 200–500 points connected in order, together with the reference grid,
  the unit circle/sphere, the basis vectors and the origin. Move the stage slider to see the objects
  after each layer — every layer is split into its **linear** step (`z = W a + b`) and its
  **activation** step (`a = σ(z)`), with a smooth morph in between.
- **Overlays.** Local Jacobian ellipses (how a tiny circle is deformed), the sign of `det J`
  (green = orientation kept, red = mirrored, i.e. a fold) and the distance each point moved.
- **Network.** 1–8 layers, hidden width 1–16 (wider layers are shown through a PCA projection),
  per-layer activation (identity, ReLU, tanh, sigmoid, GELU, sin), initialization (normal, uniform,
  Xavier, He) with scale and seed, and a **temperature** slider that randomizes the activations.
- **Training view.** Pick a task and watch backpropagation reshape space:
  - **send each class to a point** — every blue point should land on `(1, 0[, 0])`, every red point on
    `(−1, 0[, 0])` (MSE). The clearest picture of classification: each class cloud is squashed onto its
    point and the region between the classes is stretched into a cliff — a tiny version of the
    "neural collapse" seen in large classifiers;
  - **separate classes** with two output logits and cross-entropy;
  - **match a linear map** (rotation, shear, scaling) with MSE;
  - **pinned points**: drag from an input point to where its output should go.

  Datasets: two moons, circles, spirals. SGD or Adam, batch size, learning rate, steps per frame,
  redraw interval, loss curve and accuracy.
- **Continual learning.** Sequential mode trains on one target (pin, or a chunk of the dataset) at a
  time. Anti-forgetting: replay buffer, orthogonal gradient projection (OGD) or an EWC penalty.
- **Analysis.** Forgetting `F_A`, gradient interference (cosine matrix) or the neural tangent kernel,
  input sensitivity (finite differences vs. the Jacobian, Lipschitz estimates), and an invertibility
  report per layer (rank, singular values, collapse warnings).
- **You stay in control.** Nothing changes your settings or the view on its own: zoom with the wheel,
  pan by dragging. One button, **★ Recommended**, aligns everything to settings that work well for the
  current task and data (layers, width, activations, initialization, optimizer) and fits the view.
- **Share.** Export/import the full state (architecture, seed, weights, objects, pins, settings) as JSON.

## Run locally

Open `index.html` in a browser. That's it — the scripts are classic `<script>` files, so it also works
from `file://`. (Three.js is loaded from cdnjs; the 2D view works offline.)

To run the math tests (backprop vs. finite differences, Jacobians, eigen-solver, training,
anti-forgetting methods):

```bash
node test/nn.test.js
```

## Deploy on GitHub Pages

1. Push this folder to a GitHub repository.
2. **Settings → Pages → Build and deployment → Source: Deploy from a branch → Branch: `main` / `(root)`**.
3. After a minute the app is live at `https://<user>.github.io/<repository>/`.

## Project structure

```
index.html      layout, styles, controls
src/nn.js       tensor/MLP module: forward, backprop, Jacobians, losses, SGD, Adam, PCA,
                datasets, Trainer (joint / sequential, replay, OGD, EWC), analysis helpers
src/viz2d.js    Canvas renderer (pan/zoom) and small charts (loss, heatmap)
src/viz3d.js    Three.js renderer with a minimal orbit camera and ray picking
src/ui.js       state, controls, traces through the layers, interaction, training loop, panels
test/           node test of the math in nn.js
```

## Math background

### A network is a composition of maps

An MLP with layers `l = 1…L` computes

```
z_l = W_l a_{l-1} + b_l          (affine map: rotate, stretch, shear, project, translate)
a_l = σ_l(z_l)                    (activation, applied coordinate-wise)
```

so the network is `f = σ_L ∘ A_L ∘ … ∘ σ_1 ∘ A_1`. Pushing a whole curve or region through each
map one at a time shows what the network does to *space* rather than to one point. Affine maps keep
straight lines straight and parallel lines parallel; the activations are what bend space. ReLU folds
everything with a negative coordinate onto an axis (when the width equals the dimension this
destroys information), tanh and sigmoid squash space into a box, and sin wraps it around, folding it
onto itself. Classification networks are trained until the two classes become **linearly
separable** in the output space: the decision boundary there is the line `logit₁ = logit₂`.
When the task is to send each class to its own point, the network must do something even more
drastic: squash each class onto a single point (the Jacobian there tends to zero, `det J ≈ 0`) while
tearing the classes apart in between (large `‖J‖` near the decision boundary). With width 2 the whole
plane typically ends up folded onto the segment between the two target points.

### The Jacobian: the local picture

Near a point `x` the network behaves like its linear approximation
`f(x + δ) ≈ f(x) + J(x) δ` with the Jacobian `J = ∂f/∂x = D_L W_L ⋯ D_1 W_1`, where
`D_l = diag(σ_l'(z_l))`. A small circle around `x` is mapped to (approximately) an ellipse whose axes
are the singular vectors of `J` scaled by its singular values — that is what the **Jacobian ellipses**
draw. `det J > 0` keeps orientation, `det J < 0` mirrors it (the map has folded space over), and
`det J ≈ 0` collapses a dimension. The largest singular value `‖J‖₂` is the local stretch factor; the
**Lipschitz constant** is the largest stretch anywhere. The app reports a sampled lower bound
`max ‖f(x) − f(y)‖ / ‖x − y‖` and the classical upper bound `Π_l ‖W_l‖₂ · max|σ_l'|`.

### Backpropagation reshapes the deformation

Training minimizes a loss `L(θ)` (cross-entropy or mean squared error) by following its gradient.
Backpropagation computes `∂L/∂θ` with the chain rule from the output back to the input:

```
g_L = ∂L/∂a_L,   ∂L/∂z_l = g_l ⊙ σ_l'(z_l),   ∂L/∂W_l = (∂L/∂z_l) a_{l-1}ᵀ,   g_{l-1} = W_lᵀ (∂L/∂z_l)
```

Every step changes every weight a little, so the *whole* deformation of space changes at once —
that is why training on one pinned point also moves the others.

### Interference and the neural tangent kernel

A gradient step `Δθ = −η ∇_θ L_i` on point `i` changes the output at another point `j` by, to first
order, `Δf(x_j) ≈ J_j Δθ = −η J_j J_iᵀ ∂L_i/∂f`, where `J_i = ∂f(x_i)/∂θ`. The matrix
`K(x_i, x_j) = J_i J_jᵀ` is the **neural tangent kernel (NTK)**: it says how strongly learning at one
input drags the output at another. The **gradient cosine** `cos(∇L_i, ∇L_j)` is the same idea
measured on losses: positive means the two points help each other, negative means they fight
(interference).

### Forgetting and how to reduce it

Training on task B after task A can raise A's loss again. The app measures
`F_A = L_A(after training on B) − L_A(after training on A)` for every task. Three classic remedies:

- **Replay buffer** — keep a few samples of old tasks and mix them into new batches.
- **Orthogonal gradient descent (OGD)** — store `∂f(x)/∂θ` on old points and project new gradients
  onto the orthogonal complement, so (to first order) old outputs do not move. It works best with
  plain SGD and when tasks are not in direct conflict; for nearby points with contradictory targets
  the first-order guarantee breaks down.
- **Elastic weight consolidation (EWC)** — add `λ/2 Σ_i F_i (θ_i − θ*_i)²`, where `F` is the diagonal
  Fisher information of the old task: weights that mattered for it are kept close to their old values.

### Wide layers: PCA projection

Layers wider than the view dimension are drawn through their top two (or three) principal
components. The axis signs are aligned with the previous stage so the animation does not flip, and the
label shows how much of the variance the projection keeps.

## License

MIT — see [LICENSE](LICENSE).
