/* Workspace shell: what opens where.

     view (always)          the canvas; nothing is docked beside it
     [data-pop] buttons     small popovers on the view (View ▾ top right, Objects bottom left)
     [data-sheet] buttons   one sheet on the right: layer inspector | analysis | experiments
     #drawerBtn             the model drawer above the strip: network | goal | training

   One rule: at most one popover or sheet is open at a time (the drawer is part of the strip and may stay
   open).  ✕, Esc, or a click outside a popover closes it.  ui.js keeps all the logic; it only renders
   panels that are visible, so after opening one we send a resize to make it draw at its real size.
*/
(function () {
  'use strict';
  const $ = (id) => document.getElementById(id);
  let open = null;              // { kind: 'pop' | 'sheet', el, pane?, trigger }

  const triggers = (sel) => [...document.querySelectorAll(sel)];
  const redraw = () => window.dispatchEvent(new Event('resize'));

  function markTriggers() {
    triggers('[data-pop]').forEach((b) => b.classList.toggle('on', !!open && open.kind === 'pop' && open.el.id === b.dataset.pop));
    triggers('[data-sheet]').forEach((b) => b.classList.toggle('on', !!open && open.kind === 'sheet' && open.pane === b.dataset.sheet));
  }

  function closeAll() {
    if (!open) return;
    open.el.classList.add('hidden');
    open = null;
    markTriggers();
  }

  function openPop(id) {
    if (open && open.kind === 'pop' && open.el.id === id) { closeAll(); return; }
    closeAll();
    const el = $(id);
    el.classList.remove('hidden');
    open = { kind: 'pop', el };
    markTriggers();
    redraw();
  }

  function openSheet(pane, toggle = true) {
    if (toggle && open && open.kind === 'sheet' && open.pane === pane) { closeAll(); return; }
    if (!(open && open.kind === 'sheet')) closeAll();
    const el = $('sheet');
    el.querySelectorAll('.pane').forEach((p) => p.classList.toggle('on', p.dataset.pane === pane));
    $('sheetTitle').textContent = el.querySelector(`.pane[data-pane="${pane}"]`).dataset.title;
    $('sumInspector').classList.toggle('hidden', pane !== 'inspector');
    el.classList.remove('hidden');
    open = { kind: 'sheet', el, pane };
    markTriggers();
    redraw();
  }

  document.addEventListener('click', (e) => {
    const pop = e.target.closest('[data-pop]');
    if (pop) { openPop(pop.dataset.pop); return; }
    const sheet = e.target.closest('[data-sheet]');
    if (sheet) { openSheet(sheet.dataset.sheet); return; }
    if (e.target.closest('[data-close]')) { closeAll(); return; }
    const proxy = e.target.closest('[data-proxy]');                 // ⋯ menu on narrow screens: press the real button
    if (proxy) { closeAll(); document.getElementById(proxy.dataset.proxy).click(); return; }
    // a click outside an open popover closes it — except "sticky" ones (Objects: you draw on the view while
    // it is open) and sheets (the view is used while reading them)
    if (open && open.kind === 'pop' && !open.el.dataset.sticky && !open.el.contains(e.target)) closeAll();
  });

  document.addEventListener('keydown', (e) => {
    if (e.key !== 'Escape' || document.body.classList.contains('wizard-open')) return;
    if (/input|select|textarea/i.test(e.target.tagName)) return;
    closeAll();
  });

  // the model drawer: network | goal | training
  $('drawerBtn').onclick = () => {
    const d = $('drawer'), show = d.classList.contains('hidden');
    d.classList.toggle('hidden', !show);
    $('drawerBtn').classList.toggle('on', show);
    $('drawerBtn').textContent = show ? 'Model ▼' : 'Model ▲';
    redraw();
  };

  // ui.js announces a picked layer (network diagram, inspector arrows): show it in the sheet
  document.addEventListener('nsd:layer', () => openSheet('inspector', false));

  window.Shell = { openPop, openSheet, closeAll };
})();
