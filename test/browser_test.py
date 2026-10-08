"""End-to-end checks of the page in a real browser (Firefox, headless) over plain WebDriver HTTP.

    python3 test/browser_test.py                 # serves the repo, starts geckodriver, runs every check
    python3 test/browser_test.py goals strip     # only the checks whose name contains one of the words

Needs Firefox and geckodriver on the PATH (or GECKODRIVER=/path/to/geckodriver). No Python packages:
only the standard library. Each check loads the page fresh, drives it the way a user would (clicks,
pointer drags, keys) and asserts on the app's state (window.__app) and on the DOM. A check fails on a
wrong value or on any JavaScript error the page reports.
"""
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


class Browser:
    """The few WebDriver calls the checks need."""

    def __init__(self, driver, page, width=1400, height=900):
        self.driver, self.page = driver, page
        caps = {'capabilities': {'alwaysMatch': {'moz:firefoxOptions': {'args': ['-headless', f'--width={width}', f'--height={height}']}}}}
        self.sid = self._req('POST', '/session', caps)['sessionId']
        self.n = 0

    def _req(self, method, path, body=None):
        data = None if body is None else json.dumps(body).encode()
        r = urllib.request.Request(self.driver + path, method=method, data=data, headers={'Content-Type': 'application/json'})
        return json.loads(urllib.request.urlopen(r, timeout=120).read())['value']

    def js(self, code):
        return self._req('POST', f'/session/{self.sid}/execute/sync', {'script': code, 'args': []})

    def open(self, query='#workspace'):
        """A real reload every time (a new query string), with an error trap installed."""
        self.n += 1
        self._req('POST', f'/session/{self.sid}/url', {'url': 'about:blank'})
        q = f'?t={self.n}' + (query if query.startswith('#') else '&' + query)
        self._req('POST', f'/session/{self.sid}/url', {'url': self.page + q})
        time.sleep(1.4)
        self.js("window.__errs=[];window.addEventListener('error',e=>window.__errs.push(e.message+' @'+e.lineno))")

    def click(self, sel):
        self.js(f"document.querySelector({json.dumps(sel)}).dispatchEvent(new MouseEvent('click',{{bubbles:true}}))")

    def set(self, el_id, value, event='input'):
        self.js(f"const e=document.getElementById('{el_id}'); e.value={json.dumps(value)}; e.dispatchEvent(new Event('{event}',{{bubbles:true}}))")

    def drag(self, x0, y0, x1, y1, canvas='c2d', button=0, steps=8):
        """Pointer drag between two points given in world coordinates of the 2D view."""
        self.js(f"""const c=document.getElementById('{canvas}'),r=c.getBoundingClientRect(),v=__app.viz2;
          const [a,b]=v.toScreen({x0},{y0}),[p,q]=v.toScreen({x1},{y1});
          const E=(t,x,y)=>c.dispatchEvent(new PointerEvent(t,{{bubbles:true,button:{button},clientX:r.left+x,clientY:r.top+y,pointerId:1}}));
          E('pointerdown',a,b); for(let k=1;k<={steps};k++) E('pointermove',a+(p-a)*k/{steps},b+(q-b)*k/{steps}); E('pointerup',p,q);""")

    def train(self, seconds):
        self.js("document.getElementById('trainPlay').click()")
        time.sleep(seconds)
        self.js("document.getElementById('trainPlay').click()")
        time.sleep(0.4)

    def screenshot(self, path):
        with open(path, 'wb') as f:
            f.write(base64.b64decode(self._req('GET', f'/session/{self.sid}/screenshot')))

    def errors(self):
        return self.js('return window.__errs || []')

    def close(self):
        self._req('DELETE', f'/session/{self.sid}')


# ---------------------------------------------------------------------------------------------------------
# The checks.  Each gets a Browser and raises AssertionError on a failure.
# ---------------------------------------------------------------------------------------------------------

def check_guided_start(b):
    b.open('')
    b.click('#wzStart'); time.sleep(0.5)
    b.click('[data-key=dim][data-value="2"]'); time.sleep(0.5)
    b.click('[data-key=problem][data-value=classify]'); time.sleep(0.8)
    b.click('.wz-data[data-id=circles]'); time.sleep(0.3)
    b.set('wzNoise', 0.12); b.set('wzN', 300)
    b.click('#wzDataUse'); time.sleep(0.8)
    assert '★ recommended' in b.js("return document.getElementById('wzNetInfo').textContent")
    b.js("document.getElementById('wzTour').checked=false")
    b.click('#wzFinish'); time.sleep(1.0)
    st = b.js("const S=__app.S;return [S.train.target,S.train.dataset,S.train.nData,S.train.noise,S.net.layers,S.net.width]")
    assert st == ['classify', 'circles', 300, 0.12, 4, 4], st
    l0 = b.js('const T=__app.trainer;T.step();return T.lossHistory.at(-1)')
    l1 = b.js('const T=__app.trainer;for(let i=0;i<1500;i++)T.step();return T.lossHistory.at(-1)')
    assert l1 < l0 * 0.05, (l0, l1)


def check_guided_goals_and_tour(b):
    b.open('')
    b.js('localStorage.clear()')
    b.click('#wzStart'); time.sleep(0.5)
    b.click('[data-key=dim][data-value="2"]'); time.sleep(0.5)
    b.click('[data-key=problem][data-value=goals]'); time.sleep(0.8)
    b.click('.wz-data[data-id=moons]'); time.sleep(0.3); b.click('#wzDataUse'); time.sleep(0.8)
    assert b.js("return document.getElementById('wzTour').checked") is True
    b.click('#wzFinish'); time.sleep(1.6)
    assert b.js('return __app.S.train.target') == 'goals'
    assert b.js('return Tour.active') is True
    seen = 0
    while b.js('return Tour.active') and seen < 20:
        b.js("document.querySelector('[data-t=next]').click()"); time.sleep(0.3); seen += 1
    assert 8 <= seen <= 13, seen


def check_shell_panels(b):
    b.open()
    b.click('#viewBtn'); time.sleep(0.2); b.click('#c2d'); time.sleep(0.2)
    assert not b.js("return document.getElementById('viewPop').classList.contains('hidden')"), 'View must stay open'
    b.click('#objBtn'); time.sleep(0.2)
    assert b.js("return document.getElementById('viewPop').classList.contains('hidden')"), 'one popover at a time'
    b.click('.pipe .pi[data-layer="1"]'); time.sleep(0.5)
    assert b.js("return document.getElementById('sheetTitle').textContent") == 'Layer inspector'
    assert b.js("return document.getElementById('wCanvas').width") > 100
    b.js("document.dispatchEvent(new KeyboardEvent('keydown',{key:'Escape',bubbles:true}))"); time.sleep(0.2)
    assert b.js("return document.getElementById('sheet').classList.contains('hidden')")


def check_strip(b):
    b.open()
    b.click('.pipe .pn[data-stage="3"]'); time.sleep(2.5)
    t = b.js('return __app.t')
    assert abs(t - 3) < 0.05, t
    n0 = b.js('return __app.net.nLayers')
    b.click('#layerPlus'); time.sleep(0.4)
    n = b.js("return [__app.net.nLayers, document.querySelectorAll('.pipe .pb').length]")
    assert n == [n0 + 1, n0 + 1], n
    b.click('#layerMinus'); time.sleep(0.4)
    assert b.js('return __app.net.nLayers') == n0


def check_objects(b):
    b.open()
    b.click('#objBtn'); time.sleep(0.2)
    b.click('#presetBtns [data-preset="star"]'); time.sleep(0.3)
    for k, v in (('scale', 1.5), ('off0', -0.8), ('n', 120)):
        b.js(f"const e=document.querySelector('[data-edit=\"{k}\"]'); e.value={v}; e.dispatchEvent(new Event('input',{{bubbles:true}}))")
    o = b.js('const o=__app.S.objects.at(-1); return [o.points.length,o.scale,o.offset[0]]')
    assert o == [120, 1.5, -0.8], o
    b.click('[data-edit="log"]'); time.sleep(0.8)
    assert b.js("return document.querySelectorAll('#objLog tr[data-s]').length") == b.js('return 2 * __app.net.nLayers + 1')   # one row per stage
    # move an object on the view, then resize it by its corner (the page opens without objects: add one)
    b.open()
    assert b.js('return __app.S.objects.length') == 0
    b.click('#objBtn'); time.sleep(0.2); b.click('#presetBtns [data-preset="star"]'); time.sleep(0.3); b.click('#objBtn'); time.sleep(0.2)
    p = b.js('return __app.S.objects[0].points[0]')
    b.drag(p[0], p[1], p[0] + 0.5, p[1] - 0.8); time.sleep(0.3)
    assert [round(v, 2) for v in b.js('return __app.S.objects[0].offset')] == [0.5, -0.8]
    c = b.js('const P=__app.S.objects[0].points; return [Math.max(...P.map(p=>p[0]))+0.04, Math.min(...P.map(p=>p[1]))-0.04]')
    b.drag(c[0], c[1], c[0] + 0.5, c[1] - 0.5); time.sleep(0.3)
    assert b.js('return __app.S.objects[0].scale') > 1.5
    # one bin clears objects and pins
    b.js("__app.S.pins.push({x:[0.2,0.3],y:[1,1]})"); b.click('#objBtn'); time.sleep(0.2); b.click('#clearObjects'); time.sleep(0.2)
    assert b.js('return [__app.S.objects.length, __app.S.pins.length]') == [0, 0]


def check_goals(b):
    b.open()
    b.click('#drawerBtn'); time.sleep(0.2)
    b.set('target', 'goals', 'change'); time.sleep(0.3)
    b.js("document.getElementById('recommendBtn').click()"); time.sleep(0.3)
    b.train(6)
    st = b.js("return document.getElementById('trainStatus').textContent")
    acc = float(st.split('accuracy ')[1].split('%')[0]) if 'accuracy ' in st else 0
    assert acc > 95, st
    b.js('__app.scrubTo(2*__app.net.nLayers)'); time.sleep(0.6)
    b.drag(1, 0, 1, 1.5); time.sleep(0.3)
    g = b.js('const g=__app.S.train.goals[0]; return [g.target, Array.from(__app.trainer.tasks[0].find(q=>q.goal===g.id).y)]')
    assert g[0] == [1, 1.5] and g[1] == [1, 1.5], g
    b.click('#goalAdd [data-kind=objectStay]'); time.sleep(0.2); b.click('#goalAdd [data-kind=classSplit]'); time.sleep(0.2)
    assert set(b.js("return __app.trainer.tasks[0].map(q=>q.type)")) == {'mse', 'ce'}
    n_goals = len(b.js('return __app.S.train.goals'))
    exp = b.js('return JSON.stringify(__app.serializeState())')
    b.js(f'__app.importState({json.dumps(exp)})'); time.sleep(0.4)
    assert len(b.js('return __app.S.train.goals')) == n_goals


def check_default_task_and_classes(b):
    b.open()
    st = b.js("const S=__app.S; return [S.train.target, S.train.dataset, S.train.nClasses, S.objects.length, S.train.goals.length]")
    assert st == ['goals', 'spirals', 3, 0, 3], st
    b.train(5)
    assert b.js('return __app.trainer.step_') > 100       # ▶ Train does something on a fresh page
    b.click('#drawerBtn'); time.sleep(0.2)
    b.set('nClasses', '4'); time.sleep(0.3)
    labels = b.js('return [...new Set(__app.trainer.tasks[0].map(q=>q.label))].sort()')
    targets = b.js('return __app.S.train.goals.map(g=>g.target.map(v=>Math.round(v*100)/100))')
    assert labels == [0, 1, 2, 3] and targets == [[1, 0], [0, 1], [-1, 0], [0, -1]], (labels, targets)
    b.set('nData', '1000'); time.sleep(0.3)
    assert b.js('return __app.trainer.tasks[0].length') == 1000
    b.set('dataset', 'moons', 'change'); time.sleep(0.3)        # moons only come in two
    assert b.js('return __app.S.train.goals.length') == 2
    b.set('dataset', 'spirals', 'change'); b.set('target', 'classify', 'change'); time.sleep(0.3)
    assert b.js("return [document.getElementById('nClasses').disabled, new Set(__app.trainer.tasks[0].map(q=>q.y)).size]") == [True, 2]


def check_pins_goal(b):
    b.open()
    b.click('#drawerBtn'); time.sleep(0.2)
    b.set('target', 'goals', 'change'); time.sleep(0.3)
    b.click('#objBtn'); time.sleep(0.2)
    b.js("[...document.querySelectorAll('#toolbar button')].find(x=>x.textContent==='Pin').click()"); time.sleep(1.2)
    assert b.js('return __app.S.train.target') == 'goals', 'the Pin tool must not switch the task away from my goals'
    assert 'pins' in b.js('return __app.S.train.goals.map(g=>g.kind)')
    b.drag(0.3, 0.2, 0.9, -0.7); time.sleep(0.3)
    assert len(b.js('return __app.S.pins')) == 1
    pin = b.js('const g=__app.S.train.goals.find(g=>g.kind==="pins"); return __app.trainer.tasks[0].filter(q=>q.goal===g.id).length')
    assert pin == 1, pin
    # alone (the default class goals would pull the same spot elsewhere), the pin is met
    while b.js("return __app.S.train.goals.some(g=>g.kind!=='pins')"):
        b.js("const g=__app.S.train.goals.find(g=>g.kind!=='pins'); document.querySelector(`[data-g=\"${g.id}\"][data-k=remove]`).click()")
        time.sleep(0.2)
    b.train(3)
    err = b.js('const p=__app.S.pins[0], o=__app.net.predict(p.x); return Math.hypot(o[0]-p.y[0], o[1]-p.y[1])')
    assert err < 0.05, err


def check_timeline(b):
    b.open()
    b.set('target', 'classify', 'change')
    b.train(4)
    steps = b.js('return __app.historySteps')
    assert len(steps) > 10
    b.js("""const c=document.getElementById('lossSpark'),r=c.getBoundingClientRect();
      c.dispatchEvent(new PointerEvent('pointerdown',{bubbles:true,clientX:r.left+r.width*0.1,clientY:r.top+5})); window.dispatchEvent(new PointerEvent('pointerup',{}));""")
    time.sleep(0.3)
    i = b.js('return __app.viewIdx')
    assert i is not None and b.js(f'return __app.net.theta[0] === __app.history[{i}].theta[0]')
    b.click('#liveBtn'); time.sleep(0.2)
    assert b.js('return __app.viewIdx') is None


def check_view_background_and_tilt(b):
    b.open()
    b.js("__app.applyGuided({dim:2,problem:'classify',data:{dataset:'moons',n:300,noise:0.08,seed:0},network:{layers:4,width:2,act:'tanh',outputLinear:true}})")
    b.js('const T=__app.trainer; for(let i=0;i<1500;i++) T.step(); __app.computeTraces();')
    # in exact 2D stages the background is the network's own decision
    agree = b.js("""const A=__app,tr=A.traces,d=tr.ds.find(x=>x.role==='data'); let ok=0,n=0;
      for (let s=0;s<tr.nStages;s++) for (let i=0;i<d.st.length;i++){ const o=A.restOfNetwork(s,[d.proj[s][2*i],d.proj[s][2*i+1]]), y=A.net.predict(d.pts[i]); n++; if((o[0]>o[1])===(y[0]>y[1])) ok++; }
      return ok/n""")
    assert agree == 1, agree
    b.js("__app.applyGuided({dim:2,problem:'classify',data:{dataset:'circles',n:300,noise:0.08,seed:0},network:{layers:4,width:4,act:'tanh',outputLinear:true}})")
    b.js('__app.scrubTo(4)'); time.sleep(0.4)
    b.js("""const c=document.getElementById('c2d'),r=c.getBoundingClientRect(),x=r.left+r.width/2,y=r.top+r.height/2;
      c.dispatchEvent(new PointerEvent('pointerdown',{bubbles:true,button:2,clientX:x,clientY:y,pointerId:1}));
      c.dispatchEvent(new PointerEvent('pointermove',{bubbles:true,clientX:x+100,clientY:y-60,pointerId:1}));
      c.dispatchEvent(new PointerEvent('pointerup',{bubbles:true,clientX:x+100,clientY:y-60,pointerId:1}));""")
    time.sleep(0.3)
    assert not b.js("return document.getElementById('gizmo').classList.contains('hidden')"), 'tilt shows the axes'


def check_dimension_switch(b):
    b.open()
    b.click('#dim3'); time.sleep(1.0); b.click('#drawerBtn'); time.sleep(0.3); b.click('#analysisBtn'); time.sleep(0.3)
    b.click('#dim2'); time.sleep(0.6)
    w = b.js("const v=__app.viz2,r=document.getElementById('c2d').getBoundingClientRect(); return [Math.round(v.w),Math.round(r.width)]")
    assert w[0] == w[1] and w[0] > 100, w     # the hidden 2D canvas must not shrink to 10 px


def check_basis_box_and_inspector(b):
    b.open()
    b.js("__app.applyGuided({dim:2,problem:'none',network:{layers:3,width:2,act:'tanh',outputLinear:false}})"); time.sleep(0.4)
    b.js('__app.scrubTo(3)'); time.sleep(0.4)
    box = b.js("return document.getElementById('basisBox').innerText")
    z = b.js("return Array.from(__app.net.stages([1,0])[3]).map(v=>v.toFixed(2).replace('-','−')).join(', ')")
    assert f'({z})' in box, (box, z)
    b.click('.pipe .pi[data-layer="1"]'); time.sleep(0.6)
    vals = b.js("return [...document.querySelectorAll('#inspExample .ex-col')].map(c=>[...c.querySelectorAll('span:not(.lbl)')].map(s=>s.textContent))")
    exp = b.js("const L=__app.net.layout[0],t=__app.net.theta; return [0,1].map(i=>(t[L.w+i*L.nin]+t[L.b+i]).toFixed(2).replace('-','−'))")
    assert vals[0] == ['1.00', '0.00'] and vals[3] == exp, (vals, exp)


def check_narrow_screen(b):
    b.open()
    r = b.js("const r=document.getElementById('modelStrip').getBoundingClientRect(); return [r.bottom, innerHeight, document.documentElement.scrollWidth, innerWidth]")
    assert r[0] <= r[1] + 1 and r[2] <= r[3], r


CHECKS = [check_guided_start, check_guided_goals_and_tour, check_shell_panels, check_strip, check_objects, check_goals, check_default_task_and_classes,
          check_pins_goal, check_timeline, check_view_background_and_tilt, check_dimension_switch, check_basis_box_and_inspector]


def main():
    words = sys.argv[1:]
    gecko = os.environ.get('GECKODRIVER') or shutil.which('geckodriver') or '/snap/bin/geckodriver'
    web_port, drv_port = free_port(), free_port()
    tmp = tempfile.mkdtemp()
    env = dict(os.environ, TMPDIR=os.environ.get('TMPDIR', tmp))
    server = subprocess.Popen([sys.executable, '-m', 'http.server', str(web_port), '--bind', '127.0.0.1'], cwd=ROOT,
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    driver = subprocess.Popen([gecko, '--port', str(drv_port)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=env)
    time.sleep(2.5)
    page, drv = f'http://127.0.0.1:{web_port}/index.html', f'http://127.0.0.1:{drv_port}'
    failed = 0
    try:
        todo = [c for c in CHECKS if not words or any(w in c.__name__ for w in words)]
        narrow = not words or any(w in 'check_narrow_screen' for w in words)
        for check in todo + ([check_narrow_screen] if narrow else []):
            b = Browser(drv, page, *((500, 760) if check is check_narrow_screen else (1400, 900)))
            t0 = time.time()
            try:
                check(b)
                errs = b.errors()
                assert not errs, f'JavaScript errors: {errs}'
                print(f'ok    {check.__name__[6:]}  ({time.time() - t0:.1f}s)')
            except Exception as e:                     # noqa: BLE001 — report and keep going
                failed += 1
                shot = os.path.join(tmp, f'{check.__name__}.png')
                try:
                    b.screenshot(shot)
                except Exception:                      # noqa: BLE001
                    shot = '(no screenshot)'
                print(f'FAIL  {check.__name__[6:]}: {e!r}\n      screenshot: {shot}')
            finally:
                b.close()
    finally:
        driver.terminate(); server.terminate()
    print('all browser checks passed' if not failed else f'{failed} check(s) failed')
    sys.exit(1 if failed else 0)


if __name__ == '__main__':
    main()
