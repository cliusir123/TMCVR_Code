import json
import sys
from pathlib import Path
import numpy as np
from common import make, raw_dir, setup, truth
from workflow import config as C

def scan(t):
    tr = truth(make(t))
    var = np.array(tr['var'])
    rows = [json.loads(l) for l in open(raw_dir(t) / 'gamma_flow_rwm.jsonl')]
    runs = sorted([r for r in rows if r['kind'] == 'run'], key=lambda r: r['gamma'])
    g = np.array([r['gamma'] for r in runs])
    bm = np.array([np.array(r['chain']['main']['bm_fixed']).mean(0) for r in runs])
    eff = var[None, :] / bm
    jp = Path(__file__).resolve().parent.parent / 'results' / 'revision' / 'j_gamma_lam1.json'
    js = json.loads(jp.read_text()) if jp.exists() else {}
    jg = float(np.median(js[t] if t in js else json.loads((raw_dir(t) / 'j_gamma.json').read_text())['flow:rwm']))
    return (g, eff, jg)

def row(t, g, e, jg):
    k = int(np.argmax(e))
    ej = np.interp(np.log(jg), np.log(g), e)
    band = g >= 0.7
    return f'| {t} | {jg:.3f} | {g[k]:.3g} | {ej / e[k]:.2f} | {e[g == 1.0][0] / e[k]:.2f} | {e[band].min() / e[band].max():.2f} |'

def main():
    setup(C.DTYPE)
    head = '| target | gamma_J | gamma* | R(gamma_J,gamma*) | R(1,gamma*) | flat [0.7,1] |\n|---|---|---|---|---|---|'
    out = ['## main test function', head]
    med = ['## median over the test functions of the per-function ratios', head]
    for t in sys.argv[1:]:
        g, eff, jg = scan(t)
        h = C.TARGETS[t]['headline']
        out.append(row(t, g, eff[:, h], jg))
        rs = []
        for j in range(eff.shape[1]):
            e = eff[:, j]
            k = int(np.argmax(e))
            rs.append((g[k], np.interp(np.log(jg), np.log(g), e) / e[k], e[g == 1.0][0] / e[k], e[g >= 0.7].min() / e[g >= 0.7].max()))
        rs = np.array(rs)
        med.append(f'| {t} | {jg:.3f} | {np.median(rs[:, 0]):.3g} | {np.median(rs[:, 1]):.2f} (min {rs[:, 1].min():.2f}) | {np.median(rs[:, 2]):.2f} | {np.median(rs[:, 3]):.2f} |')
    text = '\n'.join(out + [''] + med) + '\n'
    p = Path(__file__).resolve().parent.parent / 'results' / 'revision' / 'gamma_ratio.md'
    p.write_text(text)
    print(text)
if __name__ == '__main__':
    main()
