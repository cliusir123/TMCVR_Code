import json
import sys
from pathlib import Path
from common import make, setup
from workflow import config as C
from chains import j_gamma, transport
from mcmc.adaptation import J_LAMBDA

def main():
    setup(C.DTYPE)
    p = Path(__file__).resolve().parent.parent / 'results' / 'revision' / f'j_gamma_lam{J_LAMBDA:g}.json'
    out = json.loads(p.read_text()) if p.exists() else {}
    for t in sys.argv[1:]:
        target = make(t)
        T, _, _ = transport(t, target, 'flow')
        out[t] = j_gamma(target, T, 'rwm', dict(C.INDEP, delta=0.0))
        print(t, [round(g, 3) for g in out[t]], flush=True)
        p.write_text(json.dumps(out, indent=1))
if __name__ == '__main__':
    main()
