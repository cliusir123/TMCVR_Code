import math
import numpy as np
from common import OUTPUT, RESULT_TAG, make
from workflow import config as C
from final_tables import FAIL_RMSE, FIGNAME, TARGETS, load
METHODS = [('SNIS+CV', 'SNIS'), ('TGI-RWM+CV', 'TGI-RWM'), ('TGI-MALA+CV', 'TGI-MALA'), ('TESS+CV', 'TESS'), ('NeuTra+ZVz', 'NeuTra')]
TIE, WITHIN2 = (-math.log10(1.1), -math.log10(2.0))

def rows_of(t):
    o, d = load(t)
    names = C.TARGETS[t].get('tex_names') or [n for n, _, _ in make(t).test_functions]
    out = []
    for j, name in enumerate(names):
        prec, fail = ({}, {})
        for m, _ in METHODS:
            acc, rm = (d[m]['acc_per_s'][j], d[m]['rmse'][j])
            fail[m] = not (rm is not None and np.isfinite(rm) and (rm <= FAIL_RMSE))
            prec[m] = acc * rm ** 2 if fail[m] else acc
        best = max((prec[m] for m, _ in METHODS if not fail[m]))
        out.append((name, {m: math.log10(prec[m] / best) for m, _ in METHODS}, fail))
    return out

def summarize(per_target, weighted):
    res = []
    for m, lab in METHODS:
        if weighted:
            g = np.array([np.mean([r[1][m] for r in rows]) for rows in per_target.values()])
        else:
            g = np.array([r[1][m] for rows in per_target.values() for r in rows])
        allrows = [(t, r) for t, rows in per_target.items() for r in rows]
        gaps = np.array([r[1][m] for _, r in allrows])
        k = int(np.argmin(gaps))
        wt, wr = allrows[k]
        res.append(dict(method=lab, mean=float(g.mean()), sd=float(g.std(ddof=1)), worst=float(gaps[k]), worst_at=f'{FIGNAME[wt]} / ${wr[0]}$', within2=int((gaps >= WITHIN2).sum()), best=int((gaps >= TIE).sum()), n=len(gaps), failed=int(sum((r[2][m] for _, r in allrows)))))
    return res

def md_table(res, title, note):
    L = [f'### {title}', '', note, '', '| method | mean gap | x slower | sd | worst (target / function) | within 2x | best or tied | failed |', '|---|---|---|---|---|---|---|---|']
    for r in res:
        L.append(f"| {r['method']} | {r['mean']:.2f} | {10 ** (-r['mean']):.1f} | {r['sd']:.2f} | {r['worst']:.2f} ({r['worst_at']}) | {r['within2']}/{r['n']} | {r['best']}/{r['n']} | {r['failed']} |")
    return '\n'.join(L) + '\n'

def tex_table(A, B, nrows):

    def worst(r):
        t, f = r['worst_at'].split(' / ')
        return f"{r['worst']:.2f} ({t}, {f})"
    L = ['\\begin{table*}[t]', '\\centering', '\\small', '\\caption{Stability of the estimators: the gap $\\log_{10}(\\text{precision}/\\text{best of the row})$ of each method with its control variates to the most precise one, over the rows of Table~\\ref{tab:wallclock-full} ($0$: the best; $-1$: ten times less precise). Mean gap, the corresponding factor $10^{-\\text{mean}}$, standard deviation and worst row; number of rows within a factor $2$ of the best and within $10\\%$ of it (the bold rule); failures (RMSE above one posterior standard deviation) are counted at RMSE $=1$. (A) The main test function of each target. (B) All ' + str(nrows) + ' test functions; the mean and standard deviation are over the $12$ per-target means, so that every target has the same weight, the worst row and the counts over all rows.}', '\\label{tab:stability}', '\\begin{tabular}{@{}l rrr l rrr@{}}', '\\toprule', 'Method & mean gap & factor & sd & worst row & within $2\\times$ & best & failed \\\\']
    for lab, res in (('(A) main test function, $12$ rows', A), (f'(B) all test functions, ${nrows}$ rows, targets weighted equally', B)):
        L += ['\\midrule', f'\\multicolumn{{8}}{{@{{}}l}}{{\\emph{{{lab}}}}} \\\\']
        top = dict(mean=max((r['mean'] for r in res)), sd=min((r['sd'] for r in res)), worst=max((r['worst'] for r in res)), within2=max((r['within2'] for r in res)), best=max((r['best'] for r in res)), failed=min((r['failed'] for r in res)))

        def bf(k, r, txt):
            if r[k] != top[k]:
                return txt
            return f'{{\\boldmath{txt}}}' if txt.startswith('$') else f'\\textbf{{{txt}}}'
        for r in res:
            fac = 10 ** (-r['mean'])
            fac = f'{fac:.1f}' if fac < 10 else f'{fac:.0f}'
            w0, w1 = worst(r).split(' ', 1)
            L.append(f"{r['method']} & {bf('mean', r, f'${r[chr(109) + chr(101) + chr(97) + chr(110)]:.2f}$')} & {bf('mean', r, fac)} & {bf('sd', r, f'{r[chr(115) + chr(100)]:.2f}')} & {bf('worst', r, f'${w0}$')} {w1} & {bf('within2', r, str(r['within2']) + '/' + str(r['n']))} & {bf('best', r, str(r['best']) + '/' + str(r['n']))} & {r['failed']} \\\\")
    L += ['\\bottomrule', '\\end{tabular}', '\\end{table*}']
    return '\n'.join(L) + '\n'

def main():
    per = {t: rows_of(t) for t in TARGETS if load(t) is not None}
    main_rows = {}
    for t, rows in per.items():
        o, _ = load(t)
        h = C.TARGETS[t].get('main_fn', o['headline'])
        main_rows[t] = [rows[h]]
    nrows = sum((len(r) for r in per.values()))
    note_fail = 'Failures (RMSE > 1 sd) are counted at RMSE = 1.'
    out = [f"# Stability of the estimators: gap to the best of each row ({RESULT_TAG or 'untagged'})", '', 'gap = log10(precision / best of the row); 0 = best, -1 = 10x slower. within 2x: gap >= -log10 2; best or tied: gap >= -log10 1.1 (the bold rule).', '', md_table(summarize(main_rows, False), 'A. Main test function (12 rows)', note_fail), md_table(summarize(per, True), f'B. All test functions, every target weighted equally (mean and sd over the 12 per-target means; counts over all {nrows} rows)', note_fail), md_table(summarize(per, False), f"B'. All test functions, unweighted ({nrows} rows)", note_fail)]
    tex = tex_table(summarize(main_rows, False), summarize(per, True), nrows)
    (OUTPUT / 'paper' / f'table_stability{RESULT_TAG}.tex').write_text(tex)
    print(tex)
    p = OUTPUT.parent / 'results' / 'revision' / f'stability{RESULT_TAG}.md'
    p.write_text('\n'.join(out))
    print('\n'.join(out))
    print(p)
if __name__ == '__main__':
    main()
