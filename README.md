# Transformed Samplers with Variance Reduction: code for the paper

This repository is the code behind every table and figure of the paper:
- normalizing-flow transports;
- the TGI-RWM, TGI-MALA, TESS, NeuTra and SNIS estimators with their control variates;
- the NUTS and GI-MALA baselines;
- the scripts that produce the tables and figures.

## Requirements

| What | Version used |
|---|---|
| Python | 3.11 |
| PyTorch | 2.9 |
| NumPy | 2.2 |
| SciPy | 1.16 |
| Matplotlib | 3.10 |
| JAX | 0.10 |
| NumPyro | 0.22 |

- **JAX and NumPyro** are used only for NUTS (reference values and the NUTS baseline).
- **GPU:** a CUDA GPU is strongly recommended. The flow-based estimators run thousands of chains in parallel, and the wall-clock comparison was timed on one GPU.
- **Precision:** everything runs in float64.

```bash
pip install torch numpy scipy matplotlib jax numpyro
```

## Layout

```
distributions/  target densities: synthetic targets and real posteriors (posteriordb ports, ODE models, phi^4)
transports/     RealNVP flows, training (reverse KL, forward-KL pilot, tail penalty), analytic transports, diagnostics
mcmc/           latent target pi_z, Gaussian-invariant proposals, batched sampler, step-size adaptation (J)
hermite/        Hermite dictionary, coefficients, pruning, control-variate recorder
methods/        TGI (RWM/MALA), TESS, NeuTra, NUTS, GI-MALA, zero-variance control variates
experiments/    shared metrics
workflow/       the scripts below; workflow/config.py holds every setting (targets, training, protocol)
data/           data of the real posteriors (posteriordb, German credit)
```

Run every script from the repository root as `python workflow/<script>.py`.

**Where results go:**
- trained flows: `results/flows/<target>/`
- reference values: `results/references/`
- raw results: `results/raw/<target>/`
- tables and figures: `output/paper/`

**Environment variables:**
- `GTI_TAG` is the suffix of the result files. The paper uses `GTI_TAG=_c5k`.
- `GTI_FLOWS` selects another set of flows. The default is `flows`.

**Targets used in the paper:**
- synthetic: `banana`, `funnel`, `gmix_d10_h6`, `tmix_d10_h6`;
- real: `pdb_garch`, `pdb_one_comp`, `lotka_volterra`, `pdb_dugongs`, `pdb_gp_pois`, `eight_schools`, `sparse_logistic_german`, `phi4_N32`.

Below, `$SYN` stands for the four synthetic targets, `$REAL` for the eight real posteriors, and `$ALL` for all twelve.

```bash
SYN="banana funnel gmix_d10_h6 tmix_d10_h6"
REAL="pdb_garch pdb_one_comp lotka_volterra pdb_dugongs pdb_gp_pois eight_schools sparse_logistic_german phi4_N32"
ALL="$SYN $REAL"
```

## 1. Train the flows (once per target)

```bash
for t in $ALL; do python workflow/train.py --target $t; done
```

Each target is trained with its own settings from `workflow/config.py`: scheme, network, tail penalty, steps, batch size and learning rate. The multi-start NUTS pilot of the multimodal targets is part of the training, and its cost is counted. Every later experiment uses the last checkpoint and never retrains.

## 2. Reference values of the real posteriors

```bash
for t in $REAL; do python workflow/reference.py --target $t; done
```

These are long NUTS runs with zero-variance control variates. The synthetic targets need no reference: their values are exact.

## 3. Pilots: step sizes and NeuTra's number of leapfrog steps

```bash
for t in $ALL; do
  GTI_TAG=_c5k python workflow/pilot_batch.py  --target $t --what gamma   # gamma of TGI-RWM / TGI-MALA
  GTI_TAG=_c5k python workflow/neutra_pilot.py --target $t                # L of NeuTra
done
```

The pilots use chains of cost $10^4$, as in the paper. Their cost is not counted.

## 4. Main comparison: precision per second

```bash
for t in $ALL; do
  extra=""; [ $t = sparse_logistic_german ] && extra="--snis-chunk 2e5"
  GTI_TAG=_c5k python workflow/chainstats.py --target $t --chain-len 5000 $extra
done
GTI_TAG=_c5k python workflow/delta_table.py        # defensive independence step (delta = 0.01)
```

**What one run does:**
- Each estimate uses $N=10^6$ target evaluations, and each method makes 100 estimates.
- Each estimate averages 200 chains of cost 5000.
- All methods run on the same flow and are timed on the GPU.
- The output is `results/raw/<target>/chainstats_c5k.{json,npz}`.

**Timing:** do not run other GPU jobs at the same time.

## 5. Baselines and error against total cost

```bash
for t in $REAL; do python workflow/benchmark.py --target $t --methods nuts gi_mala; done
python workflow/benchmark.py --target banana --methods tgi_rwm tess nuts neutra snis
```

The NUTS and GI-MALA columns of the RMSE table come from the first loop. The second command gives the banana curves of the error against total cost (Figure A2). NUTS is not run on the sparse logistic regression and the $\phi^4$ field; its error there is extrapolated from the reference run.

## 6. Fixed-flow chain experiments (appendix)

```bash
python workflow/samplers_family.py --target banana --main-fn            # sampler selection across the shear family
python workflow/family_ess.py                                           # ESS_IS/N across the family
python workflow/chains.py --target banana --exp gamma   --transport accurate
python workflow/chains.py --target banana --exp pruning --transport accurate
for t in $ALL; do python workflow/chains.py --target $t --exp gamma --transport flow; done
python workflow/j_recompute.py $ALL                                     # gamma selected by J
python workflow/gamma_ratio.py $ALL                                     # J's gamma against the best gamma
python workflow/train_ablation.py --target banana                       # training ablation
```

These experiments use 500 chains of 500 burn-in steps plus 5000 recorded steps on a fixed transport. They are not timed.

## 7. Tables and figures

```bash
GTI_TAG=_c5k python workflow/final_tables.py --main5      # main table (precision per second, VRF in parentheses)
GTI_TAG=_c5k python workflow/final_tables.py --full       # appendix table: every method, every test function
GTI_TAG=_c5k python workflow/final_tables.py --nuts       # RMSE against NUTS+ZV and GI-MALA+CV
GTI_TAG=_c5k python workflow/final_tables.py --boxplots   # the 100 estimates of the main test functions
GTI_TAG=_c5k python workflow/final_tables.py              # scatter of the ratio against max w*N (fig_maxw)
GTI_TAG=_c5k python workflow/stability.py                 # gap of each estimator to the best of its row
python workflow/appendix_figs.py --a1 --a2                # banana analytic family; banana error vs cost and flow grids
```


