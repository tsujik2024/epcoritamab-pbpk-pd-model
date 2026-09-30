# epcoritamab PBPK/PD model

Attempt at reproducing Li et al., Clin Pharmacol Ther 2022 doi:10.1002/cpt.2729 (epcoritamab dose selection).
Equations and parameters are from the supplement (Table S1/S2, Figs S1-S6).

    pip install numpy scipy matplotlib
    python scripts/run_full_model.py 48 168      # step-up to 48 mg, 168 days, DLBCL + FL
    python scripts/run_dose_sweep.py DLBCL       # weekly doses 0.0128-384 mg (~2 min)
    python scripts/run_virtual_patients.py 20    # Table S2 variability (slow, see below)
    python tests/test_models.py

Plots go to `results/`.

## Layout
- `src/models/pk_submodel.py` SC depot + minimal PBPK
- `src/models/lymphocyte_trafficking.py` T/B cell trafficking, turnover, injection effect, homeostasis
- `src/models/binding_submodel.py` CD3 / CD20 / trimer binding in blood, spleen, node, lymph (+ tumor CD20 in the node)
- `src/models/tcell_activation.py` activated T cells (vATC, pATC)
- `src/models/tumor_model.py` tumor growth, T-cell killing, B-cell killing rates
- `src/models/full_model.py` couples everything (60 states), handles dosing and the TAD/Tp switches

## Things I had to guess (paper doesn't say)
- Trimers per B cell are divided by *baseline* B cells, not the live count (live count blows up when B cells are gone).
- Tumor-cell death terms in the binding equations (`kill_tumor`).
- Priming 0.16 mg / intermediate 0.8 mg, MW 149 kDa, bioavailability 1.
- k_kill_tumor is used for tumor killing (the text says k_kill_BC, the table has both).

## Known problems
- A typical patient always responds (tumor -> -100%). The paper's ORR at 48 mg is ~85%, from patient variability.
- Virtual patients are slow (~1 min each) and a few draws time out and are skipped.
- No hook-effect decline in the response curve, only in the binding-only sweep.
- The old `run_*` demos and test files from the first version don't match the new signatures.
