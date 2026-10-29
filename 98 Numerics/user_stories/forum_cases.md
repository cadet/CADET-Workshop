# CADET forum: candidate cases for the fail compilation

I went through the forum systematically: ran ~65 keyword searches against the Discourse API, pulled 279 unique candidate threads, then fetched and read the 100 most numerics-relevant ones in full. Here's what's worth putting in the fail compilation.

## Tier 1 — the best teaching cases

**[#1431 — PCC optimization timing out on an HPC](https://forum.cadet-web.de/t/issues-with-running-pcc-file-with-cadet-process-on-the-high-power-computer-cluster-hpc/1431)** (2026, 11 posts)
The single richest thread. A map of succeeded/timed-out simulations over the parameter space, gdb traces showing IDA making no progress for 2 days, and two beautifully counter-intuitive observations:
- `Dax = 1e-6` hits a 5-min wall; `Dax = 1e-5` finishes in 14 s (stiffness, not work).
- **Raising `ncol`/`npar` from 25 to 50 made it *faster*** — under-resolution produced oscillations/negative values that increased stiffness.

Resolved by `max_newton_iter ≈ 100` + `max_step_size ≈ 1`, or `abstol = 1e-9`. Your own reply in there also points at the v6 solver-metadata output (time steps, BDF order) as a stiffness indicator — a natural hook for a diagnostics section.

**[#908 — Help with k_kin in SMA](https://forum.cadet-web.de/t/help-with-k-kin-in-sma-isotherm/908)** (2024, 11 posts)
A GoSilico `k_kin = 1.8e-8` translated literally into CADET gives `kd = 5.6e7`, `ka = 38.9`. Fails at the *first section switch* with NaNs during consistent initialization. The thread contains a **full IDA debug log** (steps, Newton iters, error-test fails, step sizes, residual norms) — ready-made material for teaching people how to read solver output. Two independent fixes work: `max_errtest_fail = 10` **or** `init_step_size = 1e-15`. Root cause is that such a tiny k_kin is really an *encoding of rapid equilibrium*, so `is_kinetic = False` is the right answer.

**[#703 — Error when using quasi-stationary binding](https://forum.cadet-web.de/t/error-when-using-quasi-stationary-binding/703)** (2023 + 2025 follow-up)
Pure consistent-initialization case. Works with `is_kinetic = True`, fails with `False`. Full init (`CONSISTENT_INIT_MODE = 1`) doesn't help; **lean init (mode 3) fixes it** because all concentrations are zero at t=0 so the algebraic equations of the rapid-equilibrium binding need not be initialized. Has an attached `.h5`. Bonus: the same user hits it again two years later in CADET-Process, answered with `simulator.solver_parameters.consistent_init_mode = 3`.

**[#1283 — DG discretization + EMPM binding](https://forum.cadet-web.de/t/error-with-dg-discretization-and-empm-binding-model/1283)** (2025)
The crash message is `double free or corruption (out)` / `munmap_chunk(): invalid pointer` — pure red herring. Actual cause: DG oscillations produce *small* negative concentrations, and the binding model evaluates `c^beta` with non-integer beta. Integer beta works, FV works. Perfect illustration that the error message and the cause can live in completely different places, plus the FV-vs-DG smoothness trade-off.

**[#644 — IDAS error, can't get simulation to run](https://forum.cadet-web.de/t/idas-error-cant-get-simulation-to-run/644)** (2023)
GIEX with pH-driven `kA` swinging over 7 orders of magnitude (6.89e-3 → 7.77e+4). `IDA_TOO_MUCH_WORK at t = 1736`. User had already tried tolerances, init step size, max steps, max Newton iters, reference concentrations. Resolved by relaxing tolerances to ~5e-5. Good example of "the model is genuinely stiff and you must decide what accuracy you actually need."

## Tier 2 — short, sharp, good as quick exercises

| Thread | Symptom | Real cause |
|---|---|---|
| [#743](https://forum.cadet-web.de/t/error-in-idasolve/743) | `IDA_CONV_FAIL at t = 0, h = 9.5e-13` | HIC `ks = 4.279` taken from a paper in mol/L, CADET wants mol/m³ — a **unit error presenting as a solver failure**. Full runnable script in the post. |
| [#622](https://forum.cadet-web.de/t/idas-convergence-failure-when-calculating-kinetic-sma-model/622) | Converges quasi-stationary, crashes kinetic | SMA characteristic charge ν = 68 in an exponent → floating-point underflow. **Reference concentrations fix it.** Ends with the suggestion that ref. concentrations should always be used. |
| [#841](https://forum.cadet-web.de/t/simulation-doesnt-converge-with-sma-and-non-integer-characteristic-charge/841) | Converges for ν = 1, 2; fails for ν = 1.69 | Salt concentration exactly 0, numerical noise dips it slightly negative, then `c^1.69`. Fix: non-zero initial salt + saturate `q` with salt. |
| [#469](https://forum.cadet-web.de/t/idasolve-error-ida-ill-input-tout-too-close-to-t0/469) | `IDA_ILL_INPUT: tout too close to t0` | A `linspace` solution time landing a float-epsilon away from a section time. Contains the 5-step recipe CADET-Process uses to sanitize solution times. |
| [#903](https://forum.cadet-web.de/t/error-in-dextrane-puls-simulation/903) | `returncode=3221225477`, empty stderr | `section_continuity = []` instead of `[0, 0]`. Silent access violation — nice "the error message gave me nothing" case. |
| [#870](https://forum.cadet-web.de/t/error-when-completing-simulation-with-polynomial-flow-ramp/870) | Kernel hangs on "busy" | Polynomial flow ramp starting at flow rate **exactly 0** breaks consistent initialization. Workaround: start at `1e-12`. |
| [#1273](https://forum.cadet-web.de/t/colloidal-isotherm-hangs/1273) | Simulation hangs, no error | `bpp = 0` → division by zero in the colloidal Jacobian; plus a protein radius mis-computed as ~4.5 m. Since fixed in CADET-Core #452. |
| [#1217](https://forum.cadet-web.de/t/cstr-causes-hanging-simulation-with-low-values/1217) | Hangs for `init_liquid_volume < 1e-4` | `flow_rate_filter` set equal to `flow_rate` — CSTR drains as fast as it fills. |
| [#1235](https://forum.cadet-web.de/t/negative-elution-with-langmuir-mpm/1235) | Negative outlet concentrations | MPM `Keq > 1e8` → unphysically square isotherm. Contains an excellent diagnostic method: **isolate the isotherm, plot q(c_p, c_0) analytically, and sanity-check parameter ranges before blaming the solver.** |
| [#1199](https://forum.cadet-web.de/t/simulation-failed-using-hic-isotherm/1199) | Never terminates with a CSTR in the flowsheet | CADET-Core 4.4.0 bug; fine in 5.0.3. Good "check your version first" case. |

## Tier 3 — background/context material for the chapter

- **[#714](https://forum.cadet-web.de/t/bad-allocation-error-when-increasing-number-of-discretization-steps/714)** — `bad allocation` at `ncol = 200`, actually memory exhaustion from `write_solution_bulk`. Contains your EOC formula, the "only set RELTOL/ABSTOL as small as required" advice, and Scott Altern's widely-cited rule of thumb (24 axial nodes/cm, 30 particle nodes; fewer during fitting).
- **[#465](https://forum.cadet-web.de/t/question-on-hyperthreading-in-cadet/465)** — Leweke's explanation of why time stepping is serial, the nested spatial parallelization, and **"10 sims × 1 thread beats 1 sim × 10 threads"**. Plus breakthrough-curve artefacts that only vanish above 150 axial / 100 radial nodes.
- **[#1087](https://forum.cadet-web.de/t/solver-nthreads-not-working-in-v5-0/1087)** — `solver.nthreads` silently does nothing because conda-forge v5+ ships with `ENABLE_THREADING=OFF`. Benchmark table shows 2.5–3× slowdowns vs 4.4. A real "my simulation got slow and it's not my model" case.
- **[#55](https://forum.cadet-web.de/t/discretization-error/55)** — Leweke's a-priori error-estimate explanation of why you cannot pick `h` from a target error without knowing C and k. Good framing for why mesh refinement studies are unavoidable.
- **[#664](https://forum.cadet-web.de/t/variation-of-column-porosity-inside-the-column/664)** — `ncol = 10000`, WENO order 1, **14 h runtime** chasing a result that was wrong because the wrong unit's outlet was being plotted. Cautionary tale about refining before verifying.
- **[#570](https://forum.cadet-web.de/t/cyclic-steady-state-in-cadet/570)** and **[#772](https://forum.cadet-web.de/t/section-dependent-isotherms-and-alternative-approaches/772)** — restarting a simulation from `last_state_y`. Both run through `consistent_init_mode` 0/2/3/4/6/7 and hit `IDA_ERR_FAIL at t = 0`; #772 shows the restarted state containing values that appear nowhere in the original solution.
- **[#1314](https://forum.cadet-web.de/t/how-to-solve-equilibrium-reactions-numerically-stable-and-efficient-with-sundials/1314)** — not CADET, but a clean worked discussion of stiffness from fast reactions, kinetic vs. rapid-equilibrium formulation, and the overdetermined-system/conserved-moieties fix.

## Patterns worth structuring the chapter around

Five recurring root causes cover nearly every thread:

1. **Parameters outside physical range** (units, ν = 68, Keq = 1e8, 4 m/s velocity, radius in metres) — the solver is the messenger, not the culprit. The diagnostic is to isolate and plot the isotherm.
2. **Missing scaling** — SMA/GIEX without reference concentrations or reference pH. Several threads resolve purely by introducing them.
3. **Consistent initialization** — quasi-stationary binding, zero initial flow rate, simulation restarts, section switches. The `CONSISTENT_INIT_MODE` knob is poorly known and solves several of these.
4. **Section/solution time bookkeeping** — float-epsilon collisions, empty `section_continuity`, solution times past the last section. Cheap to cause, opaque to diagnose.
5. **Under-resolution raising stiffness** — the #1431 observation that more cells can mean less wall-clock is the most counter-intuitive and most worth teaching.

Also notable: in roughly half of these, the **error message actively misleads** (`double free or corruption` for negative concentrations, `bad allocation` for output size, `returncode=3221225477` for a missing list, silent hangs for division by zero). A "how to read what CADET actually tells you" section seems warranted.
