---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# 99.7g Diagnosis: My Simulation Is Slow

## Contents

**Tools:** ...

**Python used:** CADET-Python, ...

**Introduces:** ...

1. ...
2. ...
...

...

+++

## Diagnosis — "my simulation is slow"

> **DRAFT.** Structure and argument only. Nothing here is measured yet; every number is a placeholder and every prediction below is written down so it can be falsified.

Every other notebook in this chapter isolates one knob in a clean setting. That is what made the findings trustworthy, and it is also why none of them looks like anybody's actual problem. This one goes the other way: **one realistically slow job, diagnosed end to end**, using what the rest of the chapter established.

The chapter's result makes this tractable. Stiffness is not the answer — fast binding, fast reactions, small particles and small dead volumes are all free. So whatever is costing you is on a short list, and the list is ordered.

+++

## The job

A realistic SMA gradient elution that someone would actually write, containing **several ordinary misconfigurations at once** — none of them obviously wrong, all of them invisible in the result.

| | as written | |
|---|---|---|
| gradient | 60 constant sections | should be one linear ramp |
| grid | 4× finer than needed | nothing tells you it is |
| `RELTOL` | default, ~3 decades tighter than the question needs | — |
| output | full bulk + particle solution written | the outlet is what gets looked at |

**The point of stacking them is that they multiply.** Three "small" over-specifications are not three problems, they are a product — which is how a job that should take two minutes takes forty.

**Design note / open decision.** SMA gradient elution is the proposal because the staircase gradient is so common. An SMB switching schedule would exercise the same mechanism harder. Pick one before building.

+++

```python
# TODO: utility/setting_slow_job.py
#   get_model(**overrides) -> the misconfigured baseline, with every knob
#                             in the ladder below exposed as an override
import utility.slow_job_explorer as diagnosis

diagnosis.show_baseline()        # chromatogram + wall clock + step count
```

+++

## Step 0 — rule out what everyone blames first

The instinct is to blame the chemistry: the binding is too fast, the reaction is too fast, make the rate constants gentler.

That is measured, and it is wrong. Stiffness 1 swept binding over fourteen decades for 1.3 % more steps; Stiffness 8 swept a reaction over eleven decades for 1 %; Stiffness 9 swept a dead volume over eight for 0.9 %. **The rate constants are not where the time went.** One cell to say so, then move on.

*(This is also the cell that earns the rest of the chapter its place in the session.)*

+++

## The ladder

Sorted by payoff per unit of effort, not by how interesting it is.

| rung | question | expected from the chapter |
|---|---|---|
| 1 | How many sections? Is that staircase deliberate? | ×13 measured in Stiffness 6 |
| 2 | Halve the grid. Did a digit you care about move? Repeat. | ×9 per step, and fewer steps too |
| 3 | Loosen `RELTOL` a decade at a time until it does. | ×5 for four decades |
| 4 | What are you writing out? | **unmeasured — see below** |
| 5 | Only now, look at the model. | mostly a dead end, see Stiffness 4/5 |

**The rule that makes this legitimate:** every rung reports the chromatogram error against the original. A speed-up that changes the answer is not a speed-up, it is a different simulation. Reuse the normalized RMSE the rest of the chapter uses, and state the threshold up front — *before* seeing which rungs pass.

+++

```python
# TODO: run the rungs cumulatively, each one keeping the previous fix.
#       Columns: rung | wall clock | steps | cumulative speed-up | chromatogram error
diagnosis.ladder()
```

+++

## Predictions, written down first

So that the measurement can falsify them, as it has repeatedly in this chapter.

1. **Sections dominate.** The staircase alone is worth more than everything else combined.
2. **The grid is second and compounds**, because it pays twice — per step *and* in steps.
3. **Output volume is the surprise.** It is the one rung that costs *wall clock without costing steps*, so every metric this chapter collects is blind to it. I suspect it is larger than people think; I have not measured it, and it may well be negligible.
4. **The model rung is a dead end.** Stiffness 5 found LRMP took *more* steps than GRM (2023 vs 1465), and Stiffness 4 found refining the particle grid *halved* them. Coarsening the physics is not reliably faster.
5. **Total: somewhere around ×20**, and no single fix explains it.

If (1) and (2) hold and (3) is negligible, the notebook gets shorter and the chapter's existing measurements carry it. If (3) is large, that is a new result and it belongs back in the conclusion.

+++

## To build

- `utility/setting_slow_job.py` — the misconfigured baseline, every ladder knob as an override
- `utility/slow_job_explorer.py` — `show_baseline()`, `ladder()`, chromatogram error against the original
- **measure the output-volume rung first.** It is the one unmeasured claim; if it is real it changes the ranking, and it needs a wall-clock measurement rather than a step count
- decide SMA gradient vs SMB switching
- decide where this sits in the running order — it is arguably the *first* notebook of the chapter, not the last, with the studies as its evidence base

+++

> **Take-away (draft).** Nothing here announced itself. A too-fine grid gives a slightly better answer; a too-tight tolerance gives a slightly better answer; dense output gives more data. None of them produces a symptom, which is exactly why they survive in real input files — and why the fix is a *checklist*, not a diagnosis. Stiffness, had it been the problem, would have told you.
>
> Count your sections, halve your grid, loosen your tolerance, check what you are writing. Then look at the model.

+++

## Summary

...

**Now:** Return to [**99.7g User Stories**](../99.7g_user_stories_lesson.ipynb).
