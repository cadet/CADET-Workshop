[![CC BY NC SA 4.0][cc-by-nc-sa-shield]][cc-by-nc-sa]

This work is licensed under a
[Creative Commons Attribution 4.0 International License][cc-by-nc-sa].

[cc-by-nc-sa]: https://creativecommons.org/licenses/by-nc-sa/4.0/
[cc-by-nc-sa-shield]: https://img.shields.io/badge/License-CC%20BY%20NC%20SA%204.0-lightgrey.svg

# CADET Tutorials

This is a repository for introductory examples and tutorials of the [CADET-Process](https://cadet-process.readthedocs.io/) software for modelling and simulating chromatographic processes.

For more information, see also:
- **Website (including documentation):** https://cadet-process.readthedocs.io/
- **Forum:** https://forum.cadet-web.de
- **Source:** https://github.com/fau-advanced-separations/CADET-Process

## Download the tutorials
To run the tutorials locally, we recommend installing [Anaconda](https://www.anaconda.com/).
Anaconda is a high-performance scientific distribution of Python that includes many common packages needed for scientific and engineering work.
Download the installer from their website and run it for the local (!) user.
Optionally, use [mamba](https://github.com/mamba-org/mamba) which uses a faster dependency solver than `conda`.
To use `mamba`, replace `conda` commands with `mamba`.

The easiest way to download the tutorials is to clone this repository.
For this purpuse, make sure, [git](https://git-scm.com/downloads) is installed.

From a `git bash` navigate to to where you want to store the tutorials and run
```
git clone https://github.com/cadet/CADET-Workshop
```

Then, from an `Anaconda Prompt`, `cd` into the directory and install all the requirements by running

```
mamba env create -f ./environment.yml
```

This will create a new conda environment called `cadet`.
Activate it using the following command

```
conda activate cadet
```

## Workshop structure

Chapters contain numbered topics. A lesson and its exercises share the same topic number and name. The `a` / `b` markers put the lesson before the exercises; the final word states their role.

```text
02 CADET Fundamentals/
  02.3 Inlet Profiles/
    02.3a_inlet_profiles_lesson.md
    02.3b_inlet_profiles_exercises.md
```

The source branch keeps `.md` files as the ground truth. Generate and test `.ipynb` files in temporary directories, then delete them. The generated `teaching` and `solution` branches publish `.ipynb` files. Topics without a separate exercise notebook contain only a lesson. Chapter `00` is optional Python preparation; `99` contains additional examples. The [GUI Workbench](<99 Misc/99.2 GUI Workbench/99.2a_gui_workbench_lesson.md>) is available there for optional process setup and code export. [99.3 Reactions](<99 Misc/99.3 Reactions/>) contains the optional batch-CSTR excursion and its exercises.

The main sequence after Process Simulation is **04 Characterization and Optimization**:

1. **04.1 Optimization Essentials** (chapter introduction, no exercises): introduce the concepts through minimizing a tracer-fitting error and maximizing productivity under a purity constraint.
2. **04.2 Parameter Fitting:** fit tracer data with the estimation GUI, then connect references, comparisons and simulations for a full-curve Python fit, and carry an accepted system result into the column fit.
3. **04.3 Fractionation:** define product pools and assess separation performance.
4. **04.4 Process Optimization:** improve operating conditions using the characterized model.

Detailed optimization examples are optional reference material in [99.4 Optimization Basics](<99 Misc/99.4 Optimization Basics/>) and [99.5 Advanced Optimization](<99 Misc/99.5 Advanced Optimization/>).

Each lesson and exercise begins with a contents overview and its introduced or practised concepts and unit operations. **Python used** lists the concrete coding skills involved. **Tools** identifies CADET-Process (Python), CADET-GUI widgets or other interactive controls. Widget roles follow the CADET-GUI name directly.

Blue **notes** give context, green **tips** guide actions, and amber **warnings** flag common pitfalls.

Shared GUI setup for the early lessons lives in [lesson_setup.py](<02 CADET Fundamentals/resources/first_steps/lesson_setup.py>). Solution-tagged calls populate completed checkpoints; teaching removes those calls. The Python examples remain in the notebooks.

## Getting started
Fire up a `jupyter-lab` and get started with the Tutorials.

The optional [Python preparation](<00 Python Preparation/>) chapter covers the basics needed for the workshop.


## Fixing potential problems.
- If you get the following error `The code execution cannot proceed because VCRUNTIME140_1.dll was not found. Reinstalling the program may fix this problem.`, please visit https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist and install the latest Microsoft Visual C++ Redistributable.
- Some of the notebooks include interactive graphs. To enable them, please open an Anaconda prompt and run:
    - For JupyterLab 2.0+
      ```
      jupyter labextension install @jupyter-widgets/jupyterlab-manager
      jupyter lab clean
      jupyter lab build
      ```
    - For JupyterLab 3.0+: install `ipympl`
