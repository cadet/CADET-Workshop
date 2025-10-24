import argparse
import os
import shutil
from abc import abstractmethod
from pathlib import Path
import subprocess
import git

from joblib import Parallel, delayed
import pathos


# --------------- general utilities ---------------

def run_command(command, cwd=None):
    """Run a shell command and raise on failure.

    Parameters
    command : str
        The shell command to execute.
    cwd : str or Path, optional
        Working directory in which to execute the command.

    Returns
    subprocess.CompletedProcess
        The completed process object returned by ``subprocess.run`` with ``check=True``.

    Raises
    subprocess.CalledProcessError
        If the command returns a nonzero exit status.
    """
    return subprocess.run(command, shell=True, check=True, cwd=cwd)


def on_error(func, path, exc_info):
    """Error handler for ``shutil.rmtree``.

    This handler grants write permission and retries deletion when a path is not writable.

    Parameters
    func : callable
        The function from ``shutil`` that raised the error. Usually ``os.remove`` or ``os.rmdir``.
    path : str or Path
        The path that could not be removed.
    exc_info : tuple
        Exception info tuple as provided by ``shutil.rmtree``.

    Notes
    This function is meant to be passed to the ``onerror`` argument of ``shutil.rmtree``.
    """
    import stat
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise


def delete_dir_contents(dir_path):
    """Delete all contents of a directory but keep the directory itself.

    Parameters
    dir_path : str or Path
        Directory whose contents should be deleted.

    Raises
    PermissionError
        If a file cannot be deleted even after permission change.
    """
    for item in os.listdir(dir_path):
        p = Path(dir_path) / item
        if p.is_dir():
            shutil.rmtree(p, onerror=on_error)
        else:
            try:
                p.unlink()
            except PermissionError:
                os.chmod(p, 0o666)
                p.unlink()


# --------------- stash helpers ---------------

def stash_if_dirty(repo: git.Repo) -> bool:
    """Stash tracked and untracked changes if there is anything to save.

    Parameters
    repo : git.Repo
        The repository to operate on.

    Returns
    bool
        True if a stash was created. False if the working tree was clean.

    Raises
    git.GitCommandError
        If the repository is in an unmerged state and stashing fails.
    """
    if repo.is_dirty(untracked_files=True):
        try:
            repo.git.stash("push", "-u", "-m", "process_repo.py")
            return True
        except git.GitCommandError:
            raise
    return False


def pop_stash_safely(repo: git.Repo, stashed: bool):
    """Apply the top stash and drop it only if the apply is clean.

    Parameters
    repo : git.Repo
        The repository to operate on.
    stashed : bool
        Whether a stash was created earlier and should be applied back.

    Notes
    If conflicts occur during apply, the stash is kept and a message is printed.
    If there are unmerged entries after apply, the stash is preserved for manual resolution.
    """
    if not stashed:
        return
    try:
        repo.git.stash("apply")
    except git.GitCommandError as e:
        print(e)

    if repo.index.unmerged_blobs():
        print("Stash apply created conflicts. Stash kept. Resolve manually if needed.")
        return

    try:
        repo.git.stash("drop")
    except git.GitCommandError as e:
        print(e)


# --------------- parallel backends ---------------

class ParallelizationBase:
    """Base class for parallelization backends.

    Parameters
    n_cores : int, default 1
        Number of worker processes to use when running tasks.
    """

    def __init__(self, n_cores=1):
        self.n_cores = int(n_cores) if n_cores is not None else 1

    @abstractmethod
    def run(self, func, args_list):
        """Run ``func`` over ``args_list`` and return results.

        Parameters
        func : callable
            Function to execute. It will be called with the provided positional arguments.
        args_list : list of tuple
            Each tuple provides positional arguments for a single call of ``func``.

        Returns
        list
            List of results returned by ``func`` for each set of arguments.
        """
        return


class SequentialBackend(ParallelizationBase):
    """Simple sequential execution backend."""

    def run(self, func, args_list):
        """Run tasks sequentially.

        Parameters
        func : callable
            Function to execute.
        args_list : list of tuple
            Argument sets for each call.

        Returns
        list
            Results in the same order as ``args_list``.
        """
        results = []
        for args in args_list:
            results.append(func(*args))
        return results


class JoblibBackend(ParallelizationBase):
    """Joblib based parallel execution backend."""

    def run(self, func, args_list):
        """Run tasks in parallel using joblib.

        Parameters
        func : callable
            Function to execute.
        args_list : list of tuple
            Argument sets for each call.

        Returns
        list
            Results in the same order as ``args_list``.
        """
        return Parallel(n_jobs=self.n_cores)(delayed(func)(*args) for args in args_list)


class PathosBackend(ParallelizationBase):
    """Pathos based parallel execution backend."""

    def run(self, func, args_list):
        """Run tasks in parallel using pathos process pool.

        Parameters
        func : callable
            Function to execute. Must be picklable for process based execution.
        args_list : list of tuple
            Argument sets for each call.

        Returns
        list
            Results in the same order as ``args_list``.
        """
        if not args_list:
            return []
        with pathos.pools.ProcessPool(ncpus=self.n_cores) as pool:
            results = pool.map(func, *zip(*args_list))
        return results


def run_func_over_args_list(func, args_list, backend=None, n_cores=1):
    """Run a function over a list of inputs and collect outputs.

    Parameters
    func : callable
        Function to execute for each item.
    args_list : list
        List of inputs. Each element can be a tuple of positional arguments. If an element
        is not a tuple or list, it is wrapped as a single argument.
    backend : ParallelizationBase, optional
        Backend to use. If not provided, a sequential backend is used when ``n_cores`` is 1,
        otherwise a pathos backend is used.
    n_cores : int, default 1
        Number of worker processes to use for parallel backends.

    Returns
    list
        Results returned by ``func`` for each element in ``args_list``.
    """
    if not args_list:
        return []

    if type(args_list[0]) not in (list, tuple):
        args_list = [(x,) for x in args_list]

    if backend is None:
        backend = SequentialBackend() if (n_cores is None or int(n_cores) == 1) else PathosBackend(n_cores=int(n_cores))

    return backend.run(func, args_list)


# --------------- domain specific helpers ---------------

def convert_myst_to_ipynb(myst_file_path):
    """Convert a MyST markdown file to a Jupyter notebook using jupytext.

    Parameters
    myst_file_path : str or Path
        Path to a MyST markdown file. Files containing the string ``README`` are skipped.

    Returns
    subprocess.CompletedProcess or None
        The completed process if conversion is executed. None if the file is skipped.
    """
    if "README" in myst_file_path:
        return
    return run_command(f'jupytext --to ipynb "{myst_file_path}"')


def punch_notebook(ipynb_file_path, nbtoolbelt_config_path):
    """Execute ``nbtb punch`` on a notebook.

    Parameters
    ipynb_file_path : str or Path
        Path to the notebook file.
    nbtoolbelt_config_path : str or Path
        Path to the nbtoolbelt configuration file.

    Returns
    subprocess.CompletedProcess
        The completed process object from the command.
    """
    nb_dir = Path(ipynb_file_path).parent
    return run_command(
        f'nbtb punch --config "{nbtoolbelt_config_path}" "{ipynb_file_path}"',
        cwd=nb_dir
    )


def run_notebook(ipynb_file_path, nbtoolbelt_config_path):
    """Execute ``nbtb run`` on a notebook.

    Parameters
    ipynb_file_path : str or Path
        Path to the notebook file.
    nbtoolbelt_config_path : str or Path
        Path to the nbtoolbelt configuration file.

    Returns
    subprocess.CompletedProcess
        The completed process object from the command.
    """
    nb_dir = Path(ipynb_file_path).parent
    return run_command(
        f'nbtb run --config "{nbtoolbelt_config_path}" "{ipynb_file_path}"',
        cwd=nb_dir
    )


def convert_ipynb_to_myst_md(ipynb_file_path):
    """Convert a Jupyter notebook to MyST markdown using jupytext.

    Parameters
    ipynb_file_path : str or Path
        Path to the notebook file.

    Returns
    subprocess.CompletedProcess
        The completed process object from the command.
    """
    return run_command(f'jupytext --to md:myst "{ipynb_file_path}"')


def setup_teaching_copy(new_repo_dir="CADET-Workshop-teaching"):
    """Create a sibling repository clone and check out the teaching branch.

    A clean copy is created next to the current repository. If the target directory exists,
    its contents are removed and the directory is reused.

    Parameters
    new_repo_dir : str, default "CADET-Workshop-teaching"
        Directory name for the new clone relative to the parent of the current repository root.

    Raises
    git.GitCommandError
        If git operations fail.
    """
    repo = git.Repo(search_parent_directories=True)
    repo_root = Path(repo.working_tree_dir)
    parent_dir = os.path.split(repo_root)[0]
    os.chdir(parent_dir)

    if os.path.exists(new_repo_dir):
        delete_dir_contents(new_repo_dir)

    new_repo = git.Repo.clone_from(repo_root.as_posix(), new_repo_dir)
    new_repo.git.checkout("teaching")


# --------------- core actions ---------------

def create_solution(run=False, commit=False, push=False, n_cores=1, on_fail_restore_dev=False):
    """Create solution branch artifacts from markdown on dev.

    This function checks out the solution branch, aligns it to dev, converts markdown files to
    notebooks, optionally executes notebooks, removes markdown files except README, and
    optionally commits and pushes the results.

    Parameters
    run : bool, default False
        If True, execute notebooks with ``nbtb run``.
    commit : bool, default False
        If True, create a commit with the changes.
    push : bool, default False
        If True, push the solution branch to the remote with force with lease.
    n_cores : int, default 1
        Number of worker processes used when executing notebooks.
    on_fail_restore_dev : bool, default False
        If True and an error occurs, attempt to restore the index and working tree on dev.

    Raises
    Exception
        Any exception encountered during processing is re-raised after cleanup.

    Notes
    If the current branch is not dev, the function prints a message and returns without action.
    Local uncommitted changes are temporarily stashed and restored at the end.
    """
    repo = git.Repo(search_parent_directories=True)
    current_branch = repo.active_branch.name
    repo_root = Path(repo.working_tree_dir)

    if current_branch != "dev":
        print("Not on dev branch. Skipping create_solution script.")
        return

    stashed = False
    try:
        stashed = stash_if_dirty(repo)

        repo.git.checkout("solution")
        run_command("git reset --hard dev")

        myst_files = list(repo_root.glob("**/*.md"))

        run_func_over_args_list(
            func=convert_myst_to_ipynb,
            args_list=[m.as_posix() for m in myst_files],
            n_cores=n_cores,
        )

        for m in myst_files:
            if "README" in m.as_posix():
                continue
            try:
                run_command(f'git rm --quiet -- "{m.as_posix()}"')
            except subprocess.CalledProcessError:
                pass

        if run:
            ipynb_files = list(repo_root.glob("**/*.ipynb"))
            nbtoolbelt_config_path = f"{repo_root}/.nbtoolbelt.json"
            run_func_over_args_list(
                func=run_notebook,
                args_list=[(p.as_posix(), nbtoolbelt_config_path) for p in ipynb_files],
                n_cores=n_cores,
            )

        if commit:
            repo.git.add("-A")
            run_command('git commit -m "Update solution"')

        if push:
            run_command("git push --force-with-lease --set-upstream origin solution")

    except Exception as e:
        print(f"An error occurred: {e}")
        if on_fail_restore_dev:
            try:
                run_command("git restore --staged .", cwd=repo_root)
                run_command("git restore .", cwd=repo_root)
            except Exception as ee:
                print(f"Restore warning: {ee}")
        raise
    finally:
        try:
            repo.git.checkout("dev")
        except git.GitCommandError as e:
            print(e)
        pop_stash_safely(repo, stashed)


def create_teaching(commit=False, push=False, n_cores=1, on_fail_restore_dev=False):
    """Create teaching branch artifacts from markdown on dev and keep punched notebooks.

    This function checks out the teaching branch, aligns it to dev, converts markdown files to
    notebooks, runs ``nbtb punch`` on the notebooks, removes markdown files except README,
    and optionally commits and pushes the results. Notebooks are kept on the teaching branch.

    Parameters
    commit : bool, default False
        If True, create a commit with the changes. Notebook files are force added to bypass gitignore.
    push : bool, default False
        If True, push the teaching branch to the remote with force with lease.
    n_cores : int, default 1
        Number of worker processes used when punching notebooks.
    on_fail_restore_dev : bool, default False
        If True and an error occurs, attempt to restore the index and working tree on dev.

    Raises
    Exception
        Any exception encountered during processing is re-raised after cleanup.

    Notes
    If the current branch is not dev, the function prints a message and returns without action.
    Local uncommitted changes are temporarily stashed and restored at the end.
    """
    repo = git.Repo(search_parent_directories=True)
    current_branch = repo.active_branch.name
    repo_root = Path(repo.working_tree_dir)

    if current_branch != "dev":
        print("Not on dev branch. Skipping create_teaching script.")
        return

    stashed = False
    try:
        stashed = stash_if_dirty(repo)

        repo.git.checkout("teaching")
        run_command("git reset --hard dev")

        myst_files = list(repo_root.glob("**/*.md"))

        run_func_over_args_list(
            func=convert_myst_to_ipynb,
            args_list=[m.as_posix() for m in myst_files],
            n_cores=n_cores,
        )

        for m in myst_files:
            if "README" in m.as_posix():
                continue
            try:
                run_command(f'git rm --quiet -- "{m.as_posix()}"')
            except subprocess.CalledProcessError:
                pass

        ipynb_files = list(repo_root.glob("**/*.ipynb"))
        nbtoolbelt_config_path = f"{repo_root}/.nbtoolbelt.json"
        run_func_over_args_list(
            func=punch_notebook,
            args_list=[(p.as_posix(), nbtoolbelt_config_path) for p in ipynb_files],
            n_cores=n_cores,
        )

        if commit:
            run_command("git add -A")
            for p in ipynb_files:
                try:
                    repo.git.add("-f", p.as_posix())
                except git.GitCommandError:
                    pass
            try:
                run_command('git commit -m "Update teaching"')
            except subprocess.CalledProcessError:
                pass

        if push:
            run_command("git push --force-with-lease --set-upstream origin teaching")

    except Exception as e:
        print(f"An error occurred: {e}")
        if on_fail_restore_dev:
            try:
                run_command("git restore --staged .", cwd=repo_root)
                run_command("git restore .", cwd=repo_root)
            except Exception as ee:
                print(f"Restore warning: {ee}")
        raise
    finally:
        try:
            repo.git.checkout("dev")
        except git.GitCommandError as e:
            print(e)
        pop_stash_safely(repo, stashed)


# --------------- cli ---------------

def main(**kwargs):
    """Command line entry point for post commit tasks.

    The function parses command line arguments, overlays provided keyword arguments, and runs
    the solution and teaching creation workflows.

    Parameters
    **kwargs
        Optional keyword arguments to override parsed arguments. Only non None values are applied.

    Notes
    The default behavior is to not execute notebooks. The number of cores is parsed from the
    ``--n_cores`` argument and falls back to 1 on invalid input.
    """
    parser = argparse.ArgumentParser(description="Perform post commit tasks.")
    parser.add_argument("--run", action="store_true", help="Run nbtb on notebooks for solution.")
    parser.add_argument("--commit", action="store_true", help="Commit changes.")
    parser.add_argument("--push", action="store_true", help="Push changes to remote.")
    parser.add_argument("--on_fail_restore_dev", action="store_true", help="Reset to dev branch on error.")
    parser.add_argument("--n_cores", help="Number of cores to use.")

    args = parser.parse_args()

    for kwarg_key, kwarg_value in kwargs.items():
        if kwarg_value is None:
            continue
        setattr(args, kwarg_key, kwarg_value)

    args.run = bool(getattr(args, "run", False))
    try:
        n_cores = int(args.n_cores) if args.n_cores is not None else 1
    except ValueError:
        n_cores = 1

    create_solution(args.run, args.commit, args.push, n_cores, args.on_fail_restore_dev)
    create_teaching(args.commit, args.push, n_cores, args.on_fail_restore_dev)
    
if __name__ == "__main__":
    main()