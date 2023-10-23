import argparse
import os
import shutil
from abc import abstractmethod
from pathlib import Path
import subprocess
import git

from joblib import Parallel, delayed
import pathos


def on_error(func, path, exc_info):
    """
    Error handler for ``shutil.rmtree``.

    If the error is due to an access error (read only file)
    it attempts to add write permission and then retries.

    If the error is for another reason it re-raises the error.

    Taken from StackOverflow

    Usage : ``shutil.rmtree(path, onerror=onerror)``
    """
    import stat
    # Is the error an access error?
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise


class ParallelizationBase:
    def __init__(self, n_cores=1):
        self.n_cores = n_cores

    @abstractmethod
    def run(self, func, args_list):
        """Run a function over a list of input args and return the output.

        Parameters
        ----------
        func : callable
            The function wrapping the shell command.
        args_list : list | iterable
            List of args to the func.

        Returns
        -------
        List of function returns
        """
        return


class SequentialBackend(ParallelizationBase):
    def run(self, func, args_list):
        results = []
        for args in args_list:
            results.append(func(*args))
        return results


class JoblibBackend(ParallelizationBase):
    def run(self, func, args_list):
        Parallel(n_jobs=self.n_cores)(delayed(func)(*args) for args in args_list)


class PathosBackend(ParallelizationBase):
    def run(self, func, args_list):
        with pathos.pools.ProcessPool(ncpus=self.n_cores) as pool:
            # *zip(*args_list) because for multiple args, pathos expects (func, args1_list, args2_list)
            results = pool.map(func, *zip(*args_list))
        return results


def run_func_over_args_list(func, args_list, backend=None, n_cores=1):
    """Run a function over a list of input args and return the output.

    Parameters
    ----------
    func : callable
        The function wrapping the shell command.
    args_list : list | iterable
        List of tuple of args to the func.
    backend : ParallelizationBase, optional
        Backend for parallelization. Default is SequentialBackend
    n_cores : int, optional
        Number of cores to use for parallelization

    Returns
    -------
    List of function returns
    """
    if type(args_list[0]) not in (list, tuple):
        args_list = [(x,) for x in args_list]

    if backend is None and n_cores == 1:
        backend = SequentialBackend()
    else:
        backend = PathosBackend(n_cores=n_cores)

    return backend.run(func, args_list)


def run_command(command):
    """Run a shell command and return its output.

    Parameters
    ----------
    command : str
        The shell command to run.

    Returns
    -------
    None
    """
    return subprocess.run(command, shell=True, check=True)


def create_solution(run=False, commit=False, push=False, n_cores=1, on_fail_restore_dev=False, base_branch="dev",
                    target_branch="solution"
                    ):
    """Create solution files.

    Parameters
    ----------
    run : bool, optional
        Run nbtb on notebooks.
    commit : bool, optional
        Commit changes.
    push : bool, optional
        Push changes to remote.
    n_cores : int, optional
        Number of cpu cores to use for parallelization
    on_fail_restore_dev : bool, optional
        Reset the repo to dev on failure.
    base_branch : str, optional
        Branch on which these actions are based. If git is not currently on this branch: abort script.
    target_branch : str, optional
        Name for the newly created branch.

    Returns
    -------
    None
    """
    repo = git.Repo(search_parent_directories=True)
    current_branch = repo.active_branch.name
    repo_root = Path(repo.working_tree_dir)  # Gets the root directory of the repo

    if current_branch != base_branch:
        print(f"Not on {base_branch} branch. Skipping create_solution script.")
        return

    try:
        # Checkout the 'solution' branch
        repo.git.checkout(target_branch)

        # Reset to base_branch
        run_command(f"git reset --hard {base_branch}")

        # Find all myst files recursively
        myst_files = list(repo_root.glob("**/*.md"))

        # Run jupytext for each myst file
        run_func_over_args_list(func=convert_myst_to_ipynb,
                                args_list=[myst_file.as_posix() for myst_file in myst_files],
                                n_cores=n_cores)

        # Remove all myst files except for the README
        for myst_file in myst_files:
            if "README" in myst_file.as_posix():
                continue
            os.remove(myst_file)

        # Find all ipynb files recursively
        ipynb_files = list(repo_root.glob("**/*.ipynb"))

        # Run nbtb run for each ipynb file
        if run:
            nbtoolbelt_config_path = f"{repo_root}/.nbtoolbelt.json"
            run_func_over_args_list(func=run_notebook,
                                    args_list=[(ipynb_file.as_posix(), nbtoolbelt_config_path)
                                               for ipynb_file in ipynb_files],
                                    n_cores=n_cores)

        if commit:
            # Commit all changes to ipynb files
            repo.git.add(".")  # Less error-prone than working with path lists
            run_command(f'git commit -m "Update {target_branch}"')

        if push:
            # Push files to remote
            run_command(f"git push --force --set-upstream origin {target_branch}")

        # Switch back to base_branch
        repo.git.checkout(base_branch)

    except Exception as e:
        print(f"An error occurred: {e}")
        if on_fail_restore_dev:
            # Restore branch to previous state
            run_command("git restore --staged .")
            run_command("git restore .")

            # Switch back to base_branch
            repo.git.checkout(base_branch)

            try:
                repo.git.stash("pop")
            except git.GitCommandError as e:
                print(e)

        raise


def create_teaching(commit=False, push=False, n_cores=1, on_fail_restore_dev=False, base_branch="dev",
                    target_branch="teaching"):
    """Create teaching files.

    Parameters
    ----------
    commit : bool, optional
        Commit changes.
    push : bool, optional
        Push changes to remote.
    n_cores : int, optional
        Number of cpu cores to use for parallelization
    on_fail_restore_dev : bool, optional
        Reset the repo to dev on failure.

    Returns
    -------
    None
    """
    repo = git.Repo(search_parent_directories=True)
    current_branch = repo.active_branch.name
    repo_root = Path(repo.working_tree_dir)  # Gets the root directory of the repo

    if current_branch != base_branch:
        print(f"Not on {base_branch} branch. Skipping create_teaching script.")
        return

    try:
        # Checkout the 'teaching' branch
        repo.git.checkout(target_branch)

        # Reset to base_branch
        run_command(f"git reset --hard {base_branch}")

        # Find all myst files recursively
        myst_files = list(repo_root.glob("**/*.md"))

        # Run jupytext for each myst file
        run_func_over_args_list(func=convert_myst_to_ipynb,
                                args_list=[myst_file.as_posix() for myst_file in myst_files],
                                n_cores=n_cores)

        # Remove all md files except for the README file
        for myst_file in myst_files:
            if "README" in myst_file.as_posix():
                continue
            os.remove(myst_file)

        # Find all ipynb files recursively
        ipynb_files = list(repo_root.glob("**/*.ipynb"))

        # Run nbtb punch for each ipynb file
        nbtoolbelt_config_path = f"{repo_root}/.nbtoolbelt.json"
        run_func_over_args_list(func=punch_notebook,
                                args_list=[(ipynb_file.as_posix(), nbtoolbelt_config_path) for ipynb_file in
                                           ipynb_files],
                                n_cores=n_cores)

        if commit:
            # Commit all changes to myst files (our source of truth)
            repo.git.add(".")  # Less error-prone than working with path lists
            run_command(f'git commit -m "Update {target_branch}"')

        if push:
            # Push files to remote
            run_command(f"git push --force --set-upstream origin {target_branch}")

        # Clean up
        for ipynb_file in ipynb_files:
            os.remove(ipynb_file)

        # Switch back to base_branch
        repo.git.checkout(base_branch)


    except Exception as e:
        print(f"An error occurred: {e}")
        if on_fail_restore_dev:
            # Restore branch to previous state
            run_command("git restore --staged .")
            run_command("git restore .")

            # Switch back to base_branch
            repo.git.checkout(base_branch)

            try:
                repo.git.stash("pop")
            except git.GitCommandError as e:
                print(e)

        raise


def convert_myst_to_ipynb(myst_file_path):
    """Run jupytext with --to ipynb flag on myst_file_path. Will skip README files.

    Parameters
    ----------
    myst_file_path : str
        path to myst file as posix.

    Returns
    -------
    None
    """
    if "README" in myst_file_path:
        return
    return run_command(f'jupytext --to ipynb "{myst_file_path}"')


def punch_notebook(ipynb_file_path, nbtoolbelt_config_path):
    """Execute nbtb punch on ipynb_file_path.

    Parameters
    ----------
    ipynb_file_path : str
        path to myst file as posix.

    nbtoolbelt_config_path : str
        path to .nbtoolbelt.json config file

    Returns
    -------
    None
    """
    return run_command(f'nbtb punch --config {nbtoolbelt_config_path} "{ipynb_file_path}"')


def run_notebook(ipynb_file_path, nbtoolbelt_config_path):
    """Execute nbtb run on ipynb_file_path.

    Parameters
    ----------
    ipynb_file_path : str
        path to myst file as posix.

    nbtoolbelt_config_path : str
        path to .nbtoolbelt.json config file

    Returns
    -------
    None
    """
    return run_command(f'nbtb run --config {nbtoolbelt_config_path} "{ipynb_file_path}"')


def convert_ipynb_to_myst_md(ipynb_file_path):
    """Convert ipynb file to .md myst file.

    Parameters
    ----------
    ipynb_file_path : str
        path to myst file as posix.

    Returns
    -------
    None
    """
    return run_command(f'jupytext --to md:myst "{ipynb_file_path}"')


def setup_teaching_copy(new_repo_dir="CADET-Workshop-teaching", target_branch="teaching"):
    """Create a copy of this directory and check out the teaching branch in it

    Parameters
    ----------
    new_repo_dir : str
        Name for the new repo folder.
    target_branch : str
        Name of the branch to check out in the new repo folder.

    Returns
    -------

    """

    repo = git.Repo(search_parent_directories=True)

    # Get the root directory of this repo
    repo_root = Path(repo.working_tree_dir)

    # Get the directory in which this repo lives
    parent_dir = os.path.split(repo_root)[0]

    os.chdir(parent_dir)

    # Clean up folder if it exists
    if os.path.exists(new_repo_dir):
        delete_dir_contents(new_repo_dir)

    # Clone from this repo to new repo
    new_repo = git.Repo.clone_from(repo_root, new_repo_dir)

    new_repo.git.checkout(target_branch)


def delete_dir_contents(dir_path):
    """Delete all contents of a directory but leave the directory itself.

    Parameters
    ----------
    dir_path : str | Path
        Path to directory to empty.

    Returns
    -------
    None
    """
    for item in os.listdir(dir_path):
        shutil.rmtree(item, onerror=on_error)


def main(**kwargs):
    """Run post-commit tasks based on command-line arguments.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description='Perform post-commit tasks.')
    parser.add_argument('--run', action='store_true', help='Run nbtb on notebooks.')
    parser.add_argument('--commit', action='store_true', help='Commit changes.')
    parser.add_argument('--push', action='store_true', help='Push changes to remote.')
    parser.add_argument('--on_fail_restore_dev', action='store_true',
                        help='Reset to dev branch on error.')
    parser.add_argument('--n_cores', help='Number of cores to use.')

    args = parser.parse_args()
    if args.n_cores is None:
        args.n_cores = 1

    # This isn't great, but for now (and with argparse) the best I could think of
    for kwarg_key, kwarg_value in kwargs.items():
        if kwarg_value is None:
            continue
        args.__setattr__(kwarg_key, kwarg_value)

    args.commit = True

    create_solution(args.run, args.commit, args.push, args.n_cores, args.on_fail_restore_dev, base_branch="test-ci",
                    target_branch="tmp_solution")
    create_teaching(args.commit, args.push, args.n_cores, args.on_fail_restore_dev, base_branch="test-ci",
                    target_branch="tmp_teaching")


if __name__ == "__main__":
    main()
