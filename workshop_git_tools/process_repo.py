import argparse
import os
from pathlib import Path
import subprocess
import git

from joblib import Parallel, delayed


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


def create_solution(run=False, commit=False, push=False, n_cores=1):
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

    Returns
    -------
    None
    """
    repo = git.Repo(search_parent_directories=True)
    current_branch = repo.active_branch.name
    repo_root = Path(repo.working_tree_dir)  # Gets the root directory of the repo

    if current_branch != "dev":
        print("Not on dev branch. Skipping create_solution script.")
        return

    try:
        # Stash everything
        run_command("git stash")

        # Checkout the 'solution' branch
        repo.git.checkout("solution")

        # Reset to `dev`
        run_command("git reset --hard dev")

        # Find all myst files recursively
        myst_files = list(repo_root.glob("**/*.md"))

        # Run jupytext for each myst file

        for myst_file in myst_files:
            if "README" in myst_file.as_posix():
                continue
            convert_myst_to_ipynb(myst_file.as_posix())
            os.remove(myst_file)

        # Find all ipynb files recursively
        ipynb_files = list(repo_root.glob("**/*.ipynb"))

        # Run nbtb run for each ipynb file
        if run:
            nbtoolbelt_config_path = f"{repo_root}/.nbtoolbelt.json"
            for ipynb_file in ipynb_files:
                run_notebook(ipynb_file.as_posix(), nbtoolbelt_config_path)
            # Parallel(n_jobs=n_cores)(
            #     delayed(run_notebook)(ipynb_file.as_posix(), nbtoolbelt_config_path) for ipynb_file in ipynb_files
            # )

        if commit:
            # Commit all changes to ipynb files
            repo.git.add(".")
            run_command('git commit -m "Update solution"')

        if push:
            # Push files to remote
            run_command("git push --force-with-lease --set-upstream origin solution")

        # Switch back to dev
        repo.git.checkout("dev")

        try:
            repo.git.stash("pop")
        except git.GitCommandError:
            pass

    except Exception as e:
        print(f"An error occurred: {e}")
        # run_command("git restore --staged .")
        # run_command("git restore .")

        raise


def create_teaching(commit=False, push=False, n_cores=1):
    """Create teaching files.

    Parameters
    ----------
    commit : bool, optional
        Commit changes.
    push : bool, optional
        Push changes to remote.
    n_cores : int, optional
        Number of cpu cores to use for parallelization

    Returns
    -------
    None
    """
    repo = git.Repo(search_parent_directories=True)
    current_branch = repo.active_branch.name
    repo_root = Path(repo.working_tree_dir)  # Gets the root directory of the repo

    if current_branch != "dev":
        print("Not on dev branch. Skipping create_solution script.")
        return

    try:
        # Stash everything
        run_command("git stash")

        # Checkout the 'teaching' branch
        repo.git.checkout("teaching")

        # Reset to `dev`
        run_command("git reset --hard dev")

        # Find all myst files recursively
        myst_files = list(repo_root.glob("**/*.md"))

        # Run jupytext for each myst file
        # Parallel(n_jobs=n_cores)(delayed(convert_myst_to_ipynb)(myst_file.as_posix()) for myst_file in myst_files)
        for myst_file in myst_files:
            convert_myst_to_ipynb(myst_file.as_posix())


        # Remove all md files except for the README file
        for myst_file in myst_files:
            if "README" in myst_file.as_posix():
                continue
            os.remove(myst_file)

        # Find all ipynb files recursively
        ipynb_files = list(repo_root.glob("**/*.ipynb"))

        # Run nbtb punch for each ipynb file
        nbtoolbelt_config_path = f"{repo_root}/.nbtoolbelt.json"
        for ipynb_file in ipynb_files:
            punch_notebook(ipynb_file.as_posix(), nbtoolbelt_config_path)
        # Parallel(n_jobs=n_cores)(
        #     delayed(punch_notebook)(ipynb_file.as_posix(), nbtoolbelt_config_path) for ipynb_file in ipynb_files
        # )

        if commit:
            # Commit all changes to myst files (our source of truth)
            repo.git.add(".")
            run_command('git commit -m "Update teaching"')

        if push:
            # Push files to remote
            run_command("git push --force-with-lease --set-upstream origin teaching")

        # Clean up
        for ipynb_file in ipynb_files:
            os.remove(ipynb_file)

        # Switch back to dev
        repo.git.checkout("dev")

        try:
            repo.git.stash("pop")
        except git.GitCommandError as e:
            print(e)

    except Exception as e:
        print(f"An error occurred: {e}")
        # run_command("git restore --staged .")
        # run_command("git restore .")

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
    run_command(f'jupytext --to ipynb "{myst_file_path}"')
    return


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
    run_command(f'nbtb punch --config {nbtoolbelt_config_path} "{ipynb_file_path}"')


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
    run_command(f'nbtb run --config {nbtoolbelt_config_path} "{ipynb_file_path}"')


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
    run_command(f'jupytext --to md:myst "{ipynb_file_path}"')


def main():
    """Run post-commit tasks based on command-line arguments.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(description='Perform post-commit tasks.')
    parser.add_argument('--run', action='store_true', help='Run nbtb on notebooks.')
    parser.add_argument('--commit', action='store_true', help='Commit changes.')
    parser.add_argument('--push', action='store_true', help='Push changes to remote.')
    parser.add_argument('--n_cores', help='Number of cores to use.')

    args = parser.parse_args()

    args.run = False

    create_solution(args.run, args.commit, args.push, args.n_cores)
    create_teaching(args.commit, args.push, args.n_cores)


if __name__ == "__main__":
    main()
