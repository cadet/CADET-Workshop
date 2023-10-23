import git


def create_or_replace_branch(repo, branch_name):
    """Create or replace a git branch.

    Parameters
    ----------
    repo : git.Repo
        GitPython Repo object.
    branch_name : str
        Name of the branch to create or replace.

    Returns
    -------
    git.Head
        The created git Head object representing the branch.
    """
    # Check if branch exists
    if branch_name in repo.heads:
        # Delete existing branch
        repo.delete_head(branch_name, force=True)

    # Create new branch
    new_branch = repo.create_head(branch_name)

    return new_branch


def setup_repo():
    repo = git.Repo()

    repo.heads.dev.checkout()

    create_or_replace_branch(repo, 'tmp_solution')
    create_or_replace_branch(repo, 'tmp_teaching')


if __name__ == "__main__":
    setup_repo()
