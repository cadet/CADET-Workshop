from git import Repo
import shutil
import os
import unittest

from workshop_git_tools import process_repo, setup_repo


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


class Test_Workshop_Tools(unittest.TestCase):

    def __init__(self, methodName='runTest'):
        super().__init__(methodName)

    def test_workshop_git_tools(self, ):
        if os.path.exists("./tmp_test_repo"):
            shutil.rmtree("./tmp_test_repo", onerror=on_error)
        Repo.clone_from("../", "./tmp_test_repo")

        os.chdir("./tmp_test_repo")

        setup_repo.setup_repo()
        process_repo.main(run=False, commit=True)
        # ToDo: add actual test assertions.
        os.chdir("..")


if __name__ == '__main__':
    unittest.main()
