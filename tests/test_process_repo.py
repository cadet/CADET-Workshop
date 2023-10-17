from git import Repo
import shutil
import os
import unittest

from workshop_git_tools import process_repo, setup_repo
from workshop_git_tools.process_repo import on_error


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
