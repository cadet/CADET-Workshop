import json
import os
import shutil
import unittest
from pathlib import Path

from git import Repo

from workshop_git_tools import process_repo
from workshop_git_tools.process_repo import on_error

REPO_ROOT = Path(__file__).resolve().parents[1]
TMP_REPO = Path(__file__).resolve().parent / "tmp_test_repo"


def solution_sources(notebook):
    nb = json.loads(notebook.read_text())
    return [
        "".join(cell["source"])
        for cell in nb["cells"]
        if "solution" in cell.get("metadata", {}).get("tags", [])
    ]


class Test_Workshop_Tools(unittest.TestCase):

    def setUp(self):
        if TMP_REPO.exists():
            shutil.rmtree(TMP_REPO, onerror=on_error)
        source = Repo(REPO_ROOT).active_branch.name
        self.repo = Repo.clone_from(REPO_ROOT.as_posix(), TMP_REPO.as_posix(), branch=source)
        self.source = source
        self.cwd = os.getcwd()
        os.chdir(TMP_REPO)

    def tearDown(self):
        os.chdir(self.cwd)
        shutil.rmtree(TMP_REPO, onerror=on_error)

    def test_preview_build(self):
        lessons = process_repo.lesson_files(TMP_REPO)
        root_docs = sorted(p.name for p in TMP_REPO.glob("*.md"))
        self.assertTrue(lessons)

        process_repo.main(
            [],
            commit=True,
            source=self.source,
            teaching_branch="teaching-preview",
            solution_branch="solution-preview",
        )
        self.assertEqual(self.repo.active_branch.name, self.source)

        for branch, punched in (("teaching-preview", True), ("solution-preview", False)):
            self.repo.git.checkout(branch)
            for lesson in lessons:
                notebook = lesson.with_suffix(".ipynb")
                self.assertFalse(lesson.exists(), lesson)
                self.assertTrue(notebook.exists(), notebook)
                sources = solution_sources(notebook)
                if not sources:
                    continue
                if punched:
                    self.assertTrue(all(s == "" for s in sources), notebook)
                else:
                    self.assertTrue(any(s != "" for s in sources), notebook)
            self.assertEqual(sorted(p.name for p in TMP_REPO.glob("*.md")), root_docs)
            tracked = self.repo.git.ls_files().splitlines()
            self.assertFalse([f for f in tracked if ".ipynb_checkpoints" in f])


if __name__ == '__main__':
    unittest.main()
