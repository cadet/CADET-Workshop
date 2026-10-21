import json
import sys
import os
import shutil
import unittest
from pathlib import Path

from git import Repo
import jupytext

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".workshop_git_tools"))

import process_repo
from process_repo import on_error

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
        # Preserve starters and widgets, not only the solution-tagged answers.
        expected_cells = {
            lesson: [
                (cell.cell_type, cell.source, cell.metadata.get("tags", []),
                 process_repo.punch_placeholder(cell))
                for cell in jupytext.read(lesson).cells
            ]
            for lesson in lessons
        }
        support_scripts = {
            (lesson.parent / line.split(maxsplit=1)[1]).resolve()
            for lesson in lessons
            for cell in jupytext.read(lesson).cells if cell.cell_type == "code"
            for line in cell.source.splitlines() if line.startswith("%run ")
        }

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
            for script in support_scripts:
                self.assertTrue(script.is_file(), script)
            for lesson in lessons:
                notebook = lesson.with_suffix(".ipynb")
                self.assertFalse(lesson.exists(), lesson)
                self.assertTrue(notebook.exists(), notebook)
                published = json.loads(notebook.read_text())
                self.assertEqual(published["metadata"]["jupytext"]["formats"], "ipynb")
                cells = published["cells"]
                expected = expected_cells[lesson]
                self.assertEqual(len(cells), len(expected), notebook)
                for cell, (cell_type, source, tags, placeholder) in zip(cells, expected):
                    self.assertEqual(cell["cell_type"], cell_type, notebook)
                    self.assertEqual(cell.get("metadata", {}).get("tags", []), tags, notebook)
                    expected_source = placeholder if punched and "solution" in tags else source
                    self.assertEqual("".join(cell["source"]), expected_source, notebook)
                sources = solution_sources(notebook)
                if not sources:
                    continue
                if punched:
                    # At most a one-line hint remains of each answer.
                    self.assertTrue(all(len(s.splitlines()) <= 1 for s in sources), notebook)
                else:
                    self.assertTrue(any(s.strip() for s in sources), notebook)
            self.assertEqual(sorted(p.name for p in TMP_REPO.glob("*.md")), root_docs)
            tracked = self.repo.git.ls_files().splitlines()
            self.assertFalse([f for f in tracked if ".ipynb_checkpoints" in f])


if __name__ == '__main__':
    unittest.main()
