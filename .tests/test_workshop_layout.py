import sys
import re
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path
from urllib.parse import unquote

import jupytext

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / ".workshop_git_tools"))
from IPython.core.inputtransformer2 import TransformerManager

from process_repo import lesson_files


REPO_ROOT = Path(__file__).resolve().parents[1]


class TestWorkshopLayout(unittest.TestCase):
    def test_code_cells_have_valid_syntax(self):
        transformer = TransformerManager()
        for path in lesson_files(REPO_ROOT):
            notebook = jupytext.read(path)
            for index, cell in enumerate(notebook.cells):
                if cell.cell_type == "code" and cell.source.strip():
                    with self.subTest(path=path.relative_to(REPO_ROOT), cell=index):
                        compile(transformer.transform_cell(cell.source), str(path), "exec")

    def test_source_chapters_contain_no_generated_notebooks(self):
        for chapter in REPO_ROOT.iterdir():
            if chapter.is_dir() and re.match(r"^\d{2} ", chapter.name):
                self.assertEqual(list(chapter.rglob("*.ipynb")), [], chapter)
        for path in lesson_files(REPO_ROOT):
            self.assertEqual(jupytext.read(path).metadata["jupytext"]["formats"], "md:myst", path)

    def test_exercise_cells_are_visible_and_widgets_explained(self):
        hiding_tags = {
            "hide-input", "hide-output", "hide-cell",
            "remove-input", "remove-output", "remove-cell",
        }
        for path in lesson_files(REPO_ROOT):
            if not path.stem.endswith("_exercises"):
                continue
            notebook = jupytext.read(path)
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                self.assertEqual(notebook.cells[0].cell_type, "markdown")
                self.assertIn("Shift+Enter", notebook.cells[0].source)
                for index, cell in enumerate(notebook.cells):
                    # Solution-only checkpoints (empty on teaching) may hide their source.
                    checkpoint = "solution" in cell.metadata.get("tags", [])
                    if not checkpoint:
                        self.assertFalse(cell.metadata.get("jupyter", {}).get("source_hidden", False))
                    self.assertFalse(cell.metadata.get("jupyter", {}).get("outputs_hidden", False))
                    for flag in ["hide_input", "hide_output", "inputHidden", "outputHidden", "hidden", "collapsed"]:
                        self.assertFalse(cell.metadata.get(flag, False))
                    self.assertNotIn("slideshow", cell.metadata)
                    self.assertNotIn("@deathbeds/jupyterlab-fonts", cell.metadata)
                    self.assertFalse(hiding_tags.intersection(cell.metadata.get("tags", [])))
                    if cell.cell_type == "code" and ".display()" in cell.source:
                        explanation = notebook.cells[index - 1]
                        self.assertEqual(explanation.cell_type, "markdown")
                        self.assertIn("Shift+Enter", explanation.source)

    def test_chapter_topic_and_notebook_roles(self):
        topics = defaultdict(set)
        for path in lesson_files(REPO_ROOT):
            relative = path.relative_to(REPO_ROOT)
            with self.subTest(path=relative):
                self.assertEqual(len(relative.parts), 3)
                chapter, topic, filename = relative.parts
                self.assertRegex(chapter, r"^\d{2} .+")
                self.assertRegex(topic, r"^\d{2}\.\d+ .+")
                topic_id = topic.split(" ", 1)[0]
                self.assertEqual(topic_id.split(".")[0], chapter[:2])
                self.assertRegex(filename, rf"^{re.escape(topic_id)}[ab]_[a-z0-9_]+_(lesson|exercises)\.md$")
                is_exercises = filename.endswith("_exercises.md")
                marker = "b" if is_exercises else "a"
                self.assertTrue(filename.startswith(f"{topic_id}{marker}_"))
                lesson_name = filename
                if is_exercises:
                    lesson_name = f"{topic_id}a" + filename[len(topic_id) + 1:]
                    lesson_name = lesson_name.removesuffix("_exercises.md") + "_lesson.md"
                    self.assertLess(lesson_name, filename)
                self.assertTrue((path.parent / lesson_name).is_file())
                topics[chapter].add(int(topic_id.split(".")[1]))
        self.assertTrue(topics)
        for chapter, numbers in topics.items():
            with self.subTest(chapter=chapter):
                self.assertEqual(sorted(numbers), list(range(1, len(numbers) + 1)))

    def test_topic_titles_and_section_headings(self):
        for path in lesson_files(REPO_ROOT):
            with self.subTest(path=path.relative_to(REPO_ROOT)):
                topic = path.parent.name
                expected = topic + (": Exercises" if path.stem.endswith("_exercises") else "")
                titles = []
                fenced = False
                for line in path.read_text().splitlines():
                    if line.lstrip().startswith("```"):
                        fenced = not fenced
                    elif not fenced and line.startswith("# "):
                        titles.append(line[2:])
                    elif not fenced:
                        self.assertIsNone(re.match(r"^#{2,6}\s+\d+(?:\.\d+)*[.)]\s+", line))
                self.assertEqual(titles, [expected])

    def test_local_links_point_to_files(self):
        # JupyterLab opens links as files: a link to a folder fails with an error.
        link = re.compile(r"\]\(([^)\s]+)\)")
        for path in lesson_files(REPO_ROOT):
            for cell in jupytext.read(path).cells:
                if cell.cell_type != "markdown":
                    continue
                # Code in fenced blocks or backticks is not a link.
                text = re.sub(r"```.*?```|`[^`\n]*`", "", cell.source, flags=re.S)
                for target in link.findall(text):
                    if re.match(r"^(\w+:|#)", target):
                        continue
                    with self.subTest(path=path.relative_to(REPO_ROOT), link=target):
                        resolved = path.parent / unquote(target.split("#", 1)[0])
                        self.assertTrue(resolved.is_file(), resolved)

    def test_notebook_checkpoints_are_not_lessons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            topic = root / "02 CADET Fundamentals" / "02.1 First Model"
            checkpoint = topic / ".ipynb_checkpoints" / "02.1a_first_model_lesson-checkpoint.md"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_text("# Cached lesson\n")
            lesson = topic / "02.1a_first_model_lesson.md"
            lesson.write_text("# 02.1 First Model\n")
            self.assertEqual(lesson_files(root), [lesson])


if __name__ == "__main__":
    unittest.main()
