"""Practice repository for the optional git exercise in 01.4 Contributing.

The repository lives in a temporary folder, so nothing touches the workshop files
and nothing is sent anywhere.
"""

import subprocess
import tempfile
from pathlib import Path


class PracticeRepository:
    """A local git repository with a `dev` branch and one commit."""

    def __init__(self):
        self.path = Path(tempfile.mkdtemp(prefix="git-practice-"))
        self.notes = self.path / "notes.md"
        self.git("init", "--initial-branch=dev", show=False)
        self.notes.write_text("# Column notes\n")
        self.git("add", "notes.md", show=False)
        self.git("commit", "-m", "Add column notes", show=False)
        print(f"Practice repository on branch dev: {self.path}")

    def git(self, *args, show=True):
        """Run a git command in the practice repository and print its output."""
        result = subprocess.run(
            # A fixed author, so commits work without a personal git configuration.
            ["git", "-c", "user.name=Workshop", "-c", "user.email=workshop@example.org", *args],
            cwd=self.path,
            capture_output=True,
            text=True,
        )
        if show:
            print((result.stdout + result.stderr).strip())

    def edit(self, line):
        """Add a line to notes.md, like editing a file in an editor."""
        with self.notes.open("a") as file:
            file.write(line + "\n")

    def history(self):
        """Show the commits of all branches as a graph."""
        self.git("log", "--oneline", "--graph", "--all", "--decorate")
