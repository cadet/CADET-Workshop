# -*- coding: utf-8 -*-
"""

This script holds the path to the CADET executable used by the notebooks of
this chapter, so that it is written down once instead of in every notebook.

Call configure() at the top of a notebook, before building any model:

    import utility.cadet_runtime as cadet_runtime
    cadet_runtime.configure()

The path is taken from the CADET_PATH environment variable if it is set, and
otherwise from DEFAULT_EXECUTABLE below. Participants running these notebooks
on their own machine set CADET_PATH, or edit the default.

"""

import os
import warnings

from cadet import Cadet

DEFAULT_EXECUTABLE = (
    r"C:\Users\jmbr\Desktop\CADET_compiled"
    r"\master8_post6a5withConsInitFixAndCSTRparInterface_b580377"
    r"\aRELEASE\bin\cadet-cli.exe"
)


def executable():
    """Path to the CADET executable that the notebooks should use.

    Returns
    -------
    string

    Raises
    ------
    FileNotFoundError
        If neither CADET_PATH nor the default points at an existing file.
    """
    path = os.environ.get('CADET_PATH', DEFAULT_EXECUTABLE)

    if not os.path.exists(path):
        raise FileNotFoundError(
            "No CADET executable at '" + str(path) + "'. Set the CADET_PATH "
            "environment variable, or edit DEFAULT_EXECUTABLE in "
            "utility/cadet_runtime.py."
        )

    return path


def configure():
    """Point CADET-Python at the executable and report which one it is.

    Returns
    -------
    string
        The configured path.
    """
    path = executable()

    # CADET-Python notes that it was handed the executable rather than the
    # installation root and infers the root itself. That is exactly what we
    # want, so the notice does not need to greet the audience on cell one.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            'ignore', message='.*not the root of the CADET installation.*')
        Cadet.cadet_path = path

    return path
