# -*- coding: utf-8 -*-
"""

This script holds the path to the CADET executable used by the notebooks of
this chapter, so that it is written down once instead of in every notebook.

Call configure() at the top of a notebook, before building any model:

    import utility.cadet_runtime as cadet_runtime
    cadet_runtime.configure()

The path is taken from the CADET_PATH environment variable if it is set, and
otherwise from the current Python environment (the selected notebook kernel).
Set CADET_PATH to use a custom build instead.

"""

import os
import sys
import warnings

from cadet import Cadet


def executable():
    """Path to the CADET executable that the notebooks should use.

    Returns
    -------
    string

    Raises
    ------
    FileNotFoundError
        If the selected executable does not exist or is not a file.
    """
    if 'CADET_PATH' in os.environ:
        path = os.environ['CADET_PATH']
    elif os.name == 'nt':
        path = os.path.join(sys.prefix, 'Library', 'bin', 'cadet-cli.exe')
    else:
        path = os.path.join(sys.prefix, 'bin', 'cadet-cli')

    if not os.path.isfile(path):
        raise FileNotFoundError(
            "No CADET executable at '" + str(path) + "'. Install CADET in "
            "the current Python environment (" + sys.prefix + "), or set "
            "the CADET_PATH environment variable to the executable."
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
