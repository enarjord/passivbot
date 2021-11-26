from __future__ import annotations

import logging
import os
import sys

import passivbot.utils.logs

passivbot.utils.logs.set_logger_class()

log = logging.getLogger(__name__)

if "--nojit" in sys.argv:
    os.environ["NOJIT"] = "true"

if os.environ.get("NOJIT", "false") in ("true", "1"):

    log.info("numba.njit compilation is disabled")

    def numba_njit(pyfunc=None, **kwargs):
        def wrap(func):
            return func

        if pyfunc is not None:
            return wrap(pyfunc)
        else:
            return wrap

    def numba_jitclass(cls_or_spec=None, spec=None):
        def wrap(cls):
            return cls

        if cls_or_spec is not None:
            return wrap(cls_or_spec)
        return wrap


else:
    log.info("numba.njit compilation is enabled")
    from numba import njit as numba_njit  # type: ignore[no-redef]
    from numba.experimental import jitclass as numba_jitclass  # type: ignore[no-redef]

try:
    from .version import __version__
except ImportError:  # pragma: no cover
    __version__ = "0.0.0.not-installed"
    try:
        from importlib.metadata import version
        from importlib.metadata import PackageNotFoundError

        try:
            __version__ = version("passivbot")
        except PackageNotFoundError:
            # package is not installed
            pass
    except ImportError:
        try:
            from importlib_metadata import version  # type: ignore[no-redef]
            from importlib_metadata import PackageNotFoundError  # type: ignore[no-redef]

            try:
                __version__ = version("passivbot")
            except PackageNotFoundError:
                # package is not installed
                pass
        except ImportError:
            try:
                from pkg_resources import get_distribution, DistributionNotFound

                try:
                    __version__ = get_distribution("passivbot").version
                except DistributionNotFound:
                    # package is not installed
                    pass
            except ImportError:
                # pkg resources isn't even available?!
                pass
