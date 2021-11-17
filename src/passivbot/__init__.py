import passivbot.utils.logs

passivbot.utils.logs.set_logger_class()

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
