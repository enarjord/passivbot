import os


if "NOJIT" in os.environ and os.environ["NOJIT"] == "true":
    print("not using numba")

    def numba_njit(pyfunc=None, **kwargs):
        def wrap(func):
            return func

        if pyfunc is not None:
            return wrap(pyfunc)
        else:
            return wrap


else:
    print("using numba")
    from numba import njit as numba_njit  # type: ignore[no-redef]
