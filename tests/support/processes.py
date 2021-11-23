from __future__ import annotations

import pprint

import attr


@attr.s(frozen=True)
class ProcessResult:
    """
    This class holds the resulting data from a subprocess command.

    :keyword int exitcode:
        The exitcode returned by the process
    :keyword str stdout:
        The ``stdout`` returned by the process
    :keyword str stderr:
        The ``stderr`` returned by the process
    :keyword list,tuple cmdline:
        The command line used to start the process

    .. admonition:: Note

        Cast :py:class:`~ProcessResult` to a string to pretty-print it.
    """

    exitcode = attr.ib()
    stdout = attr.ib()
    stderr = attr.ib()
    cmdline = attr.ib(default=None, kw_only=True)

    @exitcode.validator
    def _validate_exitcode(self, attribute, value):
        if not isinstance(value, int):
            raise ValueError(f"'exitcode' needs to be an integer, not '{type(value)}'")

    def __str__(self):
        """
        Pretty print the class instance.
        """
        message = self.__class__.__name__
        if self.cmdline:
            message += f"\n Command Line: {self.cmdline}"
        if self.exitcode is not None:
            message += f"\n Exitcode: {self.exitcode}"
        if self.stdout or self.stderr:
            message += "\n Process Output:"
        if self.stdout:
            message += f"\n   >>>>> STDOUT >>>>>\n{self.stdout}\n   <<<<< STDOUT <<<<<"
        if self.stderr:
            message += f"\n   >>>>> STDERR >>>>>\n{self.stderr}\n   <<<<< STDERR <<<<<"
        return message + "\n"


@attr.s(frozen=True)
class ProcessJsonResult(ProcessResult):
    """
    This class holds the resulting data from a subprocess command.

    :keyword dict json:
        The dictionary returned from the process ``stdout`` if it could JSON decode it.

    Please look at :py:class:`~ProcessResult` for the additional supported keyword
    arguments documentation.
    """

    json = attr.ib(default=None, kw_only=True)

    def __str__(self):
        """
        Pretty print the class instance.
        """
        message = super().__str__().rstrip()
        if self.json:
            message += "\n JSON Object:\n"
            message += "".join(f"  {line}" for line in pprint.pformat(self.json))
        return message + "\n"

    def __eq__(self, other):
        """
        Allow comparison against the parsed JSON or the output
        """
        if self.json:
            return self.json == other
        return self.stdout == other
