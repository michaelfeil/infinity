# SPDX-License-Identifier: MIT
# Copyright (c) 2023-now michaelfeil


class AutoPadding:
    """itertools.cycle with custom behaviour to pad to max length"""

    def __init__(self, length: int, **kwargs):
        self.length = length
        self.kwargs = kwargs

    def _resolve(self, x, iteration: int):
        """pad x to length of self.length"""
        x = typer_option_resolve(x)
        if not isinstance(x, (list, tuple)):
            return x
        elif len(x) == 1:
            return x[0]
        elif len(x) == self.length:
            return x[iteration]
        else:
            raise ValueError(f"Expected length {self.length} but got {len(x)}")

    def __iter__(self):
        """iterate over kwargs and pad them to length of self.length"""
        for iteration in range(self.length):
            kwargs = {}
            for key, value in self.kwargs.items():
                kwargs[key] = self._resolve(value, iteration)
            yield kwargs


def typer_option_resolve(*args):
    """returns the value or the default value"""
    if len(args) == 1:
        return (
            args[0].default  # if it is a typer option
            if hasattr(args[0], "default") and hasattr(args[0], "envvar")
            else args[0]  # if it is a normal value
        )
    return (a.default if (hasattr(a, "default") and hasattr(a, "envvar")) else a for a in args)
