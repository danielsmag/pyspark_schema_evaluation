from __future__ import annotations

from typing import Any, Optional
from pyspark.sql import DataFrame


__all__ = [
    "_resolve_callable",
    "_locate_df",
    "_rebuild_args_kwargs",
    "_locate_arg",
]


def _resolve_callable(
    value_or_callable: Any, self_obj: Any | object | None, decorator: str
) -> Any | object:
    if not callable(value_or_callable):
        return value_or_callable
    if self_obj is None:
        try:
            return value_or_callable()
        except TypeError as exc:
            raise ValueError(
                f"@{decorator}: callable needs a bound instance (self/cls) or must be callable with no args: {exc}"
            ) from exc
    try:
        return value_or_callable(self_obj)
    except Exception as exc:
        raise ValueError(
            f"@{decorator}: failed to resolve callable on {self_obj!r}: {exc}"
        ) from exc


def _locate_df(
    args, kwargs, arg_name_for_df: str
) -> tuple[DataFrame | Any, int | None, bool]:
    df: Optional[DataFrame] = None
    df_index: Optional[int] = None
    df_in_kwargs: bool = False

    if arg_name_for_df in kwargs and isinstance(kwargs[arg_name_for_df], DataFrame):
        df = kwargs[arg_name_for_df]
        df_in_kwargs = True

    if df is None:
        for idx, arg in enumerate(args):
            if isinstance(arg, DataFrame):
                df = arg
                df_index = idx
                break

    if df is None:
        raise ValueError(
            f"@{arg_name_for_df}: Could not locate a DataFrame argument "
            f"Expected {arg_name_for_df} as kwarg or a positional DataFrame"
        )
    return df, df_index, df_in_kwargs


def _locate_arg(
    args, kwargs, arg_name_for_arg: str, raise_error: bool = False
) -> tuple[Any | object, int | None, bool]:
    arg: Any = None
    arg_index: Optional[int] = None
    arg_in_kwargs: bool = False

    if arg_name_for_arg in kwargs and isinstance(
        kwargs[arg_name_for_arg], str | int | float | bool | list | dict
    ):
        arg = kwargs[arg_name_for_arg]
        arg_in_kwargs = True

    if arg is None:
        for idx, arg in enumerate(args):
            if isinstance(arg, str | int | float | bool | list | dict):
                arg = arg
                arg_index = idx
                break

    if arg is None:
        if raise_error:
            raise ValueError(
                f"@{arg_name_for_arg}: Could not locate a argument "
                f"Expected {arg_name_for_arg} as kwarg or a positional argument"
            )
    return arg, arg_index, arg_in_kwargs


def _rebuild_args_kwargs(
    df_valid: DataFrame,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    df_index: int | None,
    df_in_kwargs: bool,
    arg_name_for_df: str,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    if df_in_kwargs:
        new_kwargs: dict[Any, Any] = dict(kwargs)
        new_kwargs[arg_name_for_df] = df_valid
        return args, new_kwargs
    if df_index is None:
        raise RuntimeError("df_index is None after locating DataFrame")
    args_list: list[Any] = list(args)
    args_list[df_index] = df_valid
    return tuple(args_list), kwargs
