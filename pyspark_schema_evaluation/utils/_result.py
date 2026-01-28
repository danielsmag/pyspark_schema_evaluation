from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Callable,
    Generic,
    TypeAlias,
    TypeGuard,
    TypeVar,
    Union,
    cast,
    Optional,
)

T = TypeVar("T")   # success type
U = TypeVar("U")   # success type after transform
E = TypeVar("E")   # error type
F = TypeVar("F")   # error type after transform
ExcType = TypeVar("ExcType", bound=Exception)
K = TypeVar("K")   # key type


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    value: T
    
    def unwrap(self) -> T:
        return self.value

    def and_then(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """Chain computations that may fail (aka bind/flatMap)"""
        return fn(self.value)

    def or_else(self, fn: Callable[[E], Result[T, F]]) -> Result[T, F]:
        """Handle error with a fallback that may also fail."""
        return cast(Ok[T], self)

@dataclass(frozen=True, slots=True)
class Err(Generic[E]):
    error: E
    
    def and_then(self, fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
        """On Err, skip the function and return self."""
        return cast(Err[E], self)
    
    def or_else(self, fn: Callable[[E], Result[T, F]]) -> Result[T, F]:
        """On Err, apply the recovery function."""
        return fn(self.error)


Result: TypeAlias = Union[Ok[T], Err[E]]


@dataclass(frozen=True, slots=True)
class Some(Generic[T]):
    value: T
    
    def map(self, fn: Callable[[T], U]) -> Option[U]:
        return Some(fn(self.value))
    
    def and_then(self, fn: Callable[[T], Option[U]]) -> Option[U]:
        return fn(self.value)
    
    def unwrap_or(self, default: T) -> T:
        return self.value


@dataclass(frozen=True, slots=True)
class Nothing:
    pass


Option: TypeAlias = Union[Some[T], Nothing]


def is_some(opt: Option[T]) -> TypeGuard[Some[T]]:
    return isinstance(opt, Some)

def is_nothing(opt: Option[T]) -> bool:
    return isinstance(opt, Nothing)

def unwrap_option(opt: Option[T]) -> T:
    """Return the Some value, or raise RuntimeError if Nothing."""
    if isinstance(opt, Some):
        return opt.value
    raise RuntimeError("called unwrap_option() on Nothing")

def unwrap_option_or(opt: Option[T], default: T) -> T:
    return opt.value if isinstance(opt, Some) else default

def map_option(opt: Option[T], fn: Callable[[T], U]) -> Option[U]:
    """Transform the Some value; pass Nothing through."""
    if isinstance(opt, Some):
        return Some(fn(opt.value))
    return Nothing()

def and_then_option(opt: Option[T], fn: Callable[[T], Option[U]]) -> Option[U]:
    """Chain computations on Option."""
    if isinstance(opt, Some):
        return fn(opt.value)
    return Nothing()

def option_to_result(opt: Option[T], error: E) -> Result[T, E]:
    """Convert Option to Result, using error if Nothing."""
    if isinstance(opt, Some):
        return Ok[T](opt.value)
    return Err(error)

def result_to_option(r: Result[T, E]) -> Option[T]:
    """Convert Result to Option, discarding error info."""
    if isinstance(r, Ok):
        return Some[T](r.value)
    return Nothing()


def is_ok(r: Result[T, E]) -> TypeGuard[Ok[T]]:
    return isinstance(r, Ok)

def is_err(r: Result[T, E]) -> TypeGuard[Err[E]]:
    return isinstance(r, Err)

def validate(x: int) -> Result[int, str]:
    if x < 0:
        return Err[str]("x is negative")
    return Ok[int](x)



def unwrap(r: Ok[T] | Err[E]) -> T:
    """Return the Ok value, or raise a RuntimeError if Err."""
    if isinstance(r, Ok):
        return r.value
    raise RuntimeError(f"called unwrap() on Err: {r.error!r}")

def unwrap_err(r: Ok[T] | Err[E]) -> E:
    """Return the Err value, or raise a RuntimeError if Ok."""
    if isinstance(r, Err):
        return r.error
    raise RuntimeError(f"called unwrap_err() on Ok: {r.value!r}")

def unwrap_all(r: Ok[T] | Err[E]) -> Union[T, E]:
    """Return the inner value, whether Ok or Err."""
    if isinstance(r, Ok):
        return r.value
    if isinstance(r, Err):
        return r.error
    raise RuntimeError(f"called unwrap_all() on invalid type: {type(r)!r}")

def unwrap_or(r: Result[T, E], default: T) -> T:
    return r.value if isinstance(r, Ok) else default

def expect(r: Result[T, E], msg: str) -> T:
    """Return the Ok value, or raise RuntimeError with custom message if Err."""
    if isinstance(r, Ok):
        return r.value
    raise RuntimeError(f"{msg}: {r.error!r}")


def map_(r: Result[T, E], fn: Callable[[T], U]) -> Result[U, E]:
    """Transform the Ok value; pass Err through unchanged."""
    if isinstance(r, Ok):
        return Ok[U](fn(r.value))
    return cast(Err[E], r)

def map_err(r: Result[T, E], fn: Callable[[E], F]) -> Result[T, F]:
    """Transform the Err value; pass Ok through unchanged."""
    if isinstance(r, Err):
        return Err[F](fn(r.error))
    return cast(Ok[T], r)

def tap_err(r: Result[T, E], fn: Callable[[E], None]) -> Result[T, E]:
    """Run side effect on Err, return original Result unchanged."""
    if isinstance(r, Err):
        fn(r.error)
    return r

def and_then(r: Result[T, E], fn: Callable[[T], Result[U, E]]) -> Result[U, E]:
    """Chain computations that may fail (aka bind/flatMap)."""
    if isinstance(r, Ok):
        return fn(r.value)
    return cast(Err[E], r)

def or_else(r: Result[T, E], fn: Callable[[E], Result[T, F]]) -> Result[T, F]:
    """Handle error with a fallback that may also fail."""
    if isinstance(r, Err):
        return fn(r.error)
    return cast(Ok[T], r)

def to_optional(r: Result[T, E]) -> Option[T]:
    """Convert Ok to Some, Err to None."""
    return Some[T](r.value) if isinstance(r, Ok) else Nothing()

def from_optional(opt: Optional[T], error: E) -> Result[T, E]:
    """Convert Optional to Result, using error if None."""
    return Ok[T](opt) if opt is not None else Err[E](error)

def try_(fn: Callable[[], T]) -> Result[T, Exception]:
    """Wrap a function call, catching exceptions as Err."""
    try:
        return Ok[T](fn())
    except Exception as e:
        return Err[Exception](e)

def try_call_(fn: Callable[..., T], *args, **kwargs) -> Result[T, Exception]:
    """Call fn with args/kwargs, catching exceptions as Err."""
    try:
        return Ok[T](fn(*args, **kwargs))
    except Exception as e:
        return Err[Exception](e)
    
def try_option_(
    fn: Callable[[], T | None], 
    error: str = "Value was None"
) -> Result[T, str]:
    """Call fn, treating None and exceptions as string errors."""
    try:
        value: T | None = fn()
        if value is not None:
            return Ok[T](value)
        return Err[str](error)
    except Exception as e:
        return Err[str](f"{error}: {e}")


def try_option_typed_(
    fn: Callable[[], T | None], 
    error_on_none: E, 
    error_on_exc: Callable[[Exception], E]
) -> Result[T, E]:
    """Call fn with typed errors."""
    try:
        value = fn()
        if value is not None:
            return Ok(value)
        return Err(error_on_none)
    except Exception as e:
        return Err(error_on_exc(e))

def try_with_(
    fn: Callable[[], T], 
    exc_types: tuple[type[Exception], ...] = (Exception,)
) -> Result[T, Exception]:
    """Wrap a function call, catching specific exception types as Err."""
    try:
        return Ok[T](fn())
    except exc_types as e:
        return Err[Exception](e)

def collect(results: list[Result[T, E]]) -> Result[list[T], E]:
    """Convert list of Results to Result of list. Stops at first Err."""
    values: list[T] = []
    for r in results:
        if isinstance(r, Err):
            return r
        values.append(r.value)
    return Ok[list[T]](values)

def partition(results: list[Result[T, E]]) -> tuple[list[T], list[E]]:
    """Split results into (ok_values, err_values)."""
    oks: list[T] = []
    errs: list[E] = []
    for r in results:
        if isinstance(r, Ok):
            oks.append(r.value)
        else:
            errs.append(r.error)
    return oks, errs

def partition_with_key(
    items: list[tuple[K, Result[T, E]]]
) -> tuple[list[tuple[K, T]], list[tuple[K, E]]]:
    """Split (key, Result) pairs into successes and failures, preserving keys."""
    oks: list[tuple[K, T]] = []
    errs: list[tuple[K, E]] = []
    for key, r in items:
        if isinstance(r, Ok):
            oks.append((key, r.value))
        else:
            errs.append((key, r.error))
    return oks, errs


# Function	Purpose
# unwrap_err	Get the error value (raises if Ok)
# expect	Like unwrap but with custom error message
# or_else	Error recovery chain (opposite of and_then)
# to_optional	Convert Ok(v) → v, Err → None
# from_optional	Convert None → Err, value → Ok
# try_	Wrap exception-throwing code as Result
# try_with	Same but catches specific exception types
# collect	list[Result[T, E]] → Result[list[T], E] (stops at first error)
# partition	Split results into (ok_values, err_values)
# validate	Validate a value, returning Err if invalid
# and_then_option	Option-style chaining (map/flatMap)
# option_to_result	Convert Option to Result, using error if Nothing
# result_to_option	Convert Result to Option, discarding error info
