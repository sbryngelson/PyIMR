"""Small forward-mode scalar used by the IMR tangent-linear solver.

The value path remains ordinary NumPy/SciPy code. Sensitivity solves replace
each scalar with a ``Dual`` carrying all requested directional derivatives,
then unpack the resulting augmented ODE for SciPy's real-valued integrators.
"""

from __future__ import annotations

from numbers import Real

import numpy as np

def at_set(array, index, value):
  """`array[index] = value`, returning the array, for numpy AND jax alike.

  jax arrays are immutable and spell this `.at[index].set(value)`. Dispatching
  on the attribute rather than on a namespace keeps the call sites free of an
  `xp` they would otherwise need only for this -- numpy arrays have no `.at`,
  and neither do the object arrays the Dual route builds. See PLAN.md W11
  stage 4: the thermal fields are assembled by slice assignment, which is what
  kept them off the jax backend.
  """
  if hasattr(array, "at"): return array.at[index].set(value)
  array[index] = value
  return array

class Dual:
  __slots__ = ("value", "tangent")
  __array_priority__ = 1000

  def __init__(self, value, tangent):
    self.value = float(value)
    self.tangent = np.asarray(tangent, dtype=float)

  @staticmethod
  def _coerce(other, width):
    if isinstance(other, Dual): return other
    return Dual(other, np.zeros(width))

  def _binary(self, other, value, left, right):
    other = self._coerce(other, self.tangent.size)
    return Dual(value(self.value, other.value), left(self.value, other.value) * self.tangent + right(self.value, other.value) * other.tangent)

  @staticmethod
  def _array_operation(values, operation):
    return np.frompyfunc(operation, 1, 1)(values)

  def __add__(self, other):
    if isinstance(other, np.ndarray): return self._array_operation(other, lambda value: self + value)
    return self._binary(other, lambda a, b: a + b, lambda _a, _b: 1.0, lambda _a, _b: 1.0)

  __radd__ = __add__

  def __sub__(self, other):
    if isinstance(other, np.ndarray): return self._array_operation(other, lambda value: self - value)
    return self._binary(other, lambda a, b: a - b, lambda _a, _b: 1.0, lambda _a, _b: -1.0)

  def __rsub__(self, other):
    if isinstance(other, np.ndarray): return self._array_operation(other, lambda value: value - self)
    return self._coerce(other, self.tangent.size).__sub__(self)

  def __mul__(self, other):
    if isinstance(other, np.ndarray): return self._array_operation(other, lambda value: self * value)
    return self._binary(other, lambda a, b: a * b, lambda _a, b: b, lambda a, _b: a)

  __rmul__ = __mul__

  def __truediv__(self, other):
    if isinstance(other, np.ndarray): return self._array_operation(other, lambda value: self / value)
    return self._binary(other, lambda a, b: a / b, lambda _a, b: 1.0 / b, lambda a, b: -a / b**2)

  def __rtruediv__(self, other):
    if isinstance(other, np.ndarray): return self._array_operation(other, lambda value: value / self)
    return self._coerce(other, self.tangent.size).__truediv__(self)

  def __pow__(self, other):
    if isinstance(other, np.ndarray): return self._array_operation(other, lambda value: self**value)
    other = self._coerce(other, self.tangent.size)
    result = self.value**other.value
    if not np.any(other.tangent):
      tangent = other.value * self.value ** (other.value - 1.0) * self.tangent
      return Dual(result, tangent)
    if self.value <= 0.0: raise ValueError("a differentiated exponent requires a positive base")
    tangent = result * (other.tangent * np.log(self.value) + other.value * self.tangent / self.value)
    return Dual(result, tangent)

  def __rpow__(self, other):
    if isinstance(other, np.ndarray): return self._array_operation(other, lambda value: value**self)
    return self._coerce(other, self.tangent.size).__pow__(self)

  def __neg__(self): return Dual(-self.value, -self.tangent)
  def __pos__(self): return self

  def __abs__(self):
    if self.value > 0.0: return self
    if self.value < 0.0: return -self
    return Dual(0.0, np.zeros_like(self.tangent))

  def sqrt(self):
    result = np.sqrt(self.value)
    return Dual(result, self.tangent / (2.0 * result))

  def cbrt(self):
    result = np.cbrt(self.value)
    return Dual(result, self.tangent / (3.0 * result**2))

  def exp(self):
    result = np.exp(self.value)
    return Dual(result, result * self.tangent)

  def expm1(self): return Dual(np.expm1(self.value), np.exp(self.value) * self.tangent)
  def log(self): return Dual(np.log(self.value), self.tangent / self.value)
  def log1p(self): return Dual(np.log1p(self.value), self.tangent / (1.0 + self.value))
  def arcsinh(self): return Dual(np.arcsinh(self.value), self.tangent / np.sqrt(1.0 + self.value**2))
  def sin(self): return Dual(np.sin(self.value), np.cos(self.value) * self.tangent)
  def cos(self): return Dual(np.cos(self.value), -np.sin(self.value) * self.tangent)

  def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
    if method != "__call__" or kwargs: return NotImplemented
    unary = {
      np.sqrt: "sqrt",
      np.cbrt: "cbrt",
      np.exp: "exp",
      np.expm1: "expm1",
      np.log: "log",
      np.log1p: "log1p",
      np.arcsinh: "arcsinh",
      np.sin: "sin",
      np.cos: "cos",
      np.absolute: "__abs__",
    }
    name = unary.get(ufunc)
    if name is not None and len(inputs) == 1: return getattr(self, name)()
    if len(inputs) == 2:
      left, right = inputs
      operations = {
        np.add: "__add__",
        np.subtract: "__sub__",
        np.multiply: "__mul__",
        np.divide: "__truediv__",
        np.true_divide: "__truediv__",
        np.power: "__pow__",
      }
      # `maximum` is not an arithmetic operation and has no reverse form: it
      # SELECTS one operand, so the tangent of the winner comes with it. `_rhs`
      # reaches it through the radius floor, which used Python's `max` until the
      # array namespace became swappable (W11 stage 2a) -- `max` cannot be
      # written against a namespace, and a traced array cannot be compared with
      # `>`. At the kink the two operands are equal and this returns the left,
      # matching what `max` did.
      if ufunc is np.maximum or ufunc is np.minimum:
        wins = primal(left) >= primal(right) if ufunc is np.maximum else primal(left) <= primal(right)
        chosen = left if wins else right
        return chosen if isinstance(chosen, Dual) else Dual(chosen, np.zeros_like(self.tangent))
      operation = operations.get(ufunc)
      if operation is not None:
        if isinstance(left, Dual): return getattr(left, operation)(right)
        reverse = {"__add__": "__radd__", "__sub__": "__rsub__", "__mul__": "__rmul__", "__truediv__": "__rtruediv__", "__pow__": "__rpow__"}[
          operation
        ]
        return getattr(right, reverse)(left)
    return NotImplemented

  def __float__(self): return self.value
  def __lt__(self, other): return self.value < primal(other)
  def __le__(self, other): return self.value <= primal(other)
  def __gt__(self, other): return self.value > primal(other)
  def __ge__(self, other): return self.value >= primal(other)

  def __eq__(self, other):
    try:
      return self.value == primal(other)
    except (TypeError, ValueError):
      return False

  def __ne__(self, other): return not self == other
  def __repr__(self): return f"Dual(value={self.value!r}, tangent={self.tangent!r})"
def primal(value): return value.value if isinstance(value, Dual) else value

def primal_array(values):
  array = np.asarray(values)
  if array.dtype != object: return np.asarray(array, dtype=float)
  return np.fromiter((float(primal(value)) for value in array.flat), dtype=float, count=array.size).reshape(array.shape)

def unpack(values, width):
  array = np.asarray(values, dtype=object)
  flat = array.ravel()
  primal_values = np.empty(flat.size)
  tangents = np.zeros((flat.size, width))
  for index, value in enumerate(flat):
    if isinstance(value, Dual):
      primal_values[index] = value.value
      tangents[index] = value.tangent
    else:
      primal_values[index] = value
  return primal_values.reshape(array.shape), tangents

def seed(value, width, index=None, scale=1.0):
  tangent = np.zeros(width)
  if index is not None: tangent[index] = scale
  return Dual(value, tangent)

def is_real_scalar(value): return isinstance(value, Real) and not isinstance(value, Dual)
