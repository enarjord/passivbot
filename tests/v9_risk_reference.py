"""Independent rational oracle for the small v9 risk arithmetic graph.

This is test code, not a live policy or fallback. Fraction arithmetic and explicit
nearest-even binary32 rounding keep the oracle independent of Rust/Metal FMA.
"""

from fractions import Fraction
import struct


def f32(value):
    """Round an exact rational to binary32, including fused-expression results."""
    exact = value if isinstance(value, Fraction) else Fraction(value)
    negative = exact < 0
    magnitude = abs(exact)
    guess = struct.unpack("I", struct.pack("f", float(magnitude)))[0]
    candidates = []
    for bits in range(max(0, guess - 1), min(0x7F7FFFFF, guess + 1) + 1):
        number = struct.unpack("f", struct.pack("I", bits))[0]
        candidates.append((abs(Fraction(number) - magnitude), bits & 1, number))
    rounded = min(candidates)[2]
    return -rounded if negative else rounded


def add(a, b):
    return f32(Fraction(a) + Fraction(b))


def sub(a, b):
    return f32(Fraction(a) - Fraction(b))


def mul(a, b):
    return f32(Fraction(a) * Fraction(b))


def div(a, b):
    return f32(Fraction(a) / Fraction(b))


def fma(a, b, c):
    return f32(Fraction(a) * Fraction(b) + Fraction(c))


def coefficients(span):
    alpha = div(2, add(f32(span), 1))
    return alpha, sub(1, alpha)


def sample(*, previous_d, previous_m, previous_upnl, upnl, net, budget,
           slow_span, fast_span, first_in_minute=True):
    """One authored event; lifecycle assertions live in fixtures, not this oracle."""
    _, beta_slow = coefficients(slow_span)
    alpha_fast, beta_fast = coefficients(fast_span)
    upnl, net, budget = map(f32, (upnl, net, budget))
    loss = sub(sub(previous_upnl, upnl), net)
    drawdown = max(0.0, fma(beta_slow, previous_d, loss)
                   if first_in_minute else add(previous_d, loss))
    equity = add(budget, upnl)
    peak = add(equity, drawdown)
    ratio = 1.0 if equity <= 0 or peak <= 0 else div(drawdown, peak)
    ema = fma(beta_fast, previous_m, mul(alpha_fast, ratio))
    return dict(drawdown=drawdown, raw=ratio, ema=ema, equity=equity,
                nonpositive_equity=equity <= 0 or peak <= 0)


def allowance(balance, effective_pct, drawdown):
    return max(0.0, fma(f32(effective_pct), add(f32(balance), drawdown), -drawdown))
