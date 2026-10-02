//! Exact accumulation of finite binary64 cashflows, with one rounding at readout.
//! Separate unsigned magnitudes retain cancellation across every exponent level.
use std::collections::BTreeSet;

// A finite f64 occupies bits 0..2097 in units of 2^-1074. Another 64 carry bits
// cover any usize-sized input tape; 34 words provide 2176 bits without allocation.
const WORDS: usize = 34;
#[derive(Clone)]
pub(crate) struct CurrencySum {
    positive: [u64; WORDS],
    negative: [u64; WORDS],
}
impl CurrencySum {
    pub fn new() -> Self {
        Self {
            positive: [0; WORDS],
            negative: [0; WORDS],
        }
    }
    pub fn add(&mut self, value: f64) {
        assert!(
            value.is_finite(),
            "currency accumulator requires finite inputs"
        );
        let bits = value.to_bits();
        let exponent = ((bits >> 52) & 0x7ff) as usize;
        let fraction = bits & ((1_u64 << 52) - 1);
        let significand = fraction | if exponent == 0 { 0 } else { 1_u64 << 52 };
        let shift = exponent.saturating_sub(1);
        let words = if value.is_sign_negative() {
            &mut self.negative
        } else {
            &mut self.positive
        };
        let index = shift / 64;
        let offset = shift % 64;
        Self::add_word(words, index, significand << offset);
        if offset > 11 {
            Self::add_word(words, index + 1, significand >> (64 - offset));
        }
    }
    fn add_word(words: &mut [u64; WORDS], mut index: usize, mut value: u64) {
        while value != 0 {
            let (next, carry) = words[index].overflowing_add(value);
            words[index] = next;
            value = u64::from(carry);
            index += 1;
        }
    }
    pub fn subtract(&mut self, other: &Self) {
        for i in 0..WORDS {
            Self::add_word(&mut self.positive, i, other.negative[i]);
            Self::add_word(&mut self.negative, i, other.positive[i]);
        }
    }
    pub fn difference(&self, other: &Self, reasons: &mut BTreeSet<String>) -> f64 {
        let mut result = self.clone();
        result.subtract(other);
        result.value(reasons)
    }
    pub fn value(&self, reasons: &mut BTreeSet<String>) -> f64 {
        // Preserve visibility when gross accumulated magnitudes exceed f64 range,
        // even if their exact difference is representable.
        if self.positive[32] >> 50 != 0
            || self.negative[32] >> 50 != 0
            || self.positive[33] != 0
            || self.negative[33] != 0
        {
            reasons.insert("numeric_range_approximation".into());
        }
        let negative = self
            .positive
            .iter()
            .rev()
            .cmp(self.negative.iter().rev())
            .is_lt();
        let (larger, smaller) = if negative {
            (&self.negative, &self.positive)
        } else {
            (&self.positive, &self.negative)
        };
        let mut magnitude = [0_u64; WORDS];
        let mut borrow = false;
        for i in 0..WORDS {
            let (a, first) = larger[i].overflowing_sub(smaller[i]);
            let (b, second) = a.overflowing_sub(u64::from(borrow));
            magnitude[i] = b;
            borrow = first || second;
        }
        let Some(top) = magnitude.iter().rposition(|v| *v != 0) else {
            return 0.0;
        };
        let mut highest = top * 64 + (63 - magnitude[top].leading_zeros() as usize);
        let sign = u64::from(negative) << 63;
        if highest < 52 {
            return f64::from_bits(sign | magnitude[0]);
        }
        let shift = highest - 52;
        let index = shift / 64;
        let offset = shift % 64;
        let mut significand = magnitude[index] >> offset;
        if offset > 11 {
            significand |= magnitude[index + 1] << (64 - offset);
        }
        if shift > 0 {
            let half = shift - 1;
            let half_set = magnitude[half / 64] & (1_u64 << (half % 64)) != 0;
            let lower_set = magnitude[..half / 64].iter().any(|v| *v != 0)
                || magnitude[half / 64] & ((1_u64 << (half % 64)) - 1) != 0;
            if half_set && (lower_set || significand & 1 != 0) {
                significand += 1;
                if significand == 1_u64 << 53 {
                    significand >>= 1;
                    highest += 1;
                }
            }
        }
        if highest > 2097 {
            reasons.insert("numeric_range_approximation".into());
            return f64::MAX.copysign(if negative { -1.0 } else { 1.0 });
        }
        f64::from_bits(sign | (((highest - 51) as u64) << 52) | (significand & ((1_u64 << 52) - 1)))
    }
}

pub(crate) fn currency_sum(values: &[f64], reasons: &mut BTreeSet<String>) -> f64 {
    let mut total = CurrencySum::new();
    for &value in values {
        total.add(value);
    }
    total.value(reasons)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn nested_cancellation_retains_every_scale_and_subnormal() {
        let mut total = CurrencySum::new();
        for exponent in -1074..=1023 {
            let value = if exponent < -1022 {
                f64::from_bits(1_u64 << (exponent + 1074))
            } else {
                2.0_f64.powi(exponent)
            };
            total.add(value);
        }
        total.add(-1.0);
        for exponent in (-1074..=1023).rev() {
            let value = if exponent < -1022 {
                f64::from_bits(1_u64 << (exponent + 1074))
            } else {
                2.0_f64.powi(exponent)
            };
            total.add(-value);
        }
        assert_eq!(total.value(&mut BTreeSet::new()), -1.0);
    }
    #[test]
    fn rounding_is_nearest_even_and_true_overflow_is_visible() {
        for (values, expected) in [
            (vec![1.0, 2.0_f64.powi(-53)], 1.0),
            (
                vec![1.0, 2.0_f64.powi(-53), f64::from_bits(1)],
                f64::from_bits(1.0_f64.to_bits() + 1),
            ),
            (
                vec![f64::MIN_POSITIVE, -f64::from_bits(1)],
                f64::from_bits((1_u64 << 52) - 1),
            ),
            (vec![f64::MAX, f64::MAX, -f64::MAX], f64::MAX),
            (vec![f64::MAX, f64::MAX], f64::MAX),
        ] {
            for sign in [-1.0, 1.0] {
                let mut reasons = BTreeSet::new();
                assert_eq!(
                    currency_sum(
                        &values.iter().map(|v| v * sign).collect::<Vec<_>>(),
                        &mut reasons
                    ),
                    expected * sign
                );
            }
        }
    }
}
