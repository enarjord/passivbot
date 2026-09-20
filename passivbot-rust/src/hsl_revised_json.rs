//! Exact numeric transport for experimental HSL JSON bindings only. Keep legacy
//! orchestrator parsing unchanged while preserving every submitted binary64 value.
use serde::{de::Error, Deserialize, Deserializer};
use serde_json::value::RawValue;
use std::collections::BTreeMap;

fn parse<E: Error>(raw: &RawValue) -> Result<f64, E> {
    let value: f64 = raw.get().parse().map_err(E::custom)?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(E::custom("nonfinite HSL JSON number"))
    }
}
pub(crate) fn number<'de, D: Deserializer<'de>>(d: D) -> Result<f64, D::Error> {
    let raw = Box::<RawValue>::deserialize(d)?;
    parse(&raw)
}
pub(crate) fn optional<'de, D: Deserializer<'de>>(d: D) -> Result<Option<f64>, D::Error> {
    Option::<Box<RawValue>>::deserialize(d)?
        .map(|raw| parse(&raw))
        .transpose()
}
#[derive(Deserialize)]
struct Number(#[serde(deserialize_with = "number")] f64);
pub(crate) fn prices<'de, D: Deserializer<'de>>(d: D) -> Result<BTreeMap<i64, f64>, D::Error> {
    Ok(BTreeMap::<i64, Number>::deserialize(d)?
        .into_iter()
        .map(|(t, p)| (t, p.0))
        .collect())
}
