# Config editor assistance

The checked [v9 JSON Schema](../schemas/config-v9.schema.json) describes canonical authored JSON.
It offers field completion, declared numeric types, nullability, known value choices and closed
objects. It also describes dynamic scenario/limit/scoring records, nested optimizer bounds,
literal dotted or nested overrides, and the existing narrow coin-patch permissions.

The schema is generated from the [field registry](ai/generated/config_fields.md). Shared optimizer
bindings and coin permissions reuse the existing path/permission maps. Rust metadata owns strategy
parameters and their recommended ranges. Recommended search ranges are examples, not hard domains;
they do not create a search dimension when a bound is omitted. Optional portfolio HSL authority has
no default. Editor/schema validation never inserts defaults or reads referenced coin/config files.

Runtime validators remain authoritative for activation, cross-field constraints, override precedence,
prepared-dataset eligibility, native numeric representability and trading behavior. Editor types
describe canonical JSON serialization; existing CLI coercions continue to work. Canonical bound forms
are a number or arrays of one to three elements; a null third element means a continuous range.
Historical permissive inputs are not additional editor spellings. Result records and runtime metadata
have separate loading boundaries and do not validate as authored configs.

## VS Code

The repository's `.vscode/settings.json` associates `configs/examples/*.json` with the generated schema.
For configs stored elsewhere, add an association in your own workspace or user settings:

```json
{
    "json.schemas": [
        {
            "fileMatch": ["/my-configs/*.json"],
            "url": "./schemas/config-v9.schema.json"
        }
    ]
}
```

Use the schema file's absolute path when your settings/config files live outside this repository.
Do not add `$schema` to the authored config itself: the strict root accepts configuration fields only.
The association above targets JSON; HJSON should use an editor extension that supports JSON Schema
without changing the document's field contract.

## Generate and inspect

```bash
passivbot tool config-schema                         # print the schema
passivbot tool config-schema --check                 # check the repository artifact, read-only
passivbot tool config-schema --output schema.json     # export to an explicit destination
passivbot tool config-schema --field bot.long.hsl.drawdown_ratio_threshold
PYTHONPATH=src python src/tools/generate_config_field_reference.py --check
```

`--field` emits a metadata report, separate from authored policy: declared type/default, scope,
consumers, unit/denominator, domain, optimizer binding/recommendation and coin-override permission.
Unknown metadata is null. A declared default is a schema default, not proof that an effective
value came from that default; config explanation supplies execution-context provenance separately.

To regenerate the checked schema after a field-contract change:

```bash
passivbot tool config-schema --output schemas/config-v9.schema.json
PYTHONPATH=src python src/tools/generate_config_field_reference.py
pytest tests/test_config_json_schema.py tests/test_config_field_registry.py
```

The tests check the schema against draft 2020-12, every intentional public example, supported
strategy/HSL templates, sparse/null inputs, closed dynamic records, override paths and permissions.
