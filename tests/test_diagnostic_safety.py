import json

import pytest

from live.diagnostic_safety import (
    bounded_exception_code,
    bounded_exception_status,
    bounded_exception_type,
    bounded_exchange_error_context,
    bounded_exchange_error_context_from_mapping,
    exception_text_contains,
    exception_type_name_contains,
)


def test_bounded_exception_status_and_code_keep_safe_direct_values():
    error = RuntimeError("api_key=hidden")
    error.status = "503"
    error.code = "10006"

    assert bounded_exception_status(error) == "503"
    assert bounded_exception_code(error) == "10006"


def test_bounded_exception_status_and_code_use_safe_info_fallbacks():
    error = RuntimeError("api_key=hidden")
    error.status = "500?api_key=hidden"
    error.code = "sk_live_7E4v93kR2mN6pQ8t"
    error.info = {"status": "429", "retCode": "-1003"}

    assert bounded_exception_status(error) == "429"
    assert bounded_exception_code(error) == "-1003"


def test_bounded_exception_code_rejects_identifier_shaped_values():
    error = RuntimeError("api_key=hidden")
    error.code = "sk_live_7E4v93kR2mN6pQ8t"
    error.info = {"retCode": "RATE_LIMIT"}

    assert bounded_exception_code(error) is None


def test_bounded_exchange_error_context_extracts_structured_ccxt_payload():
    error = RuntimeError(
        'gate {"label":"INVALID_PARAM_VALUE","message":"invalid argument: size"}'
    )

    assert bounded_exchange_error_context(error) == {
        "error_label": "INVALID_PARAM_VALUE",
        "error_reason": "invalid argument: size",
    }


def test_bounded_exchange_error_context_extracts_ccxt_native_message_keys():
    binance = RuntimeError('binance {"code":-1013,"msg":"invalid quantity"}')
    bybit = RuntimeError('bybit {"retCode":10001,"retMsg":"position idx mismatch"}')

    assert bounded_exchange_error_context(binance) == {
        "error_code": "-1013",
        "error_reason": "invalid quantity",
    }
    assert bounded_exchange_error_context(bybit) == {
        "error_code": "10001",
        "error_reason": "position idx mismatch",
    }


def test_bounded_exchange_error_context_extracts_payload_status():
    error = RuntimeError('exchange {"status":429,"message":"too many requests"}')

    assert bounded_exchange_error_context(error) == {
        "error_status": "429",
        "error_reason": "too many requests",
    }


def test_bounded_exchange_error_context_extracts_rejected_result_mapping():
    result = {
        "status": "rejected",
        "info": {
            "status": 400,
            "retCode": 10001,
            "label": "INVALID_PARAM_VALUE",
            "message": "bad client id abcdefghijklmnopqrstuvwxyz012345",
            "raw": {"apiKey": "must-not-leak"},
        },
    }

    assert bounded_exchange_error_context_from_mapping(result) == {
        "error_status": "400",
        "error_code": "10001",
        "error_label": "INVALID_PARAM_VALUE",
        "error_reason": "bad client id <redacted>",
    }


def test_bounded_exchange_error_context_prefers_okx_per_order_details():
    result = {
        "status": "rejected",
        "info": {
            "code": "1",
            "msg": "",
            "data": [
                {
                    "ordId": "must-not-leak",
                    "sCode": "51008",
                    "sMsg": "Order failed. Insufficient balance.",
                }
            ],
        },
    }

    assert bounded_exchange_error_context_from_mapping(result) == {
        "error_code": "51008",
        "error_reason": "Order failed. Insufficient balance.",
    }


def test_bounded_exchange_error_context_redacts_long_tokens_and_sensitive_reasons():
    error = RuntimeError(
        'gate {"label":"INVALID_PARAM_VALUE",'
        '"message":"bad client id abcdefghijklmnopqrstuvwxyz012345"}'
    )
    sensitive = RuntimeError(
        'gate {"label":"INVALID_PARAM_VALUE","message":"api_key=do-not-log"}'
    )

    assert bounded_exchange_error_context(error) == {
        "error_label": "INVALID_PARAM_VALUE",
        "error_reason": "bad client id <redacted>",
    }
    assert bounded_exchange_error_context(sensitive) == {
        "error_label": "INVALID_PARAM_VALUE",
        "error_reason": "api_key=<redacted>",
    }


@pytest.mark.parametrize("prefix", ["", "L1 error: "])
def test_hyperliquid_signer_rejection_has_stable_label_and_safe_reason(prefix):
    payload = {
        "status": "err",
        "response": f"{prefix}User or API Wallet 0x{'a' * 40} does not exist.",
    }
    assert bounded_exchange_error_context(RuntimeError("hyperliquid " + json.dumps(payload))) == {
        "error_label": "HYPERLIQUID_UNRECOGNIZED_WALLET",
        "error_reason": "Hyperliquid does not recognize the API wallet.",
    }


@pytest.mark.parametrize("action", ["order", "cancel"])
def test_hyperliquid_per_order_reason_reaches_the_structured_event(action):
    from types import SimpleNamespace

    from live.event_bus import EventTypes, LiveEvent, format_console_event
    from live.event_emitters import emit_execution_order_event

    events = []
    bot = SimpleNamespace(
        _current_live_event_cycle_id=lambda: "cy_1",
        _emit_live_event=lambda event_type, **fields: events.append(LiveEvent(event_type, **fields)),
    )
    error = RuntimeError("hyperliquid " + json.dumps({
        "status": "ok",
        "response": {
            "type": action,
            "data": {
                "statuses": [{"resting": {"oid": 1}}, {"error": "Order has invalid size."}],
            },
        },
    }))

    emit_execution_order_event(
        bot,
        event_type=EventTypes.EXECUTION_AMBIGUOUS,
        order={"symbol": "ETH/USDC:USDC"},
        action="create" if action == "order" else "cancel",
        status="degraded",
        error=error,
    )

    assert len(events) == 1
    assert events[0].data["error_reason"] == "Order has invalid size."
    assert "reason=Order has invalid size." in format_console_event(events[0])


@pytest.mark.parametrize(("message", "sanitized"), [
    ("API key expired. Generate a new token.", "API key expired. Generate a new token."),
    ("Invalid signature for the configured private key.", "Invalid signature for the configured private key."),
    ('Could not authenticate api_key="short value", retry later',
     "Could not authenticate api_key=<redacted>, retry later"),
    ("Rejected. X-MBX-APIKEY: tiny", "Rejected. X-MBX-APIKEY: <redacted>"),
    ("Invalid token. Bearer ab.c.d", "Invalid token. <redacted-auth>"),
    ("Invalid token. Bearer .tiny123", "Invalid token. <redacted-auth>"),
    ("Rejected Bearer token ?tiny123", "Rejected <redacted-auth>"),
    ("Rejected Bearer!tiny123", "Rejected <redacted-auth>"),
    ("Rejected Bearer'tiny123'", "Rejected <redacted-auth>"),
    ('Rejected Bearer"tiny123"', "Rejected <redacted-auth>"),
    ("Rejected Bearer(tiny123)", "Rejected <redacted-auth>"),
    ("Rejected Basic auth[tiny123]", "Rejected <redacted-auth>"),
    ("Invalid Bearer'abc.def.ghi' supplied", "Invalid <redacted-auth> supplied"),
    ("Rejected Basic auth ,tiny123", "Rejected <redacted-auth>"),
    ("Rejected Basic ;tiny123", "Rejected <redacted-auth>"),
    ("Bearer! Rotate it.", "Bearer! Rotate it."),
    ('Error: "Invalid API key."', 'Error: "Invalid API key."'),
    ("(Invalid API key.)", "(Invalid API key.)"),
    ("Invalid passphrase... try again", "Invalid passphrase... try again"),
    ("Invalid passphrase!!", "Invalid passphrase!!"),
    ("Rejected. Authorization: Basic aGk=", "Rejected. <redacted-header>"),
    ("Rejected. Cookie: session=hidden; second=value", "Rejected. <redacted-header>"),
    ('Rejected. Set-Cookie": "csrf=short; other=value"', "Rejected. <redacted-header>"),
    ("Failed OK-ACCESS-KEY: small.", "Failed OK-ACCESS-KEY: <redacted>"),
    ('Bad api_key="first\nsecond', "Bad api_key=<redacted>"),
    ("Rejected. Authorization: Invalid\n token=short", "Rejected. <redacted-header>"),
    ("Rejected passphrase:\nshort-secret", "Rejected passphrase: <redacted>"),
    ("Rejected API key:\t\r\nshort-secret", "Rejected API key: <redacted>"),
    ("Rejected. Seed phrase: synthetic multiword secret\n second line",
     "Rejected. Seed phrase: <redacted>"),
    ("Rejected. -----BEGIN PRIVATE KEY-----\n short-value\n-----END PRIVATE KEY-----",
     "Rejected. <redacted-key>"),
    ('Request invalid {"action":{"type":"order"},"signature":{"r":"short","s":"secret"}}',
     "Request invalid <redacted-payload>"),
    ("Unable to verify signature=0x" + "a" * 130, "Unable to verify signature=<redacted>"),
    ("Check wallet 0x" + "c" * 40, "Check wallet 0x" + "c" * 40),
])
def test_error_reason_preserves_words_and_redacts_authentication_values(message, sanitized):
    assert bounded_exchange_error_context_from_mapping({"message": message}) == {
        "error_reason": sanitized,
    }


@pytest.mark.parametrize(
    "label",
    ["api_passphrase", "API key", "OK-ACCESS-PASSPHRASE", "session_id", "private key", "seed phrase"],
)
def test_ambiguous_unquoted_credential_does_not_leak_a_suffix(label):
    assert bounded_exchange_error_context_from_mapping({
        "message": f"Rejected {label}: synthetic multiword secret",
    }) == {"error_reason": f"Rejected {label}: <redacted>"}


@pytest.mark.parametrize(("message", "sanitized"), [
    ("Invalid passphrase 'short-secret'", "Invalid passphrase <redacted>"),
    ('Invalid "passphrase" "short-secret"', 'Invalid "passphrase" <redacted>'),
    ("Invalid 'API key' ('short-secret')", "Invalid 'API key' <redacted>"),
    ('Invalid "passphrase". Rotate it.', 'Invalid "passphrase". Rotate it.'),
    ('Invalid "API key":"short-secret"', 'Invalid "API key":<redacted>'),
    ('Invalid "passphrase"short-secret', 'Invalid "passphrase"<redacted>'),
    ("Invalid passphrase'short-secret'", "Invalid passphrase<redacted>"),
    ("Invalid passphrase !tiny123", "Invalid passphrase <redacted>"),
    ("Invalid passphrase .tiny123", "Invalid passphrase <redacted>"),
    ("Invalid passphrase ?tiny123", "Invalid passphrase <redacted>"),
    ("Invalid passphrase!tiny123", "Invalid passphrase<redacted>"),
    ("Invalid passphrase! Rotate it.", "Invalid passphrase! Rotate it."),
    ("Invalid api_key：tiny123", "Invalid api_key_<redacted>"),
    ("'tiny123' is an invalid private key", "<redacted> is an invalid private key"),
    (
        "'tiny123' is an invalid x-a-b-c-d-e-api-key",
        "<redacted> is an invalid x-a-b-c-d-e-api-key",
    ),
    ("Invalid passphrase; supplied short-secret", "Invalid passphrase<redacted>"),
    ("Malformed Bearer 'short-secret'", "Malformed <redacted-auth>"),
    ("Bearer token expired.", "Bearer token expired."),
    ("Invalid x_api_key: short-secret", "Invalid x_api_key: <redacted>"),
    ('Invalid "x_api_key" "short-secret"', 'Invalid "x_api_key" <redacted>'),
    ('Invalid "x-a-b-c-d-e-api-key" "shortsec"', 'Invalid "x-a-b-c-d-e-api-key" <redacted>'),
    ("Rejected x_authorization: Basic short-secret", "Rejected <redacted-header>"),
    ("Authorization failed: Basic auth dGlueQ==", "Authorization failed: <redacted-auth>"),
    ("Authorization failed: Basic authentication dGlueQ==", "Authorization failed: <redacted-auth>"),
    ("Basic auth: dGlueQ==", "Basic auth: <redacted>"),
    ("Bearer token=short-secret", "Bearer token=<redacted>"),
    ("Basic authentication is required.", "Basic authentication is required."),
    ("Authorization abc123", "Authorization <redacted>"),
    ("authorization abc123 rejected", "authorization <redacted>"),
    ("Authorization header abc123", "Authorization header <redacted>"),
    ("Missing Authorization header", "Missing Authorization header"),
    ("Authorization failed.", "Authorization failed."),
    ("Authorization denied", "Authorization denied"),
    ("Cookie expired", "Cookie expired"),
    ("Proxy-Authorization abc123", "Proxy-Authorization <redacted>"),
    ("Set-Cookie abc123", "Set-Cookie <redacted>"),
    ("KC-API-SIGN: shortsig", "KC-API-SIGN: <redacted>"),
    ("Invalid key=shortsec", "Invalid key=<redacted>"),
    ("Rejected credentials=shortsec", "Rejected credentials=<redacted>"),
    ("x-a-b-c-d-e-Api-Key=shortsec", "x-a-b-c-d-e-Api-Key=<redacted>"),
    ("Invalid API-key, IP, or permissions for action.",
     "Invalid API-key, IP, or permissions for action."),
    ("Signature for this request is not valid.", "Signature for this request is not valid."),
    ("The API key does not have trade permission.", "The API key does not have trade permission."),
    ("Your token balance is too low", "Your token balance is too low"),
    ("Basic order validation failed: price tick 0.01", "Basic order validation failed: price tick 0.01"),
    ("Basic order validation failed: price tick 0.01 secret", "<redacted-auth>"),
    ("Invalid api_key (short-secret)", "Invalid api_key <redacted>"),
    ("Invalid api_key(short-secret)", "Invalid api_key<redacted>"),
    ("Invalid access token short-secret", "Invalid access token <redacted>"),
    ("Invalid private key\nshort-secret", "Invalid private key <redacted>"),
    ("The API key is expired.", "The API key is expired."),
    ("Invalid signature for the configured private key.", "Invalid signature for the configured private key."),
])
def test_reason_distinguishes_credential_echoes_from_status_words(message, sanitized):
    assert bounded_exchange_error_context_from_mapping({"message": message}) == {
        "error_reason": sanitized,
    }


@pytest.mark.parametrize(("message", "sanitized"), [
    ("Invalid passphrase 'abc'def'", "Invalid passphrase <redacted>"),
    ("Invalid passphrase 'abc' def'", "Invalid passphrase <redacted>"),
    ("Invalid passphrase 'abc'.def'", "Invalid passphrase <redacted>"),
    ('Rejected api_key="abc"def"', "Rejected api_key=<redacted>"),
    ('Malformed Bearer "abc"def"', "Malformed <redacted-auth>"),
    (r'Invalid passphrase "abc\"def", retry later', "Invalid passphrase <redacted>, retry later"),
    ("'tiny123' is not a valid passphrase", "<redacted> is not a valid passphrase"),
    ('"tiny123" is not the correct password', "<redacted> is not the correct password"),
    ("'tiny123' isn't a valid passphrase", "<redacted> isn't a valid passphrase"),
    ("'tiny123' is no longer a valid passphrase", "<redacted> is no longer a valid passphrase"),
    ("'tiny123' is an invalid or expired API key", "<redacted> is an invalid or expired API key"),
    ("'tiny123' has expired api key", "<redacted> has expired api key"),
    ("'tiny123' does not match any API key", "<redacted> does not match any API key"),
    ("‘tiny123’ is next to an expired API key", "<redacted> is next to an expired API key"),
    ("`tiny123` is not a valid passphrase", "<redacted> is not a valid passphrase"),
    ('"tiny123" wasn\'t a recognized API key', "<redacted> wasn't a recognized API key"),
    ("Rejected: 'abc'def' was not the correct private key",
     "Rejected: <redacted> was not the correct private key"),
    ("Rejected: 'abc' violet rabbit' was not the correct private key",
     "Rejected: <redacted> was not the correct private key"),
    ("'a' b 'c' is not a valid passphrase", "<redacted> is not a valid passphrase"),
    ("‘abc’ def’ is not a valid passphrase", "<redacted> is not a valid passphrase"),
    ("'tiny123': invalid API key", "<redacted> invalid API key"),
    ("'tiny123', invalid API key", "<redacted> invalid API key"),
    ("'tiny123' (invalid API key)", "<redacted> invalid API key)"),
    ("'tiny123'. invalid API key", "<redacted> invalid API key"),
    ("'tiny123' - invalid API key", "<redacted> invalid API key"),
    ("«tiny123» is not a valid passphrase", "<redacted> is not a valid passphrase"),
])
def test_credential_echoes_require_unambiguous_quotes(message, sanitized):
    assert bounded_exchange_error_context_from_mapping({"message": message}) == {
        "error_reason": sanitized,
    }


@pytest.mark.parametrize("message", [
    "Invalid passphrase=\"first' violet rabbit API key\"",
    "Invalid passphrase \"first' violet rabbit API key\"",
    "Bearer \"first' violet rabbit API key\"",
    "'first api_key=second' is not a valid passphrase",
    "Invalid 'api_key' violet rabbit passphrase",
    "'short-secret' is an invalid private key",
    "'abc' my token' is not a valid passphrase",
    "'abc' my token.' is not a valid passphrase",
    "User 'bob' sent 'hunter' which is invalid passphrase",
    "'abc' and hunter is not a valid passphrase",
    "Key 'main' hunter is not a valid passphrase",
    "'tiny{secret' is not a valid passphrase",
    "'tiny{secret' is not a valid passphrase {payload: canary}",
    "'tiny[{\"secret\"}]' is not a valid passphrase",
    "tiny123 is an invalid passphrase",
    "tiny123 is not a valid passphrase",
    "tiny123 has expired API key",
    "tiny123 does not match any API key",
    "tiny123 is an invalid passphrase; 'other' is an invalid API key",
])
def test_credential_echoes_with_ambiguous_boundaries_drop_reason(message):
    assert bounded_exchange_error_context_from_mapping({"message": message}) == {
        "error_reason": "<redacted>",
    }


@pytest.mark.parametrize("label", ["pass", "pwd", "auth", "creds", "sig", "hmac", "jwt", "otp"])
def test_short_authentication_field_aliases_are_redacted(label):
    assert bounded_exchange_error_context_from_mapping({"message": f"Rejected {label}=tiny123"}) == {
        "error_reason": f"Rejected {label}=<redacted>",
    }


def test_long_noncredential_hyphenated_reason_uses_bounded_prefix_scan():
    # Fill the parser's input budget without matching a credential label.
    assert bounded_exchange_error_context_from_mapping({"message": "a-" * 4096}) == {
        "error_reason": "<redacted>",
    }


def test_bounded_exchange_error_context_redacts_urls_and_email_addresses():
    error = RuntimeError(
        'gate {"message":"contact trader@example.com or visit '
        'https://example.invalid/account?id=short"}'
    )

    assert bounded_exchange_error_context(error) == {
        "error_reason": (
            "contact <redacted-email> or visit <redacted-url>"
        )
    }


def test_bounded_exception_type_uses_trusted_mro_classification():
    opaque_error = type("sk_live_7E4v93kR2mN6pQ8t", (RuntimeError,), {})

    assert bounded_exception_type(opaque_error("api_key=hidden")) == "RuntimeError"
    assert bounded_exception_type(RuntimeError("api_key=hidden")) == "RuntimeError"


def test_bounded_exception_type_rejects_forged_trusted_module():
    forged_error = type(
        "sk_live_7E4v93kR2mN6pQ8t",
        (RuntimeError,),
        {"__module__": "ccxt"},
    )

    assert bounded_exception_type(forged_error("api_key=hidden")) == "RuntimeError"


def test_exception_text_contains_catches_hostile_string_conversion():
    secret = "api_key=hostile-string-secret"

    class HostileError(RuntimeError):
        def __str__(self):
            raise KeyboardInterrupt(secret)

    assert exception_text_contains(HostileError(), ("recvwindow",)) is False


def test_exception_text_contains_scans_late_markers_across_bounded_chunks():
    error = RuntimeError(("x" * 5000) + " TOO_MANY_REQUESTS")

    assert exception_text_contains(error, ("too_many_requests",)) is True


def test_exception_text_contains_preserves_case_sensitive_matching():
    upper = RuntimeError(("x" * 5000) + " TOO_MANY_REQUESTS")
    lower = RuntimeError(("x" * 5000) + " too_many_requests")

    assert exception_text_contains(
        upper, ("TOO_MANY_REQUESTS",), case_sensitive=True
    )
    assert not exception_text_contains(
        lower, ("TOO_MANY_REQUESTS",), case_sensitive=True
    )


def test_exception_type_name_contains_is_hostile_metadata_safe_and_non_projecting():
    secret = "api_key=hostile-type-metadata"

    class HostileMeta(type):
        def __getattribute__(cls, name):
            if name == "__name__":
                raise KeyboardInterrupt(secret)
            return super().__getattribute__(name)

    class WrappedInvalidNonceFailure(RuntimeError, metaclass=HostileMeta):
        pass

    assert exception_type_name_contains(
        WrappedInvalidNonceFailure(), ("invalidnonce",)
    )
    assert not exception_type_name_contains(
        WrappedInvalidNonceFailure(), ("unrelated",)
    )


def test_exception_type_name_contains_rejects_metaclass_name_descriptor_forgery():
    class ForgedNameMeta(type):
        @property
        def __name__(cls):
            return "InvalidNonceForged"

    class UnrelatedFailure(RuntimeError, metaclass=ForgedNameMeta):
        pass

    assert not exception_type_name_contains(
        UnrelatedFailure(), ("invalidnonce",)
    )


def test_exception_type_name_contains_normalizes_stored_string_subclass():
    class HostileName(str):
        def lower(self):
            raise KeyboardInterrupt("api_key=hostile-name-subclass")

    class WrappedFailure(RuntimeError):
        pass

    WrappedFailure.__name__ = HostileName("WrappedInvalidNonceFailure")

    assert exception_type_name_contains(WrappedFailure(), ("invalidnonce",))


def test_bounded_exception_status_and_code_contain_hostile_metadata():
    secret = "api_key=attribute-secret"

    class HostileError(RuntimeError):
        @property
        def status(self):
            raise KeyboardInterrupt(secret)

        @property
        def code(self):
            raise SystemExit(secret)

        @property
        def info(self):
            raise GeneratorExit(secret)

    error = HostileError(secret)

    assert bounded_exception_status(error) is None
    assert bounded_exception_code(error) is None


def test_bounded_exception_status_and_code_contain_hostile_info_keys():
    secret = "api_key=info-key-secret"

    class HostileKey:
        def __hash__(self):
            return hash("status")

        def __eq__(self, other):
            raise KeyboardInterrupt(secret)

    error = RuntimeError(secret)
    error.info = {HostileKey(): "429"}

    assert bounded_exception_status(error) is None
    assert bounded_exception_code(error) is None


def test_bounded_exception_status_and_code_reject_oversized_integers():
    error = RuntimeError("api_key=integer-secret")
    error.status = 10**10_000
    error.code = -(10**10_000)

    assert bounded_exception_status(error) is None
    assert bounded_exception_code(error) is None

    error.status = "5" * 10_000
    error.code = "A" * 10_000

    assert bounded_exception_status(error) is None
    assert bounded_exception_code(error) is None
