from live.diagnostic_safety import (
    bounded_exception_code,
    bounded_exception_status,
    bounded_exception_type,
    bounded_exchange_error_context,
    bounded_exchange_error_context_from_mapping,
    exception_text_contains,
    exception_type_name_contains,
    sanitize_diagnostic_text,
    sanitized_exception_message,
)


def test_sanitize_diagnostic_text_preserves_non_secret_failure_context():
    raw = (
        "order rejected symbol=BTC/USDT:USDT order_id=abc123 price=101.25 qty=0.4 "
        "wallet_address=0xabc url=https://example.invalid/orders/abc123?attempt=2 "
        "planning_signature=plan-hash request_signature=request-secret "
        "api_key=key-secret Authorization: Bearer bearer-secret "
        'private_key="private-secret"'
    )

    sanitized = sanitize_diagnostic_text(raw)

    for useful in (
        "symbol=BTC/USDT:USDT",
        "order_id=abc123",
        "price=101.25",
        "qty=0.4",
        "wallet_address=0xabc",
        "planning_signature=plan-hash",
        "https://example.invalid/orders/abc123?attempt=2",
    ):
        assert useful in sanitized
    for secret in (
        "request-secret",
        "key-secret",
        "bearer-secret",
        "private-secret",
    ):
        assert secret not in sanitized
    assert sanitized.count("[redacted]") == 4


def test_sanitize_diagnostic_text_redacts_cli_userinfo_and_private_key_block():
    raw = (
        "POST https://alice:password@example.invalid/private --api-key cli-secret "
        "-----BEGIN PRIVATE KEY-----\nprivate-key-body\n-----END PRIVATE KEY----- "
        "exchange_reason=price step mismatch"
    )

    sanitized = sanitize_diagnostic_text(raw)

    assert "alice" not in sanitized
    assert "password" not in sanitized
    assert "cli-secret" not in sanitized
    assert "private-key-body" not in sanitized
    assert "https://[redacted]@example.invalid/private" in sanitized
    assert "exchange_reason=price step mismatch" in sanitized


def test_sanitize_diagnostic_text_redacts_unterminated_private_key_block():
    sanitized = sanitize_diagnostic_text(
        "parser failed exchange_reason=invalid key material "
        "-----BEGIN RSA PRIVATE KEY-----\nprivate-key-body\npartial-tail"
    )

    assert "parser failed exchange_reason=invalid key material" in sanitized
    assert "BEGIN RSA PRIVATE KEY" not in sanitized
    assert "private-key-body" not in sanitized
    assert "partial-tail" not in sanitized
    assert sanitized.endswith("[redacted]")


def test_sanitize_diagnostic_text_redacts_camel_case_credentials():
    sanitized = sanitize_diagnostic_text(
        'accessToken="access-value" clientSecret=client-value '
        "requestSignature=request-value planning_signature=plan-hash"
    )

    assert "access-value" not in sanitized
    assert "client-value" not in sanitized
    assert "request-value" not in sanitized
    assert "planning_signature=plan-hash" in sanitized


def test_sanitized_exception_message_redacts_serialized_authentication_headers():
    error = RuntimeError(
        'request failed headers={"Cookie": "sessionid=TOPSECRET; csrf=SECOND", '
        '"Authorization": "Bearer AUTHSECRET", "sign": "SIGNSECRET", '
        '"X-Trace": "trace-123"} '
        "fallback={'Proxy-Authorization': 'Basic PROXYSECRET', "
        "'Set-Cookie': 'session=COOKIESECRET; Path=/'} "
        "url=https://example.invalid/orders/abc123"
    )

    sanitized = sanitized_exception_message(error)

    for secret in (
        "TOPSECRET",
        "SECOND",
        "AUTHSECRET",
        "PROXYSECRET",
        "COOKIESECRET",
        "SIGNSECRET",
    ):
        assert secret not in sanitized
    assert '"Cookie": "[redacted]"' in sanitized
    assert '"Authorization": "[redacted]"' in sanitized
    assert "'Proxy-Authorization': '[redacted]'" in sanitized
    assert "'Set-Cookie': '[redacted]'" in sanitized
    assert '"sign": "[redacted]"' in sanitized
    assert '"X-Trace": "trace-123"' in sanitized
    assert "url=https://example.invalid/orders/abc123" in sanitized


def test_sanitized_exception_message_redacts_exchange_prefixed_authentication_headers():
    sensitive_headers = {
        "APCA-API-KEY-ID": "ALPACAKEY",
        "APCA-API-SECRET-KEY": "ALPACASECRET",
        "KC-API-KEY": "KUCOINKEY",
        "KC-API-PASSPHRASE": "KUCOINPASS",
        "KC-API-SIGN": "KUCOINSIGN",
        "KC-API-PARTNER-SIGN": "KUCOINPARTNERSIGN",
        "X-BAPI-API-KEY": "BYBITKEY",
        "X-BAPI-SIGN": "BYBITSIGN",
        "OK-ACCESS-KEY": "OKXKEY",
        "OK-ACCESS-PASSPHRASE": "OKXPASS",
        "OK-ACCESS-SIGN": "OKXSIGN",
        "ACCESS-KEY": "ACCESSKEY",
        "ACCESS-PASSPHRASE": "ACCESSPASS",
        "ACCESS-SIGN": "ACCESSSIGN",
        "X-MBX-APIKEY": "BINANCEKEY",
        "api-key": "BITUNIXKEY",
        "KEY": "GATEKEY",
        "SIGN": "GATESIGN",
    }
    safe_headers = {
        "KC-API-TIMESTAMP": "kucoin-ts",
        "KC-API-PARTNER": "passivbotFutures",
        "KC-API-PARTNER-VERIFY": "true",
        "X-BAPI-TIMESTAMP": "bybit-ts",
        "OK-ACCESS-TIMESTAMP": "okx-ts",
        "ACCESS-TIMESTAMP": "access-ts",
        "X-Gate-Channel-Id": "broker-code",
        "X-Trace": "trace-123",
        "X-Trace-Sign": "diagnostic-sign",
        "X-Diagnostic-Signature": "diagnostic-signature",
    }
    error = RuntimeError(
        f"request failed headers={sensitive_headers | safe_headers}"
    )

    sanitized = sanitized_exception_message(error)

    for value in sensitive_headers.values():
        assert value not in sanitized
    for header, value in safe_headers.items():
        assert f"'{header}': '{value}'" in sanitized


def test_sanitize_diagnostic_text_redacts_kucoin_broker_signing_key():
    sanitized = sanitize_diagnostic_text(
        "config={'broker-key': 'hyphen-value', 'broker_key': 'snake-value', "
        "'brokerKey': 'camel-value', 'broker-name': 'passivbotFutures'}"
    )

    for secret in ("hyphen-value", "snake-value", "camel-value"):
        assert secret not in sanitized
    assert "'broker-name': 'passivbotFutures'" in sanitized


def test_sanitized_exception_message_contains_hostile_string_conversion():
    class HostileError(RuntimeError):
        def __str__(self):
            raise KeyboardInterrupt("api_key=must-not-escape")

    assert sanitized_exception_message(HostileError()) == "<unavailable>"


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
        "error_reason": "bad client id abcdefghijklmnopqrstuvwxyz012345",
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


def test_bounded_exchange_error_context_preserves_identifiers_and_redacts_secrets():
    error = RuntimeError(
        'gate {"label":"INVALID_PARAM_VALUE",'
        '"message":"bad client id abcdefghijklmnopqrstuvwxyz012345"}'
    )
    sensitive = RuntimeError(
        'gate {"label":"INVALID_PARAM_VALUE","message":"api_key=do-not-log"}'
    )

    assert bounded_exchange_error_context(error) == {
        "error_label": "INVALID_PARAM_VALUE",
        "error_reason": "bad client id abcdefghijklmnopqrstuvwxyz012345",
    }
    assert bounded_exchange_error_context(sensitive) == {
        "error_label": "INVALID_PARAM_VALUE",
        "error_reason": "api_key=[redacted]",
    }


def test_bounded_exchange_error_context_preserves_urls_and_email_addresses():
    error = RuntimeError(
        'gate {"message":"contact trader@example.com or visit '
        'https://example.invalid/account?id=short"}'
    )

    assert bounded_exchange_error_context(error) == {
        "error_reason": (
            "contact trader@example.com or visit "
            "https://example.invalid/account?id=short"
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
