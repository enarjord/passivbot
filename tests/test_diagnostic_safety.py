from live.diagnostic_safety import (
    bounded_exception_code,
    bounded_exception_status,
    bounded_exception_type,
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


def test_bounded_exception_type_uses_trusted_mro_classification():
    opaque_error = type("sk_live_7E4v93kR2mN6pQ8t", (RuntimeError,), {})

    assert bounded_exception_type(opaque_error("api_key=hidden")) == "RuntimeError"
    assert bounded_exception_type(RuntimeError("api_key=hidden")) == "RuntimeError"


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
