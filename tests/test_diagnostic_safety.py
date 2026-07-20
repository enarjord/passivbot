from live.diagnostic_safety import (
    bounded_exception_code,
    bounded_exception_status,
)


def test_bounded_exception_status_and_code_keep_safe_direct_values():
    error = RuntimeError("api_key=hidden")
    error.status = "503"
    error.code = "TEMP_UNAVAILABLE"

    assert bounded_exception_status(error) == "503"
    assert bounded_exception_code(error) == "TEMP_UNAVAILABLE"


def test_bounded_exception_status_and_code_use_safe_info_fallbacks():
    error = RuntimeError("api_key=hidden")
    error.status = "500?api_key=hidden"
    error.code = "ApiKeyProdSecret"
    error.info = {"status": "429", "retCode": "RATE_LIMIT"}

    assert bounded_exception_status(error) == "429"
    assert bounded_exception_code(error) == "RATE_LIMIT"


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
