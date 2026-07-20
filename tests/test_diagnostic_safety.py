from live.diagnostic_safety import (
    bounded_exception_code,
    bounded_exception_status,
    bounded_exception_type,
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
