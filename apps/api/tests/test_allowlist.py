from app.services.source_allowlist import is_allowed_source


ALLOWLIST = ["w3.org", "github.com/w3c/process", "github.com/w3c/guide", "w3c.github.io"]


def test_w3c_domain_allowed() -> None:
    assert is_allowed_source("https://www.w3.org/policies/process/", ALLOWLIST)


def test_w3c_repo_allowed() -> None:
    assert is_allowed_source("https://github.com/w3c/process/blob/main/README.md", ALLOWLIST)


def test_untrusted_domain_rejected() -> None:
    assert not is_allowed_source("https://example.com/fake-process", ALLOWLIST)

