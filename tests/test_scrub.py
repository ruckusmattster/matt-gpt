import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("scrub_data", Path(__file__).parents[1] / "scripts" / "scrub_data.py")
scrub_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scrub_data)


def run(text, redactions=()):
    return scrub_data.scrub(text, list(redactions))[0]


def test_email_phone_postcode():
    out = run("mail me at someone@example.com or 07123 456 789, I'm at SW1A 1AA")
    assert "example.com" not in out and "07123" not in out and "SW1A" not in out
    assert "[email]" in out and "[phone]" in out and "[postcode]" in out


def test_discord_ids():
    out = run("hey <@190711020006670336> <@&123456789012345678> <#123456789012345678> <:waow:1017853838516035725>")
    assert out == "hey @user @role #channel :waow:"


def test_redact_file_drops_exact_lines_and_substrings():
    out = run("is the password\nhunter2\nmy pw is hunter2 lol", ["hunter2"])
    assert out == "is the password\nmy pw is [redacted] lol"


def test_non_postcodes_survive():
    out = run("got an rtx 3060 and an sd1 5xx and the i7 3rd gen")
    assert "[postcode]" not in out
