#!/usr/bin/env python3
"""Assemble the body of a release from what the conformance job produced.

The release notes are generated rather than written by hand because the two
things a reader of a Gama-G release most needs to know are both machine
outputs: what the conformance suite found, and what this version does *not*
fully evidence.  A hand-written summary can drift from the run it describes;
this cannot, because it is the run.

Usage:
    python3 tools/release_notes.py VERSION \\
        --conformance conformance.json \\
        --strict strict.txt \\
        --checksums dist/SHA256SUMS \\
        --manifest dist/build-manifest.json \\
        --sha "$GITHUB_SHA" \\
        -o release-notes.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys


def read(path: str) -> str:
    if not path or not os.path.isfile(path):
        return ""
    with open(path, encoding="utf-8") as handle:
        return handle.read().rstrip("\n")


def summarise(conformance_json: str) -> str:
    """One paragraph about the conformance run, or why there is not one."""
    text = read(conformance_json)
    if not text:
        return ("The conformance report is missing from this release, so this "
                "release makes no statement about conformance.")
    try:
        report = json.loads(text)
    except json.JSONDecodeError as exc:
        return f"The conformance report could not be read ({exc})."
    if not report.get("ok", False):
        return (f"**{len(report.get('failures', []))} conformance "
                f"claim(s) FAILED.** The claim list is in the attached "
                f"`conformance.json`.")
    return (f"{len(report.get('cases', []))} claims against specification "
            f"{report.get('spec_version', '?')}, no failures.  "
            f"{len(report.get('deviations', []))} recorded deviation(s); "
            f"{len(report.get('unmet_requirements', []))} of the twenty v1.0 "
            f"requirements in section 33 are not fully evidenced.")


def describe_manifest(path: str) -> str:
    """What the manifest proves, which depends on whether it was signed.

    "Signed" is not a decoration.  A manifest signed by a key nobody can name
    implies a provenance it does not have, so an unsigned one says so, and a
    signed one names the key that did it.
    """
    text = read(path)
    if not text:
        return ("No build manifest is attached to this release, so this "
                "release makes no reproducibility statement.")
    try:
        manifest = json.loads(text)
    except json.JSONDecodeError as exc:
        return f"The build manifest could not be read ({exc})."
    digest = manifest.get("digest", "?")
    if manifest.get("signed"):
        return (f"The build manifest is signed with the Ed25519 key "
                f"`{manifest.get('public_key', '?')}`.  It is reproducible: "
                f"two builds of the same source produce the same digest, "
                f"`{digest}`, and signature.  It proves the build is "
                f"reproducible and attributable to that key; it does not "
                f"attest to any property of the binary beyond its digest.")
    return (f"The build manifest digest is `{digest}`, verified reproducible "
            f"over two builds, and is **unsigned** -- there was no signing key "
            f"for this release, so it proves the build is reproducible and "
            f"nothing about who produced it.")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", help="the version being released")
    parser.add_argument("--conformance", default="conformance.json")
    parser.add_argument("--strict", default="strict.txt")
    parser.add_argument("--checksums", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--sha", default="")
    parser.add_argument("-o", "--out", default="release-notes.md")
    args = parser.parse_args(argv)

    lines = [f"Gama-G {args.version}", ""]
    if args.sha:
        lines += [f"Built from `{args.sha}`.", ""]
    lines += ["## Conformance", "", summarise(args.conformance), ""]

    strict = read(args.strict)
    if strict:
        lines += ["## What this version does not fully evidence", "",
                  "```", strict, "```", "",
                  "This list is why the package classifier still says Alpha: "
                  "leaving it requires `ggc conform --strict` to pass, which "
                  "is the bar described in `docs/RELEASES.md`.", ""]
    else:
        lines += ["## What this version does not fully evidence", "",
                  "The strict conformance log is missing from this release, "
                  "so the list is not reproduced here.  Run "
                  "`ggc conform --strict` against this version.", ""]

    if args.manifest:
        lines += ["## The build manifest", "", describe_manifest(args.manifest),
                  ""]

    checksums = read(args.checksums)
    if checksums:
        lines += ["## Files", "", "```", checksums, "```", ""]
    lines += [
        "## Checking this against the specification",
        "",
        "```",
        "git clone https://github.com/honeyglitchgirl-droid/Gama-G && cd Gama-G",
        "git checkout v" + args.version,
        "python3 -m unittest discover -s tests -t tests -q",
        "./tools/bin/ggc conform",
        "```",
        "",
    ]
    with open(args.out, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    sys.stdout.write("\n".join(lines) + "\n")
    return 0


if __name__ == "__main__":       # pragma: no cover
    sys.exit(main())
