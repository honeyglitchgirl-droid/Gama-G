# Releases, versioning, and what a version number here promises

This is the policy the release workflow implements.  It exists because
"released" has to mean something checkable.  The check is `ggc conform`, which
reads `Gama-G_v1.0_Production_Specification.txt` and reports what this toolchain
does and does not evidence; every claim in it is either satisfied or written
down.

---

## 1. What a release is

A release is a git tag `v<version>` whose `VERSION` file contains `<version>`,
plus the files the release workflow attaches:

| file | what it is |
|---|---|
| `gama_g-<version>-py3-none-any.whl` | the toolchain, installable |
| `gama_g-<version>.tar.gz` | the source distribution |
| `build-manifest.json` | a build manifest, verified reproducible over 2 builds (signed when the repository has a signing key) |
| `SHA256SUMS` | checksums for everything above |
| `conformance.json` | the conformance run for that tag: claims, deviations, unmet requirements |
| `strict.txt` | `ggc conform --strict`, the distance to the production bar |

The workflow refuses to publish if the tag and `VERSION` disagree, if the test
suite fails, or if a conformance claim fails.  It does **not** refuse to publish
because of the contents of `strict.txt`: that list is attached and reproduced in
the release notes, which is the honest alternative to suppressing it.  A release
that quietly omitted what it does not evidence would be exactly the over-claim
specification section 43 forbids.

### Signing

The manifest is signed with Ed25519 when the repository has a
`RELEASE_SIGNING_KEY` secret, and the release notes say which of the two
happened by *reading* the manifest rather than assuming.  There is deliberately
no fallback key: a manifest signed by a key that exists only inside a build
runner attributes the build to nobody while looking like provenance, which is
worse than an unsigned manifest that says so.

To enable signing:

```sh
./tools/bin/ggc gpm keygen --key /tmp/release.key
gh secret set RELEASE_SIGNING_KEY < /tmp/release.key
rm /tmp/release.key        # not printed, not recoverable, not in the repo
```

The public key is what a verifier needs, and it belongs in the release notes and
in the documentation -- the notes publish it automatically once the key is in
place.  Until then, every manifest says `signed: no`, and the notes say what
that does and does not prove.

## 2. Version numbers

`VERSION` is a semantic version, and it is stated in exactly one place that
matters: everything else (`pyproject.toml`, the package's `__version__`, the CLI
banner) is checked to agree with it by
`tests/test_enforcement.py::test_the_version_is_stated_once_and_agrees_everywhere`.

The rule for what changes the number:

* **major** — the specification version it implements changes (the language
  itself changes incompatibly), or a program that compiled stops compiling.
* **minor** — new capability: a new backend, a new module, a new command, a
  new optimization.  Nothing that worked stops working.
* **patch** — fixes only, no new capability.

`^1.2.0` currently means: GIR 1.0, specification 1.0.  The GIR and specification
versions are separate numbers reported by `ggc --version`, because the language
and the IR move independently of the toolchain.

## 3. Alpha, Beta, and the bar for leaving them

`pyproject.toml` says `Development Status :: 3 - Alpha`, and the rule for
changing that is the point of this section:

> **The classifier moves to Beta when `ggc conform --strict` exits 0** — that
> is, when every type name in specification section 5 is usable as written,
> when every compilation stage in section 22 is implemented rather than partial,
> and when all twenty of the v1.0 requirements in section 33 are evidenced
> rather than merely not-claimed.

And it moves to `5 - Production/Stable` only when the same command has exited 0
for a release and the independent review that section 33 item 10 asks for has
happened.  That item is currently `not-claimed`, so production/stable is not
reachable today, and no release of this toolchain may claim it.

As of 1.2.1, `--strict` lists 13 things: three recorded deviations and ten
requirements answered `partial` or `not-claimed`.  The release notes reproduce
that list, so a user can see the distance rather than infer it.

### What the first release exposed

`v1.2.0` was published by the first version of this workflow, and it is kept
rather than rewritten.  It has four assets, because the publish step uploaded
only the built distributions: `conformance.json` and `strict.txt` were left in
the run's artifacts, which expire -- while this document said they were part of
the release.  Two further defects were on the by-hand path only: dispatching the
workflow for an older tag checked the *branch* out rather than the tag, and the
tag-and-`VERSION` gate was skipped for anything that was not a tag push.

All three are fixed, and `tests/test_release.py` reads the workflow rather than
trusting it, because each of them was a claim in this document that the
workflow did not implement.  `v1.2.1` is the first release produced by the fixed
workflow; `v1.2.0` remains as it was published, and this paragraph is the record
of why it looks different.

## 4. What a version does *not* promise

* **Standard library stability is not promised** (section 33 item 11 is
  `not-claimed`).  Modules can change between minor versions.
* **The GIR is not promised stable across major versions**, only within one.
* **Deprecation is not practised yet**: there is no deprecation cycle, because
  there is no compatibility policy to deprecate against.  When the classifier
  moves to Beta, this section changes to say what the cycle is.
* **Binaries are not reproducible bit-for-bit.**  The *manifests* are
  reproducible and are checked to be; a rebuild on another machine with another
  C compiler is not claimed to produce an identical binary.  The digest covers
  the compiler's version, its own interpreter version and the platform, so the
  same source built by a different Python has a different digest -- the claim
  is that one build reproduces, not that different builds agree.  That is why
  the digest in the release notes and the digest of a local build of the same
  file do not match, and why neither is the file's hash: `program_sha256` is.

## 5. Cutting a release

```sh
# 1. the version, in one place
echo 1.3.0 > VERSION
sed -i 's/^version = "1.2.1"$/version = "1.3.0"/' pyproject.toml
sed -i 's/    return "1.2.1"/    return "1.3.0"/' compiler/gamag/__init__.py

# 2. everything that can be checked locally, checked locally
python3 -m unittest discover -s tests -t tests -q
./tools/bin/ggc conform
./tools/bin/ggc conform --strict || true      # read it, do not hide it
./tools/bin/ggc difftest examples/*.gg examples/core/*.gg

# 3. commit, tag, push the tag -- the workflow does the rest
git commit -am "Gama-G 1.3.0"
git push origin main
git tag -a v1.3.0 -m "Gama-G 1.3.0"
git push origin v1.3.0
```

If the strict list grew, that is not a failure of the release — it is new
information, and it belongs in the release notes and in this document's
section 3 count.

## 6. Why the conformance report is part of the release

A test suite that compares the toolchain with itself answers "did anything
change?"  It cannot answer "is this what the specification says?", because the
specification is not in the loop.  `ggc conform` puts it in the loop, and its
result is attached to every release so that the answer travels with the
artifact instead of living in a CI log that expires.

The suite is also the mechanism that stops a gap from being forgotten: a
deviation that is not written down in `conformance/deviations.json` is a
failure, and a section 33 requirement that disappears from
`conformance/requirements.json` is a failure.  The list of things this
toolchain does not do can only get longer if someone writes it down.
