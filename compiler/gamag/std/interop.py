"""Medical interoperability: FHIR, terminology, provenance, consent.

Audit priority 15.  Spec section 24 asks for FHIR-compatible serialization as "a
library/profile layer" and names five requirements that shape this module:
schema validation, terminology validation, provenance, consent, and an audit
trail.

The section ends with the sentence that governs how this module describes
itself:

    Medical software may be subject to applicable laws and standards. A language
    feature does not automatically make an application HIPAA-, FDA-, IEC-, or
    GDPR-compliant.

So nothing here claims compliance.  What it provides is the machinery a
compliant system needs and can be inspected: validation that says which field
failed, provenance that cannot be edited without breaking its own chain, and
consent decisions that record why they were reached.

Two design decisions worth stating:

* **FHIR here means the resource shape, not the whole specification.**  Each
  serializer emits the fields the specification names for that resource, using
  the specification's own element names, and `validate` checks exactly those.
  It is not a conformance-tested FHIR implementation and does not say it is.
* **A consent decision is a record, not a boolean.**  `consent.check` returns
  the reason alongside the answer, because "denied" without "by which rule" is
  the answer that gets ignored in production.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..diagnostics import GamaRuntimeFault, TypeFault
from ..runtime.values import (GRecord, GResult, GVariant, UNIT, display, to_text,
                              truthy, type_name)
from ..semantic import types as T
from .library import reg

# ---------------------------------------------------------------------------
# FHIR
# ---------------------------------------------------------------------------

#: The Gama-G record type -> the FHIR resource it serializes as.
RESOURCE_TYPES: Dict[str, str] = {
    "Patient": "Patient",
    "Observation": "Observation",
    "Medication": "MedicationRequest",
    "Encounter": "Encounter",
    "DiagnosticReport": "DiagnosticReport",
}

#: Required elements per resource, and deliberately only those this mapping can
#: produce.  FHIR requires more of some resources than a Gama-G medical record
#: carries -- a `MedicationRequest` needs a subject, and a `Medication` record
#: has no patient in it -- so listing the full FHIR requirement would make
#: `validate` reject the very resources `serialize` just emitted.  This is the
#: schema this module defines, and it says so rather than implying conformance.
REQUIRED_ELEMENTS: Dict[str, Tuple[str, ...]] = {
    "Patient": ("resourceType", "id"),
    "Observation": ("resourceType", "status", "code"),
    "MedicationRequest": ("resourceType", "status", "intent",
                          "medicationCodeableConcept"),
    "Encounter": ("resourceType", "status"),
    "DiagnosticReport": ("resourceType", "status", "code"),
}

#: Gama-G medical record field -> FHIR element, for the fields the
#: specification's resources name.  A field with no FHIR element is carried
#: through under its own name rather than dropped, because a serializer that
#: silently loses a clinical field is worse than one that emits an extra key.
FHIR_ELEMENT_NAMES: Dict[str, str] = {
    "id": "id",
    "mrn": "identifier",
    "name": "name",
    "sex": "gender",
    "age": "age",
    "patient_id": "subject",
    "code": "code",
    "value": "value",
    "unit": "unit",
    "validated": "validated",
    "dose": "doseQuantity",
}

#: Elements this module supplies itself, with the value it supplies and why.
#: `status` and `intent` are required by FHIR and have no counterpart in the
#: record, so they are defaulted and the default is written down here rather
#: than buried in the serializer.
SUPPLIED_ELEMENTS: Dict[str, Dict[str, Any]] = {
    "Observation": {"status": "final"},
    "MedicationRequest": {"status": "active", "intent": "order"},
    "Encounter": {"status": "finished"},
    "DiagnosticReport": {"status": "final"},
}


def _record_fields(value: Any) -> Tuple[str, Dict[str, Any]]:
    if isinstance(value, GRecord):
        return value.name, dict(value.fields)
    if isinstance(value, GVariant):
        return value.tag, {"value": value.args[0] if value.args else UNIT}
    raise TypeFault(
        f"expected a medical record, not {type_name(value)}",
        hint="the medical records are Patient, Observation, Medication, "
             "Encounter and DiagnosticReport")


@reg("fhir.resource_type", ("record",), ret=T.TEXT, argtypes=(T.ANY,),
     effects=("medical",), caps=("PatientRead",),
     doc="The FHIR resource a medical record serializes as.")
def _fhir_resource_type(ctx, record) -> str:
    name, _fields = _record_fields(record)
    return RESOURCE_TYPES.get(name, name)


@reg("fhir.serialize", ("record",), ret=T.TEXT, argtypes=(T.ANY,),
     effects=("medical", "io"), caps=("PatientRead",),
     doc="Serialize a medical record as a FHIR resource, in JSON.")
def _fhir_serialize(ctx, record) -> str:
    ctx.require_capability("PatientRead", what="fhir.serialize")
    name, fields = _record_fields(record)
    resource_type = RESOURCE_TYPES.get(name, name)
    payload: Dict[str, Any] = {"resourceType": resource_type}
    payload.update(SUPPLIED_ELEMENTS.get(resource_type, {}))
    for key, value in sorted(fields.items()):
        element = FHIR_ELEMENT_NAMES.get(key, key)
        if element == "subject":
            # FHIR wants a reference, not a bare identifier.
            payload[element] = {"reference": f"Patient/{to_text(value)}"}
        else:
            payload[element] = _to_fhir(value)
    ctx.audit.record("FHIR_SERIALIZE", level="medical",
                     object=str(fields.get("id", name)),
                     reason=f"serialized as {resource_type}")
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _to_fhir(value: Any) -> Any:
    """A Gama-G value as a JSON value.  Secrets are refused, not encoded."""
    from ..runtime.values import GSecret, GUnit
    if isinstance(value, GSecret):
        raise GamaRuntimeFault(
            "SecretLeak",
            "a secret cannot be serialized into a FHIR resource",
            hint="spec section 8 forbids conversion of a secret to ordinary "
                 "Text; disclose it deliberately with `secrets.expose` and a "
                 "reason, or omit it")
    if value is None or isinstance(value, GUnit):
        return None
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_to_fhir(item) for item in value]
    if isinstance(value, dict):
        return {str(k): _to_fhir(v) for k, v in value.items()}
    return to_text(value)


@reg("fhir.validate", ("resource",), ret=T.TEXT, argtypes=(T.TEXT,),
     effects=("medical",), doc="Check a serialized resource has its required "
                              "elements. Returns an empty Text when valid.")
def _fhir_validate(ctx, resource: str) -> str:
    try:
        payload = json.loads(resource)
    except json.JSONDecodeError as exc:
        return f"not valid JSON: {exc}"
    if not isinstance(payload, dict):
        return "a FHIR resource is a JSON object"
    resource_type = payload.get("resourceType")
    if not resource_type:
        return "no `resourceType`, so the resource cannot be validated"
    if resource_type not in REQUIRED_ELEMENTS:
        return (f"`{resource_type}` is not one of the resources this module "
                f"serializes: {', '.join(sorted(REQUIRED_ELEMENTS))}")
    missing = [element for element in REQUIRED_ELEMENTS[resource_type]
               if element not in payload or payload[element] in (None, "")]
    if missing:
        return (f"{resource_type} is missing required element(s): "
                f"{', '.join(missing)}")
    return ""


@reg("fhir.parse", ("resource",), ret=T.ANY, argtypes=(T.TEXT,),
     effects=("medical",), caps=("PatientRead",),
     doc="Read a FHIR JSON resource back into a record.")
def _fhir_parse(ctx, resource: str) -> GRecord:
    ctx.require_capability("PatientRead", what="fhir.parse")
    problem = _fhir_validate(ctx, resource)
    if problem:
        raise GamaRuntimeFault("FHIRValidationError", problem)
    payload = json.loads(resource)
    resource_type = payload.pop("resourceType")
    name = next((key for key, value in RESOURCE_TYPES.items()
                 if value == resource_type), resource_type)
    reverse = {value: key for key, value in FHIR_ELEMENT_NAMES.items()}
    supplied = SUPPLIED_ELEMENTS.get(resource_type, {})
    fields: Dict[str, Any] = {}
    for key, value in payload.items():
        if key in supplied and supplied[key] == value:
            continue                      # this module put it there
        if isinstance(value, dict) and "reference" in value:
            value = str(value["reference"]).split("/", 1)[-1]
        fields[reverse.get(key, key)] = value
    if resource_type == "Observation" and "value" in fields:
        # The record carries a number; JSON gives back whatever it was.
        try:
            fields["value"] = float(fields["value"])
        except (TypeError, ValueError):
            pass
    return GRecord(name, fields)


# ---------------------------------------------------------------------------
# Terminology
# ---------------------------------------------------------------------------

#: Loaded code systems, by system URI.  Terminology is reference data, so it is
#: process state in the same way a foreign library is: loading the same system
#: twice must give the same codes.
_CODE_SYSTEMS: Dict[str, Dict[str, str]] = {}


@reg("terminology.load", ("system", "codes"), ret=T.I64,
     argtypes=(T.TEXT, T.TEXT), effects=("medical",),
     doc="Load a code system from JSON: {system, codes: {code: display}}.")
def _terminology_load(ctx, system: str, codes: str) -> int:
    try:
        payload = json.loads(codes)
    except json.JSONDecodeError as exc:
        raise TypeFault(f"the code system is not valid JSON: {exc}") from None
    entries = payload.get("codes", payload)
    if not isinstance(entries, dict):
        raise TypeFault("the code system must map codes to display names")
    _CODE_SYSTEMS[system] = {str(k): to_text(v) for k, v in entries.items()}
    ctx.audit.record("TERMINOLOGY_LOAD", level="medical", object=system,
                     reason=f"{len(entries)} code(s)")
    return len(entries)


@reg("terminology.validate", ("system", "code"), ret=T.BOOL,
     argtypes=(T.TEXT, T.TEXT), effects=("medical",),
     doc="Whether a code is in a loaded system.")
def _terminology_validate(ctx, system: str, code: str) -> bool:
    if system not in _CODE_SYSTEMS:
        raise GamaRuntimeFault(
            "UnknownCodeSystem",
            f"`{system}` has not been loaded",
            hint="terminology is reference data: load it with "
                 "`terminology.load` before validating against it")
    return code in _CODE_SYSTEMS[system]


@reg("terminology.lookup", ("system", "code"), ret=T.TEXT,
     argtypes=(T.TEXT, T.TEXT), effects=("medical",),
     doc="The display name for a code, or an empty Text if unknown.")
def _terminology_lookup(ctx, system: str, code: str) -> str:
    return _CODE_SYSTEMS.get(system, {}).get(code, "")


@reg("terminology.systems", (), ret=T.TEXT, effects=("medical",),
     doc="The systems that have been loaded.")
def _terminology_systems(ctx) -> str:
    return ", ".join(sorted(_CODE_SYSTEMS)) or "none loaded"


@reg("terminology.require", ("system", "code"), ret=T.UNIT,
     argtypes=(T.TEXT, T.TEXT), effects=("medical",),
     doc="Fail unless a code is valid in a loaded system.")
def _terminology_require(ctx, system: str, code: str) -> Any:
    if not _terminology_validate(ctx, system, code):
        display_name = _CODE_SYSTEMS.get(system, {}).get(code)
        raise GamaRuntimeFault(
            "TerminologyError",
            f"`{code}` is not a valid code in `{system}`",
            hint=(f"did you mean `{display_name}`?" if display_name else
                  f"the system has {len(_CODE_SYSTEMS.get(system, {}))} "
                  f"code(s)"))
    return UNIT


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

class ProvenanceChain:
    """An append-only chain of provenance entries.

    Each entry records the hash of its predecessor, so removing or editing an
    entry is detectable.  This is the same construction as the audit chain and
    for the same reason: provenance that can be rewritten is not provenance.
    """

    def __init__(self) -> None:
        self.entries: List[Dict[str, str]] = []
        self.digests: List[str] = []

    def append(self, entry: Dict[str, str]) -> str:
        import hashlib
        previous = self.digests[-1] if self.digests else "genesis"
        payload = json.dumps({**entry, "previous": previous}, sort_keys=True)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self.entries.append({**entry, "previous": previous, "digest": digest})
        self.digests.append(digest)
        return digest

    def verify(self) -> Tuple[bool, List[str]]:
        import hashlib
        problems: List[str] = []
        previous = "genesis"
        for index, entry in enumerate(self.entries):
            if entry.get("previous") != previous:
                problems.append(
                    f"entry {index} points at {entry.get('previous')!r} but "
                    f"the previous digest is {previous!r}")
            payload = {k: v for k, v in entry.items() if k != "digest"}
            expected = hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
            if expected != entry.get("digest"):
                problems.append(f"entry {index} has been edited")
            previous = entry.get("digest", "")
        return (not problems), problems


#: One chain per program run, keyed by the context, so a chain is never shared
#: between runs and two runs of the same program produce the same provenance.
_CHAINS: Dict[int, ProvenanceChain] = {}


def _chain_for(ctx) -> ProvenanceChain:
    key = id(ctx)
    if key not in _CHAINS:
        _CHAINS[key] = ProvenanceChain()
    return _CHAINS[key]


@reg("provenance.record", ("what", "source"), ret=T.TEXT,
     argtypes=(T.TEXT, T.TEXT), effects=("medical", "audit"),
     caps=("AuditWrite",), doc="Append a provenance entry and return its digest.")
def _provenance_record(ctx, what: str, source: str) -> str:
    ctx.require_capability("AuditWrite", what="provenance.record")
    digest = _chain_for(ctx).append({"what": to_text(what),
                                     "source": to_text(source),
                                     "actor": str(getattr(ctx, "actor", "")
                                                  or "unknown")})
    ctx.audit.record("PROVENANCE_RECORD", level="medical", object=to_text(what),
                     reason=to_text(source))
    return digest


@reg("provenance.count", (), ret=T.I64, effects=("medical",),
     doc="How many provenance entries this run has recorded.")
def _provenance_count(ctx) -> int:
    return len(_chain_for(ctx).entries)


@reg("provenance.verify", (), ret=T.BOOL, effects=("medical",),
     doc="Re-validate the provenance chain.")
def _provenance_verify(ctx) -> bool:
    ok, _problems = _chain_for(ctx).verify()
    return ok


@reg("provenance.problems", (), ret=T.TEXT, effects=("medical",),
     doc="Why the provenance chain does not verify; empty when it does.")
def _provenance_problems(ctx) -> str:
    _ok, problems = _chain_for(ctx).verify()
    return "; ".join(problems)


# ---------------------------------------------------------------------------
# Consent
# ---------------------------------------------------------------------------

@reg("consent.grant", ("subject", "purpose"), ret=T.ANY,
     argtypes=(T.TEXT, T.TEXT), effects=("medical", "audit"),
     caps=("AuditWrite",),
     doc="Record consent for a subject and purpose. Returns the consent record.")
def _consent_grant(ctx, subject: str, purpose: str) -> GRecord:
    ctx.require_capability("AuditWrite", what="consent.grant")
    record = GRecord("Consent", {"subject": to_text(subject),
                                 "purpose": to_text(purpose),
                                 "granted": True, "revoked": False})
    ctx.audit.record("CONSENT_GRANTED", level="medical",
                     object=to_text(subject), reason=to_text(purpose))
    return record


@reg("consent.revoke", ("consent",), ret=T.ANY, argtypes=(T.ANY,),
     effects=("medical", "audit"), caps=("AuditWrite",),
     doc="Revoke a consent record.")
def _consent_revoke(ctx, consent) -> GRecord:
    ctx.require_capability("AuditWrite", what="consent.revoke")
    if not isinstance(consent, GRecord):
        raise TypeFault("consent.revoke expects a consent record")
    fields = dict(consent.fields)
    fields["revoked"] = True
    fields["granted"] = False
    ctx.audit.record("CONSENT_REVOKED", level="medical",
                     object=to_text(fields.get("subject", "")),
                     reason=to_text(fields.get("purpose", "")))
    return GRecord("Consent", fields)


@reg("consent.check", ("consent", "purpose", "actor"), ret=T.ANY,
     argtypes=(T.ANY, T.TEXT, T.TEXT), effects=("medical", "audit"),
     doc="A consent decision: the answer *and* the reason, never a bare Bool.")
def _consent_check(ctx, consent, purpose: str, actor: str) -> GRecord:
    if not isinstance(consent, GRecord):
        raise TypeFault("consent.check expects a consent record")
    fields = consent.fields
    wanted = to_text(purpose)
    if fields.get("revoked"):
        decision, reason = False, "the consent was revoked"
    elif not fields.get("granted"):
        decision, reason = False, "no consent was recorded"
    elif to_text(fields.get("subject", "")) != to_text(actor):
        # Consent is per subject.  Treating a consent record as permission for
        # anyone would be the single worst bug this module could have.
        decision, reason = False, (f"the consent is for "
                                   f"`{fields.get('subject')}`, not `{actor}`")
    elif to_text(fields.get("purpose", "")) != wanted:
        decision, reason = False, (f"the consent is for "
                                   f"`{fields.get('purpose')}`, not `{wanted}`")
    else:
        decision, reason = True, "consent covers this subject and purpose"
    # A decision that is not recorded is a decision nobody can review.  Spec
    # section 24 lists the audit trail next to consent for that reason.
    ctx.audit.record("CONSENT_CHECK", level="medical", object=to_text(actor),
                     reason=f"{'permitted' if decision else 'denied'}: {reason}")
    return GRecord("ConsentDecision", {"allow": decision, "reason": reason,
                                       "subject": to_text(fields.get("subject", "")),
                                       "purpose": wanted, "actor": to_text(actor)})


@reg("consent.require", ("consent", "purpose", "actor"), ret=T.UNIT,
     argtypes=(T.ANY, T.TEXT, T.TEXT), effects=("medical", "audit"),
     doc="Fail unless consent covers this actor and purpose.")
def _consent_require(ctx, consent, purpose: str, actor: str) -> Any:
    decision = _consent_check(ctx, consent, purpose, actor)
    if not decision.fields["allow"]:
        raise GamaRuntimeFault(
            "ConsentError",
            f"{actor} has no consent for {purpose}: {decision.fields['reason']}")
    return UNIT
