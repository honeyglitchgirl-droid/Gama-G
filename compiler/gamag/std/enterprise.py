"""Enterprise modules: database, identity, workflow, messaging, observability.

Audit priority 16, and spec section 25's list of the modules an enterprise
deployment needs.  Each is implemented as far as this toolchain can honestly
take it, and the limits are written down rather than hidden behind a stub:

* **database** is real.  sqlite3 ships with Python, so `database.open` opens a
  file, `execute` runs a statement inside a transaction, and `query` returns
  rows.  Capability-gated by `DatabaseRead` and `DatabaseWrite`, because spec
  section 12 asks for exactly that.
* **identity** is real.  Principals, roles and authentication against an
  in-process credential store, with grants that must be authorised and are
  audited.  It is not an OIDC client, and does not say it is.
* **workflow** is real.  A definition is a list of steps; a run advances through
  them one at a time and records every transition, so the history is the
  explanation of the state.
* **messaging** is real but in-process: publish, subscribe, drain.  Publishing to
  a broker needs a network client, and spec section 12's `NetworkConnect`
  capability would have to be plumbed through; until it is, a broker is not
  claimed.
* **observability** is real.  Counters, gauges, structured log lines and spans,
  all with deterministic ids so two runs of the same program produce the same
  report -- which is the property spec section 1.3 leads with.

The in-process modules keep their state per *run*, never per process.  Two runs
of the same program in one process must be indistinguishable, and a module that
shared a queue between them would break that the first time someone embedded the
interpreter.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..diagnostics import GamaRuntimeFault, TypeFault
from ..runtime.values import (GOption, GRecord, UNIT, display, to_text, truthy,
                              type_name)
from ..semantic import types as T
from .library import reg

# ---------------------------------------------------------------------------
# Per-run state
#
# Keyed by the context object, so nothing leaks between runs.  A `WeakValueDict`
# would be tidier; a plain dict keyed by id() is used here because the entries
# are tiny and `_reset` exists for embedders that run millions of programs.
# ---------------------------------------------------------------------------

_STATE: Dict[int, Dict[str, Any]] = {}


def _state(ctx) -> Dict[str, Any]:
    key = id(ctx)
    if key not in _STATE:
        _STATE[key] = {
            "databases": {},       # path -> sqlite3 connection
            "principals": {},      # name -> {"roles": set, "credential": str}
            "workflows": {},       # definition name -> [steps]
            "workflow_runs": {},   # run id -> {"definition", "index", "history"}
            "topics": {},          # topic -> [messages]
            "subscriptions": {},   # topic -> count of subscribers
            "counters": {},        # name -> number
            "gauges": {},          # name -> number
            "logs": [],            # [(level, message, fields)]
            "spans": [],           # [(name, start, end, fields)]
            "sequence": 0,
        }
    return _STATE[key]


def reset() -> None:
    """Drop every module's per-run state.  For test harnesses and embedders."""
    _STATE.clear()


def _next_id(ctx) -> str:
    state = _state(ctx)
    state["sequence"] += 1
    # Derived from the sequence, not from the clock or a random source, so two
    # runs of the same program produce the same identifiers (spec section 1.3).
    digest = hashlib.sha256(f"{state['sequence']}".encode()).hexdigest()
    return digest[:16]


# ---------------------------------------------------------------------------
# database
# ---------------------------------------------------------------------------

def _database(ctx, handle) -> Tuple[str, sqlite3.Connection]:
    if not isinstance(handle, GRecord):
        raise TypeFault("expected a database handle from database.open")
    path = to_text(handle.fields.get("path", ""))
    connection = _state(ctx)["databases"].get(path)
    if connection is None:
        raise GamaRuntimeFault(
            "DatabaseError", f"the handle for {path!r} is not open",
            hint="it was closed, or this handle came from another run")
    return path, connection


@reg("database.open", ("path",), ret=T.ANY, argtypes=(T.TEXT,),
     effects=("storage",), caps=("DatabaseWrite",),
     doc="Open a SQLite database file. Requires DatabaseWrite.")
def _database_open(ctx, path: str) -> GRecord:
    ctx.require_capability("DatabaseWrite", what="database.open")
    resolved = to_text(path)
    if not resolved or resolved == ":memory:":
        resolved = ":memory:"
    connections = _state(ctx)["databases"]
    if resolved not in connections:
        try:
            connections[resolved] = sqlite3.connect(resolved)
        except sqlite3.Error as exc:
            raise GamaRuntimeFault(
                "DatabaseError", f"cannot open {resolved!r}: {exc}") from None
        ctx.audit.record("DATABASE_OPEN", level="security", object=resolved,
                         reason="the program opened a database")
    return GRecord("Database", {"path": resolved})


@reg("database.execute", ("handle", "sql"), ret=T.I64,
     argtypes=(T.ANY, T.TEXT), variadic=True, min_args=2,
     effects=("storage",), caps=("DatabaseWrite",),
     doc="Run one statement and return the number of affected rows.")
def _database_execute(ctx, handle, sql: str, *params) -> int:
    ctx.require_capability("DatabaseWrite", what="database.execute")
    path, connection = _database(ctx, handle)
    statement = to_text(sql)
    values = [_sql_value(p) for p in params]
    try:
        with connection:                   # commits on success, rolls back on error
            cursor = connection.execute(statement, values)
            affected = cursor.rowcount if cursor.rowcount >= 0 else 0
    except sqlite3.Error as exc:
        raise GamaRuntimeFault(
            "DatabaseError", f"{exc}", hint=f"statement: {statement[:120]}"
        ) from None
    ctx.audit.record("DATABASE_WRITE", level="security", object=path,
                     reason=statement[:120])
    return int(affected)


@reg("database.query", ("handle", "sql"), ret=T.ANY,
     argtypes=(T.ANY, T.TEXT), variadic=True, min_args=2,
     effects=("storage",), caps=("DatabaseRead",),
     doc="Run a query and return a List of Map rows.")
def _database_query(ctx, handle, sql: str, *params) -> List[Dict[str, Any]]:
    ctx.require_capability("DatabaseRead", what="database.query")
    _path, connection = _database(ctx, handle)
    statement = to_text(sql)
    values = [_sql_value(p) for p in params]
    try:
        connection.row_factory = sqlite3.Row
        cursor = connection.execute(statement, values)
        rows = [dict(row) for row in cursor.fetchall()]
    except sqlite3.Error as exc:
        raise GamaRuntimeFault(
            "DatabaseError", f"{exc}", hint=f"statement: {statement[:120]}"
        ) from None
    return rows


def _sql_value(value: Any) -> Any:
    """A Gama-G value as something sqlite3 accepts.

    A secret is refused rather than bound: writing one into a database by
    accident is exactly what spec section 8's serialization restriction is
    about, and a parameter binding is a serialization.
    """
    from ..runtime.values import GSecret
    if isinstance(value, GSecret):
        raise GamaRuntimeFault(
            "SecretLeak", "a secret cannot be bound into a SQL statement",
            hint="spec section 8 forbids converting a secret to ordinary data "
                 "without an audited `secrets.expose`")
    if isinstance(value, bool):
        return 1 if value else 0
    if isinstance(value, (int, float, str, bytes)) or value is None:
        return value
    return to_text(value)


@reg("database.tables", ("handle",), ret=T.ANY, argtypes=(T.ANY,),
     effects=("storage",), caps=("DatabaseRead",),
     doc="The table names in a database.")
def _database_tables(ctx, handle) -> List[str]:
    ctx.require_capability("DatabaseRead", what="database.tables")
    _path, connection = _database(ctx, handle)
    cursor = connection.execute(
        "SELECT name FROM sqlite_master WHERE type = 'table' ORDER BY name")
    return [row[0] for row in cursor.fetchall()]


@reg("database.close", ("handle",), ret=T.UNIT, argtypes=(T.ANY,),
     effects=("storage",), caps=("DatabaseWrite",), doc="Close a database.")
def _database_close(ctx, handle) -> Any:
    ctx.require_capability("DatabaseWrite", what="database.close")
    path, connection = _database(ctx, handle)
    try:
        connection.close()
    except sqlite3.Error:
        pass
    del _state(ctx)["databases"][path]
    ctx.audit.record("DATABASE_CLOSE", level="security", object=path,
                     reason="closed by the program")
    return UNIT


# ---------------------------------------------------------------------------
# identity
# ---------------------------------------------------------------------------

@reg("identity.register", ("principal", "credential"), ret=T.UNIT,
     argtypes=(T.TEXT, T.TEXT), effects=("audit",), caps=("AuditWrite",),
     doc="Register a principal with a credential.")
def _identity_register(ctx, principal: str, credential: str) -> Any:
    ctx.require_capability("AuditWrite", what="identity.register")
    name = to_text(principal)
    state = _state(ctx)["principals"]
    if name in state:
        raise GamaRuntimeFault("IdentityError",
                               f"`{name}` is already registered")
    state[name] = {"roles": set(), "credential": to_text(credential)}
    ctx.audit.record("IDENTITY_REGISTERED", level="security", object=name,
                     reason="a principal was registered")
    return UNIT


@reg("identity.authenticate", ("principal", "credential"), ret=T.BOOL,
     argtypes=(T.TEXT, T.TEXT), effects=("audit",),
     doc="Whether a credential is the one registered for a principal.")
def _identity_authenticate(ctx, principal: str, credential: str) -> bool:
    name = to_text(principal)
    record = _state(ctx)["principals"].get(name)
    # The comparison is constant-time.  A credential check that leaks its
    # answer through timing is a credential check that can be walked one byte at
    # a time, and `hmac.compare_digest` is the standard way not to.
    import hmac
    ok = record is not None and hmac.compare_digest(
        str(record["credential"]), to_text(credential))
    ctx.audit.record("IDENTITY_AUTHENTICATE", level="security", object=name,
                     reason="accepted" if ok else "rejected")
    return ok


@reg("identity.grant_role", ("principal", "role"), ret=T.UNIT,
     argtypes=(T.TEXT, T.TEXT), effects=("audit",), caps=("AuditWrite",),
     doc="Give a principal a role. Requires AuditWrite.")
def _identity_grant_role(ctx, principal: str, role: str) -> Any:
    ctx.require_capability("AuditWrite", what="identity.grant_role")
    name = to_text(principal)
    record = _state(ctx)["principals"].get(name)
    if record is None:
        raise GamaRuntimeFault("IdentityError", f"`{name}` is not registered")
    record["roles"].add(to_text(role))
    ctx.audit.record("IDENTITY_ROLE_GRANTED", level="security", object=name,
                     reason=f"role {to_text(role)}")
    return UNIT


@reg("identity.has_role", ("principal", "role"), ret=T.BOOL,
     argtypes=(T.TEXT, T.TEXT), effects=("audit",),
     doc="Whether a principal holds a role.")
def _identity_has_role(ctx, principal: str, role: str) -> bool:
    record = _state(ctx)["principals"].get(to_text(principal))
    return record is not None and to_text(role) in record["roles"]


@reg("identity.roles", ("principal",), ret=T.TEXT, argtypes=(T.TEXT,),
     effects=("audit",), doc="A principal's roles, comma separated.")
def _identity_roles(ctx, principal: str) -> str:
    record = _state(ctx)["principals"].get(to_text(principal))
    if record is None:
        return ""
    return ", ".join(sorted(record["roles"]))


# ---------------------------------------------------------------------------
# workflow
# ---------------------------------------------------------------------------

@reg("workflow.define", ("name", "steps"), ret=T.I64,
     argtypes=(T.TEXT, T.TEXT), effects=("audit",), caps=("AuditWrite",),
     doc="Define a workflow from a comma-separated list of step names.")
def _workflow_define(ctx, name: str, steps: str) -> int:
    ctx.require_capability("AuditWrite", what="workflow.define")
    names = [part.strip() for part in to_text(steps).split(",") if part.strip()]
    if not names:
        raise GamaRuntimeFault("WorkflowError", "a workflow needs at least one "
                                                "step")
    _state(ctx)["workflows"][to_text(name)] = names
    ctx.audit.record("WORKFLOW_DEFINED", object=to_text(name),
                     reason=f"{len(names)} step(s)")
    return len(names)


@reg("workflow.start", ("name",), ret=T.ANY, argtypes=(T.TEXT,),
     effects=("audit",), caps=("AuditWrite",),
     doc="Start a run of a defined workflow; returns its run record.")
def _workflow_start(ctx, name: str) -> GRecord:
    ctx.require_capability("AuditWrite", what="workflow.start")
    definition = _state(ctx)["workflows"].get(to_text(name))
    if definition is None:
        raise GamaRuntimeFault(
            "WorkflowError", f"`{name}` has not been defined",
            hint="define it with `workflow.define` first")
    run_id = _next_id(ctx)
    run = {"definition": to_text(name), "index": 0,
           "history": [f"started at {definition[0]}"]}
    _state(ctx)["workflow_runs"][run_id] = run
    ctx.audit.record("WORKFLOW_STARTED", object=run_id,
                     reason=f"definition {to_text(name)}")
    return GRecord("WorkflowRun", {"id": run_id, "definition": to_text(name),
                                   "step": definition[0], "index": 0,
                                   "done": False})


def _workflow_run(ctx, handle) -> Tuple[str, Dict[str, Any]]:
    if not isinstance(handle, GRecord):
        raise TypeFault("expected a workflow run from workflow.start")
    run_id = to_text(handle.fields.get("id", ""))
    run = _state(ctx)["workflow_runs"].get(run_id)
    if run is None:
        raise GamaRuntimeFault("WorkflowError",
                               f"run {run_id!r} does not exist in this run")
    return run_id, run


@reg("workflow.advance", ("run",), ret=T.ANY, argtypes=(T.ANY,),
     effects=("audit",), caps=("AuditWrite",),
     doc="Move a workflow run to its next step.")
def _workflow_advance(ctx, run_handle) -> GRecord:
    ctx.require_capability("AuditWrite", what="workflow.advance")
    run_id, run = _workflow_run(ctx, run_handle)
    definition = _state(ctx)["workflows"][run["definition"]]
    if run["index"] >= len(definition) - 1:
        # Advancing past the end is a no-op rather than an error: a supervisor
        # that retries must not push the run out of range.
        done = True
        step = definition[-1]
    else:
        run["index"] += 1
        step = definition[run["index"]]
        done = run["index"] >= len(definition) - 1
        run["history"].append(f"advanced to {step}")
    ctx.audit.record("WORKFLOW_ADVANCED", object=run_id, reason=f"step {step}")
    return GRecord("WorkflowRun", {"id": run_id,
                                   "definition": run["definition"],
                                   "step": step, "index": run["index"],
                                   "done": done})


@reg("workflow.state", ("run",), ret=T.TEXT, argtypes=(T.ANY,),
     effects=("audit",), doc="The step a run is on.")
def _workflow_state(ctx, run_handle) -> str:
    _run_id, run = _workflow_run(ctx, run_handle)
    definition = _state(ctx)["workflows"][run["definition"]]
    return definition[run["index"]]


@reg("workflow.history", ("run",), ret=T.TEXT, argtypes=(T.ANY,),
     effects=("audit",),
     doc="Every transition a run has made, oldest first.")
def _workflow_history(ctx, run_handle) -> str:
    _run_id, run = _workflow_run(ctx, run_handle)
    return "; ".join(run["history"])


# ---------------------------------------------------------------------------
# messaging
# ---------------------------------------------------------------------------

@reg("messaging.subscribe", ("topic",), ret=T.I64, argtypes=(T.TEXT,),
     effects=("storage",), caps=("AuditWrite",),
     doc="Subscribe to a topic; returns the subscriber count.")
def _messaging_subscribe(ctx, topic: str) -> int:
    ctx.require_capability("AuditWrite", what="messaging.subscribe")
    state = _state(ctx)
    name = to_text(topic)
    state["subscriptions"][name] = state["subscriptions"].get(name, 0) + 1
    state["topics"].setdefault(name, [])
    return state["subscriptions"][name]


@reg("messaging.publish", ("topic", "payload"), ret=T.I64,
     argtypes=(T.TEXT, T.TEXT), effects=("storage",), caps=("AuditWrite",),
     doc="Publish a message to a topic; returns the topic's depth.")
def _messaging_publish(ctx, topic: str, payload: str) -> int:
    ctx.require_capability("AuditWrite", what="messaging.publish")
    state = _state(ctx)
    name = to_text(topic)
    if not state["subscriptions"].get(name):
        # A publish nobody is listening for is dropped by every broker, and
        # saying so is better than silently accumulating messages for a
        # subscriber that will never come.
        raise GamaRuntimeFault(
            "MessagingError", f"no subscriber for `{name}`",
            hint="call `messaging.subscribe` first; this implementation is "
                 "in-process and has no broker to buffer for later")
    state["topics"].setdefault(name, []).append(to_text(payload))
    ctx.audit.record("MESSAGE_PUBLISHED", object=name,
                     reason=f"{len(payload)} character(s)")
    return len(state["topics"][name])


@reg("messaging.drain", ("topic",), ret=T.ANY, argtypes=(T.TEXT,),
     effects=("storage",), caps=("DatabaseRead",),
     doc="Take every pending message from a topic, oldest first.")
def _messaging_drain(ctx, topic: str) -> List[str]:
    ctx.require_capability("DatabaseRead", what="messaging.drain")
    state = _state(ctx)
    name = to_text(topic)
    messages = state["topics"].get(name, [])
    state["topics"][name] = []
    return list(messages)


@reg("messaging.depth", ("topic",), ret=T.I64, argtypes=(T.TEXT,),
     effects=("storage",), doc="How many messages are pending on a topic.")
def _messaging_depth(ctx, topic: str) -> int:
    return len(_state(ctx)["topics"].get(to_text(topic), []))


@reg("messaging.topics", (), ret=T.TEXT, effects=("storage",),
     doc="Topics this run has used, comma separated.")
def _messaging_topics(ctx) -> str:
    return ", ".join(sorted(_state(ctx)["topics"]))


# ---------------------------------------------------------------------------
# observability
# ---------------------------------------------------------------------------

@reg("observability.counter", ("name", "delta"), ret=T.I64,
     argtypes=(T.TEXT, T.I64), effects=("audit",),
     doc="Add to a counter and return its new value.")
def _observability_counter(ctx, name: str, delta: int) -> int:
    counters = _state(ctx)["counters"]
    key = to_text(name)
    counters[key] = counters.get(key, 0) + int(delta)
    return counters[key]


@reg("observability.gauge", ("name", "value"), ret=T.F64,
     argtypes=(T.TEXT, T.F64), effects=("audit",),
     doc="Set a gauge and return the value.")
def _observability_gauge(ctx, name: str, value: float) -> float:
    _state(ctx)["gauges"][to_text(name)] = float(value)
    return float(value)


@reg("observability.log", ("level", "message"), ret=T.UNIT,
     argtypes=(T.TEXT, T.TEXT), effects=("audit",),
     doc="Record a structured log line.")
def _observability_log(ctx, level: str, message: str) -> Any:
    _state(ctx)["logs"].append((to_text(level).lower(), to_text(message)))
    return UNIT


@reg("observability.span", ("name", "duration_ms"), ret=T.TEXT,
     argtypes=(T.TEXT, T.F64), effects=("audit",),
     doc="Record a span with its duration; returns the span id.")
def _observability_span(ctx, name: str, duration_ms: float) -> str:
    span_id = _next_id(ctx)
    _state(ctx)["spans"].append((to_text(name), float(duration_ms), span_id))
    return span_id


@reg("observability.counters", (), ret=T.TEXT, effects=("audit",),
     doc="Every counter and its value.")
def _observability_counters(ctx) -> str:
    counters = _state(ctx)["counters"]
    if not counters:
        return "no counters"
    return ", ".join(f"{k}={v}" for k, v in sorted(counters.items()))


@reg("observability.gauges", (), ret=T.TEXT, effects=("audit",),
     doc="Every gauge and its value.")
def _observability_gauges(ctx) -> str:
    gauges = _state(ctx)["gauges"]
    if not gauges:
        return "no gauges"
    return ", ".join(f"{k}={v}" for k, v in sorted(gauges.items()))


@reg("observability.logs", (), ret=T.TEXT, effects=("audit",),
     doc="Every log line this run recorded, oldest first.")
def _observability_logs(ctx) -> str:
    logs = _state(ctx)["logs"]
    if not logs:
        return "no log lines"
    return "; ".join(f"[{level}] {message}" for level, message in logs)


@reg("observability.spans", (), ret=T.TEXT, effects=("audit",),
     doc="Every span this run recorded, with its duration.")
def _observability_spans(ctx) -> str:
    spans = _state(ctx)["spans"]
    if not spans:
        return "no spans"
    return "; ".join(f"{name}={duration:.3f}ms" for name, duration, _id in spans)


@reg("observability.snapshot", (), ret=T.TEXT, effects=("audit",),
     doc="Everything this run has observed, as one report.")
def _observability_snapshot(ctx) -> str:
    state = _state(ctx)
    payload = {
        "counters": dict(sorted(state["counters"].items())),
        "gauges": {k: v for k, v in sorted(state["gauges"].items())},
        "logs": [{"level": level, "message": message}
                 for level, message in state["logs"]],
        "spans": [{"name": name, "duration_ms": duration, "id": span_id}
                  for name, duration, span_id in state["spans"]],
    }
    return json.dumps(payload, sort_keys=True)
