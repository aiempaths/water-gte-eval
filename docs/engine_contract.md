# Engine contract (GTE ↔ harness)

The Geometric Trust Engine (GTE) is a black box. The harness talks to it **only**
through `gte_eval.engine_adapter.EngineAdapter` (`src/gte_eval/engine_adapter/`).
Two adapters ship today: `MockEngine` (in-process, deterministic) and `HttpEngine`
(REST client). The contract tests in `tests/test_engine_contract.py` run against
**any** adapter and must pass for both.

## 1. Python interface

```python
class EngineAdapter(ABC):
    def reset_history(self) -> None: ...                    # clears history AND the score log/receipt chain
    def add_history(self, windows: list[HistoryWindow]) -> None: ...
    def score(self, window: Window) -> EngineResult: ...    # measures latency_ms, appends to the log
    def health(self) -> dict: ...                           # must contain "status"
    # provided by the base class:
    #   verify_receipt(result) -> bool, verify_chain() -> bool, log, chain_head
```

```python
@dataclass
class Window:
    window_id: str
    timestamps: list[str]          # ISO-8601, one per sample in the window
    sensors: dict[str, list[float]]  # tag -> values, aligned with timestamps

@dataclass
class HistoryWindow(Window):
    outcome_label: str | int | None

@dataclass(frozen=True)
class EngineResult:
    window_id: str
    closeness: float        # distance to nearest past windows; lower = more familiar
    agreement: float        # fraction of nearest neighbors sharing the majority outcome; higher = more consistent
    verdict: str            # "trust" | "route_to_human" | "unavailable"
    neighbor_ids: list[str]
    receipt: str            # sha256 hex, see §4
    latency_ms: float       # measured by the adapter base, NOT bound into the receipt
    raw: dict               # engine-specific details, NOT bound into the receipt
```

Semantics: `close + agreeing = trust`; far or mixed = `route_to_human`; no usable
history (or engine-side outage) = `unavailable` (`closeness=inf`, `agreement=0.0`,
`neighbor_ids=[]`). The harness routes `unavailable` to a human. The harness only
adds history windows that end **strictly before** the scored window starts.

## 2. Window / history JSON shapes (HTTP transport)

`Window` serializes as:

```json
{"window_id": "w00042",
 "timestamps": ["2024-06-01T00:00:00", "2024-06-01T00:05:00"],
 "sensors": {"flow": [120.1, 119.8], "pressure": [3.21, 3.20]}}
```

`HistoryWindow` adds `"outcome_label": "normal"` (string, number, or null).

## 3. HTTP endpoints (all names configurable — defaults are PLACEHOLDERs)

Configured under `engine.params` for `type: http` (see `configs/example_http.yaml`):

| key | default (PLACEHOLDER) | method | request body | success response |
| --- | --- | --- | --- | --- |
| `endpoints.reset` | `POST /history/reset` | `{}` | `{"status": "ok", ...}` |
| `endpoints.append` | `POST /history/append` | `{"windows": [<HistoryWindow JSON>, ...]}` (key from `history_wrapper`) | `{"status": "ok", ...}` |
| `endpoints.score` | `POST /score` | `{"window": <Window JSON>}` (key from `score_wrapper`) | EngineResult JSON, see §5 |
| `endpoints.health` | `GET /health` | — | any JSON object; should include `"status"` |

- `field_map`: maps each EngineResult field (`window_id`, `closeness`, `agreement`,
  `verdict`, `neighbor_ids`, `receipt`) to the name your API uses in the `/score`
  response. Defaults are identity (PLACEHOLDER names).
- `request_field_map`: renames the outgoing per-window keys
  (`window_id`, `timestamps`, `sensors`, `outcome_label`).
- `timeout_s` (per attempt), `retries` (total attempts), `backoff_s` (linear backoff
  `backoff_s * attempt` between attempts).

Error behavior of `HttpEngine`:
- Connection error / timeout on every attempt → `EngineUnavailableError`:
  `engine API unreachable at <url> after N attempt(s) (timeout=Ts): <cause>`.
- HTTP >= 400 (except 503 on `/score`) → `EngineError` with method, URL, status, and a
  200-char body excerpt.
- HTTP 503 on `/score` → `EngineResult` with `verdict="unavailable"`,
  `closeness=inf`, `agreement=0.0`, `neighbor_ids=[]`, receipt computed client-side
  per §4 so the chain stays verifiable.
- Missing mapped fields or invalid `verdict` → `EngineError` listing the response keys.

## 4. Receipt (tamper-evident hash chain)

```
receipt_n = sha256( canonical_json({
    "prev_receipt": receipt_{n-1}   # or "GENESIS" for the first score after reset
    "window_id":    window.window_id
    "inputs":       {"window_id", "timestamps", "sensors"}   # the exact Window JSON
    "outputs":      {"closeness", "agreement", "verdict", "neighbor_ids"}
}))
canonical_json = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)
```

- `latency_ms` and `raw` are **not** bound into the receipt → same inputs give
  **byte-identical receipts**.
- Receipts are chain-position dependent: scoring the same window twice on one
  adapter advances the chain (second receipt differs), while two fresh adapters
  given the same history produce identical receipts.
- `reset_history()` resets the chain to `GENESIS`.
- Verification: `adapter.verify_receipt(result)` re-derives one receipt from the
  log; `adapter.verify_chain()` re-derives the whole chain. Altering any logged
  field (closeness, agreement, verdict, neighbor_ids, window_id, receipt) breaks
  verification. `HttpEngine` verifies client-side using this same contract, so the
  server must implement it too.

## 5. `/score` response JSON

Flat object (extra keys are kept in `EngineResult.raw`); field names remapped by
`field_map`:

```json
{"window_id": "w00042", "closeness": 0.31, "agreement": 1.0,
 "verdict": "trust", "neighbor_ids": ["w00007", "w00011", "w00019"],
 "receipt": "9f2c…64hex"}
```

Strict JSON has no `Infinity`: when `verdict` is `"unavailable"` the server returns
`"closeness": null` (and clients map it to `inf`); any other `null` closeness is a
contract violation. In-process (MockEngine), `closeness` is a real float `inf`.

## 6. `/health` response

Any JSON object; should include `"status": "ok"`. `MockEngine.health()` returns
`{"status", "engine", "n_history_windows", "n_scored", "k", "closeness_max",
"agreement_min", "chain_head"}`. `HttpEngine.health()` returns the server object
under `server_health` plus client-side `status`, `engine`, `base_url`,
`n_appended_windows`, `n_scored`, `chain_head`, and `n_history_windows` when the
server reports it.

## 7. MockEngine scoring semantics (reference implementation)

Per sensor tag (sorted): fingerprint = `[mean, std, slope (least-squares vs sample
index), spikiness (excess kurtosis, m4/m2²−3), lag-1 autocorrelation]`.
History fingerprints are z-scored by running history mean/std (std floored at
1e-12). `closeness` = mean euclidean distance to the `k` nearest history windows
(ties broken by `window_id`); `agreement` = fraction of those `k` neighbors whose
`outcome_label` equals the majority label (ties by `str(label)`);
`verdict` = `trust` iff `closeness <= closeness_max` and `agreement >=
agreement_min`, else `route_to_human`; empty history → `unavailable`.
Config params: `k` (default 5), `closeness_max` (2.0), `agreement_min` (0.6).

## 8. Reference fake server

`tests/fake_engine_server.py` implements this contract with FastAPI (loopback only)
for testing `HttpEngine` without the real engine:
`POST /history/reset`, `POST /history/append`, `POST /score`, `GET /health`.
