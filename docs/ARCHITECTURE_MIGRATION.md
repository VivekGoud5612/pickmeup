# Architecture Migration Status

## Current migration

The existing RL simulation, environment, policy, rollout, and MAPPO training
logic remains in `engine/`.  It is execution-plane implementation and has not
been replaced by placeholder algorithms.

The control-plane implementation remains temporarily located at
`backend/training_service/` for compatibility with existing imports.  Its
`TrainingRun` remains a business record and must not be used as an Engine
runtime worker.  `backend/experimentation_service/` now establishes the
documented `Experiment` and `Scenario` domain concepts; migration of the
legacy training package name is a compatibility-breaking follow-up.

## Completed boundary changes

- `RuntimeRegistry` and `RuntimeWorker` are owned by `backend/engine_service`.
- `ExecutionHandle` protects runtime workers from a concrete execution
  mechanism.
- The local gateway is an adapter to `LocalEngineService`; it does not create
  or store runtime workers.
- Shared AgentRole, EngineStatus, requests, responses, and events are in
  `backend/contracts/engine`.
- Engine lifecycle events carry a control-plane `run_id`, rather than a
  presentation-oriented run name.
- Metrics are nullable until a workload telemetry adapter reports measured
  values.  No fabricated reward, win-rate, or loss values are emitted.

## Deliberately deferred

The legacy MAPPO loop is a standalone executable and does not yet expose its
per-update metrics through the Engine execution-handle interface.  Connecting
that real loop (including cooperative cancellation and pause semantics) is
intentionally deferred rather than inventing training behavior.  It belongs in
an Engine-internal workload adapter plus Telemetry integration.
