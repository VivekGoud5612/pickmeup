# Service Boundaries

## 1. Experimentation Service

### Owns

- Experiment
- Scenario
- TrainingRun
- EvaluationRun
- Experiment lifecycle
- Run lifecycle
- Experiment configuration/version references
- Reproducibility metadata
- Checkpoint/result references
- Experiment-level orchestration

### Does NOT own

- simulation execution
- physics
- robot implementation
- training loop implementation
- runtime worker processes
- raw telemetry storage
- research reasoning

---

# 2. Engine Service

### Owns

- Runtime Registry
- Runtime Worker
- Execution Handle
- Worker Launcher
- Engine execution
- Simulation execution
- Training execution
- Runtime lifecycle
- Runtime events
- Runtime state

### Does NOT own

- Experiment entity
- TrainingRun business entity
- Research questions
- Experiment history
- Research analysis
- long-term experiment analytics

---

# 3. Telemetry / Data

### Owns

Raw execution data:

- observations
- actions
- rewards
- trajectories
- sensor streams
- logs
- execution events
- runtime measurements

### Does NOT own

- experiment meaning
- research conclusions
- business lifecycle of a run

---

# 4. Evaluation

### Owns

Transformation of execution data into measurements.

Examples:

- win rate
- success rate
- reward statistics
- episode length
- trajectory metrics
- collision metrics
- energy metrics
- agent-specific metrics

### Does NOT own

- simulation execution
- experiment lifecycle
- research interpretation

---

# 5. Analytics

### Owns

Interpretation and aggregation of metrics.

Examples:

- trend detection
- experiment comparisons
- correlations
- historical analysis
- statistical summaries

### Does NOT own

- raw execution
- runtime workers
- core experiment lifecycle

---

# 6. Research Intelligence

### Owns

Higher-level research reasoning.

Potential future responsibilities:

- hypothesis generation
- experiment suggestions
- result interpretation
- research summaries
- report generation
- configuration suggestions
- code-change suggestions

### Does NOT own

- direct engine internals
- runtime lifecycle
- raw database access across every service

---

# 7. Artifact / Dataset Management

### Owns

Persistent research artifacts and provenance.

Potential artifacts:

- checkpoints
- models
- datasets
- trajectories
- experiment outputs
- evaluation results

The exact service boundary is intentionally not finalized.

---

# 8. Gateway / SDK

### Owns

Communication with Engine Service.

Responsibilities:

- request construction
- serialization
- HTTP/gRPC transport
- response parsing
- typed client API
- communication errors

### Does NOT own

- training logic
- simulation logic
- experiment business rules

---

# 9. Database Boundary

Each service owns its persistence model.

Domain entities should not be directly persisted through another service's
repository.

Use:

    Domain Entity
        |
        v
    Repository Interface
        |
        v
    Infrastructure / ORM

Do not allow:

    Service A
       |
       +---- directly accesses Service B database

---

# 10. Shared Contracts Boundary

Shared contracts may contain communication structures.

Allowed:

- DTOs
- events
- enums
- transport models
- stable cross-service value-like structures

Not allowed merely for convenience:

- domain entities
- repositories
- ORM models
- service implementations
- business logic

---

# 11. Dependency Direction

Preferred dependency direction:

    Presentation
        ↓
    Application
        ↓
    Domain

Infrastructure implements interfaces required by Application/Domain.

The domain must not depend on:

- FastAPI
- SQLAlchemy
- Redis
- Kafka
- HTTP clients
- concrete Engine implementations

---

# 12. Engine Communication

Preferred:

    Application
       ↓
    Engine SDK abstraction
       ↓
    Engine Client
       ↓
    HTTP/gRPC
       ↓
    Engine Service

The application should not construct raw HTTP requests.

---

# 13. Event Communication

Preferred:

    Engine
       ↓
    Event Contract
       ↓
    Publisher
       ↓
    Event Channel
       ↓
    Consumers

The Engine should publish facts.

Consumers decide what to do with them.

---

# 14. Extending the System

When adding functionality, first identify which responsibility owns it.

Examples:

New simulator
    -> Engine

New RL algorithm
    -> Engine / learning module

New experiment type
    -> Experimentation

New evaluation metric
    -> Evaluation

New telemetry format
    -> Telemetry/Data

New statistical analysis
    -> Analytics

New research reasoning capability
    -> Research Intelligence

New persistent artifact type
    -> Artifact/Data Management

New transport mechanism
    -> Infrastructure / SDK

Do not create a new service simply because a new class is needed.

---

# 15. Boundary Test

Before putting a piece of logic somewhere, ask:

1. Does it describe research intent?
2. Does it describe world conditions?
3. Does it execute something?
4. Does it collect raw data?
5. Does it measure performance?
6. Does it interpret measurements?
7. Does it manage persistent artifacts?
8. Is it merely communication infrastructure?

The answer should determine its boundary.

---

# 16. Evolution Rule

A new feature should ideally modify one responsibility and communicate through
an existing boundary.

If adding a feature requires unrelated services to understand its internal
implementation, the boundary should be reconsidered.