# Refactoring Instructions for Codex

## 1. Role

You are assisting with the architectural refactoring of an existing Robotics
Research Platform.

The repository contains working implementation from an earlier RL-focused
architecture.

Your task is to improve architecture and code organization while preserving
existing domain and research logic.

You are NOT the owner of the research design.

The human developer owns:

- domain decisions
- research decisions
- algorithmic decisions
- service boundaries
- business rules
- final architectural decisions

---

# 2. Read Before Changing

Before making substantial changes:

1. Inspect the repository.
2. Identify the existing services.
3. Identify current domain entities.
4. Identify existing engine logic.
5. Identify existing training logic.
6. Identify existing contracts.
7. Identify gateway/client implementations.
8. Identify database/repository implementations.
9. Identify tests.
10. Identify existing imports and dependency relationships.

Do not assume that a class is unnecessary simply because it does not fit the
new architecture immediately.

---

# 3. Preserve Working Logic

The existing engine and RL implementation contains valuable research work.

Do NOT delete working logic merely to make the architecture cleaner.

Do NOT replace working implementations with:

- zero values
- placeholder methods
- fake metrics
- empty training loops
- dummy environments
- TODO-only implementations

If a component needs to move, move it.

If an interface needs to change, adapt it.

If the implementation is genuinely incompatible, explain why before deleting
or replacing it.

---

# 4. Separate Refactoring From New Behavior

Mechanical refactoring is allowed:

- moving files
- renaming modules
- updating imports
- splitting responsibilities
- introducing interfaces
- adapting clients
- introducing DTOs
- introducing repositories
- introducing dependency injection
- improving type safety
- improving structure
- adding tests around existing behavior

New business behavior requires explicit reasoning.

Do not invent:

- research rules
- evaluation rules
- simulation behavior
- reward functions
- curriculum logic
- domain lifecycle rules
- metrics definitions

If the behavior is unspecified, leave a TODO/interface and explain what is
missing.

---

# 5. Architecture Source of Truth

Use:

    docs/ARCHITECTURE.md
    docs/SERVICE_BOUNDARIES.md

as the architectural source of truth.

Use:

    docs/REFACTORING_INSTRUCTIONS.md

as the rules for modifying the repository.

If the existing code conflicts with the architecture, do not silently resolve
the conflict by deleting functionality.

Explain:

1. Current implementation
2. Architectural conflict
3. Proposed migration
4. Files affected
5. Behavior preserved
6. New behavior required, if any

---

# 6. Do Not Over-Abstract

Do not introduce an abstraction merely because abstraction is possible.

Good reasons include:

- service boundary
- external dependency
- replaceable execution mechanism
- persistence boundary
- transport boundary
- event transport
- meaningful domain concept

Avoid abstractions that only add indirection without protecting a boundary.

---

# 7. Engine Service

The Engine Service is an execution plane.

Its internal implementation may be complex.

Do not force all simulation, RL, robotics, perception, planning, and control
logic into separate microservices prematurely.

The Engine Service may contain internal modules for these capabilities.

The external boundary should remain stable.

---

# 8. Runtime Worker

RuntimeWorker represents one currently executing runtime context.

It is NOT the same thing as TrainingRun.

TrainingRun belongs to Experimentation.

RuntimeWorker belongs to Engine runtime.

Use:

    TrainingRun ID
          |
          v
    RuntimeWorker

The runtime worker may reference:

- run identity
- engine status
- execution handle
- runtime metrics
- progress
- checkpoint metadata

---

# 9. Execution Handle

ExecutionHandle abstracts how work actually runs.

Potential implementations:

- local process
- thread
- Ray actor
- container
- Kubernetes pod
- remote GPU worker

Do not couple RuntimeWorker to one implementation unnecessarily.

---

# 10. Shared Contracts

Shared contracts are for communication.

Potential contents:

    contracts/
        engine/
            dto/
            events/
            models/
            enums/

Potential shared models include:

- TrainingConfiguration
- TrainingProgress
- CheckpointMetadata
- PerformanceMetrics

Potential enums include:

- CheckpointType
- AgentRole
- relevant lifecycle enums

Potential DTOs include:

- StartTrainingRequest
- StopTrainingRequest
- PauseTrainingRequest
- ResumeTrainingRequest
- SaveCheckpointRequest
- EvaluationRequest
- corresponding responses

Potential events include:

- TrainingInitializedEvent
- TrainingStartedEvent
- TrainingPausedEvent
- TrainingResumedEvent
- TrainingStoppedEvent
- TrainingCompletedEvent
- TrainingFailedEvent
- CheckpointCreatedEvent
- EvaluationStartedEvent
- EvaluationCompletedEvent

Only add a contract when it represents a real communication boundary.

---

# 11. AgentRole

AgentRole is relevant to evaluation and agent-specific performance metrics.

It should be represented consistently across the relevant contracts.

Do not hard-code role assumptions into generic infrastructure.

The existing RL roles may include:

- Tank
- Dealer
- Healer
- Boss

The system should remain capable of supporting additional roles later.

---

# 12. DTO Rules

DTOs represent communication.

They may change when the communication boundary changes.

Do not treat DTOs as domain entities.

Do not place API-specific DTOs into the domain merely because they contain
similar data to a domain object.

---

# 13. Domain Rules

Domain entities represent concepts with:

- identity
- lifecycle
- behavior

Value objects represent concepts without independent identity.

Do not create domain entities simply to hold data.

Do not create value objects simply to avoid primitive fields.

Use domain objects when they protect actual domain meaning.

---

# 14. Database Rules

Do not couple domain entities directly to SQLAlchemy ORM models.

Preferred:

    Domain Entity
         |
    Repository Interface
         |
    ORM Model
         |
    Database

Persistence details belong to infrastructure.

Database schema should support the domain but should not define the domain.

---

# 15. Event Rules

Events represent facts.

They should be immutable.

Examples:

    TrainingStarted
    CheckpointCreated
    EvaluationCompleted

The publisher should not know the consumers' business logic.

Handlers belong to the service responsible for reacting to the event.

---

# 16. Gateway / SDK Rules

The Engine Gateway is a communication layer.

It should hide:

- HTTP
- gRPC
- serialization
- transport errors

from application use cases.

The SDK should not contain Engine implementation.

---

# 17. Runtime Registry Rules

RuntimeRegistry tracks active runtime workers.

Initial implementation may be:

    dict[RunID, RuntimeWorker]

Do not prematurely introduce:

- Redis
- Kubernetes
- Ray
- distributed schedulers

unless the architecture explicitly requires them.

The abstraction should allow these implementations later.

---

# 18. Event Infrastructure

Start with local/in-memory event infrastructure.

Keep the event abstraction independent of the concrete transport.

Future implementations may include:

- Redis
- Kafka
- NATS
- RabbitMQ
- gRPC streams

Do not implement future infrastructure unless required.

---

# 19. Extensibility

The architecture must allow new functionality later.

When adding a new feature:

1. Identify its responsibility.
2. Identify its owner.
3. Identify the existing boundary it communicates through.
4. Add the smallest required implementation.
5. Avoid modifying unrelated services.

Example:

Adding a new simulator should primarily affect Engine internals.

Adding a new metric should primarily affect Evaluation.

Adding a new research analysis should primarily affect Analytics/Research.

Adding a new transport should primarily affect infrastructure.

---

# 20. Permission Policy

Proceed without asking for permission for:

- import fixes
- mechanical file moves
- formatting
- type fixes
- obvious test fixes
- updating references after approved refactors

Ask before:

- deleting working domain logic
- replacing algorithms
- removing major modules
- changing business behavior
- changing service boundaries
- changing database semantics
- replacing the engine implementation
- introducing a major new dependency
- changing contracts in a breaking way

If uncertain, explain the proposed change and ask.

---

# 21. Breaking Changes

Prefer backwards-compatible changes when practical.

If a breaking change is necessary:

1. Identify all consumers.
2. Identify affected contracts.
3. Update the boundary deliberately.
4. Update tests.
5. Explain the migration.

Do not silently break consumers.

---

# 22. Testing

After architectural refactoring:

- run existing tests
- add tests for new boundaries
- verify imports
- verify service initialization
- verify Engine lifecycle
- verify gateway communication
- verify events
- verify repositories
- verify serialization/deserialization

Do not claim functionality works if it has not been tested.

---

# 23. Placeholder Policy

If implementation details are unknown:

Prefer:

    interface
    TODO
    explicit NotImplementedError

over inventing behavior.

A placeholder is acceptable when the architecture is known but the actual
implementation has not yet been designed.

Clearly explain the placeholder.

---

# 24. Change Report

After making a substantial change, report:

### Changed

- files changed
- architecture changed
- responsibilities moved

### Preserved

- existing functionality
- algorithms
- domain behavior

### Added

- new interfaces
- new contracts
- new services
- new tests

### Not Implemented

- intentionally deferred functionality

### Risks

- compatibility concerns
- migrations
- behavior that needs manual verification

---

# 25. Most Important Rule

Do not confuse architectural cleanliness with deleting complexity.

The existing project contains real research implementation.

The goal is:

    preserve the intelligence
    reorganize the boundaries
    isolate responsibilities
    make communication explicit
    make infrastructure replaceable
    make future capabilities easier to add

The architecture should make future work easier, not erase previous work.