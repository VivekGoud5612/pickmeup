# Robotics Research Platform — Architecture

## 1. Purpose

This repository is evolving from an RL training project into a broader
Robotics Research Platform.

Reinforcement Learning is one capability of the platform, not the definition
of the platform.

The platform should eventually support:

- simulation
- robotics
- reinforcement learning
- multi-agent systems
- perception
- state estimation
- planning
- control
- experimentation
- evaluation
- telemetry
- analytics
- research intelligence
- artifact and dataset management

The architecture must therefore avoid coupling the entire platform to one
algorithm, simulator, training implementation, or research workflow.

---

# 2. Core Architectural Principle

Separate:

1. Research intent
2. Experimental conditions
3. Execution
4. Data collection
5. Evaluation
6. Analysis
7. Persistent research artifacts

The most important distinction is:

    Experiment = what we are trying to find out

    Scenario = under what world conditions

    Run = what actually happened

    Engine = executes the world

    Telemetry = records what happened

    Evaluation = measures what happened

    Analysis = interprets what happened

---

# 3. High-Level Architecture

                         ROBOTICS RESEARCH PLATFORM
                                  |
                   +--------------+--------------+
                   |                             |
             CONTROL / RESEARCH PLANE       EXECUTION PLANE
                   |                             |
          +--------+---------+             +-----+------+
          |                  |             |            |
   Experimentation       Research       Engine       Workers
      Service          Intelligence    Service
          |                                |
          |                          Runtime Registry
          |                                |
          |                          Runtime Workers
          |                                |
          |                         Execution Handles
          |                                |
          |                              Engine
          |
          +--------------------+
                               |
                         Telemetry / Data
                               |
                          Evaluation
                               |
                           Analytics


The exact number of physical services is not fixed yet.

Responsibilities should be separated first.

Deployment boundaries can evolve later.

---

# 4. Control Plane vs Execution Plane

## Control / Research Plane

The control plane describes what should happen and tracks what happened.

It contains concepts such as:

- experiments
- scenarios
- training runs
- evaluation runs
- configurations
- checkpoints
- research metadata
- experiment lifecycle

It should not directly contain the implementation details of simulation
execution.

---

## Execution Plane

The execution plane actually performs work.

The Engine Service belongs here.

It is responsible for:

- creating runtime workers
- starting execution
- pausing execution
- resuming execution
- stopping execution
- exposing runtime state
- publishing runtime events
- executing simulations
- running training
- running robotics workloads
- interacting with execution resources

The execution plane should not own research meaning.

---

# 5. Experimentation

An experiment represents research intent.

It answers:

> What are we trying to find out?

An experiment may contain:

- research question
- experiment configuration
- configuration version
- scenarios
- runs
- evaluation criteria
- checkpoints/results references
- reproducibility metadata

An experiment does not execute the simulation itself.

---

# 6. Scenario

A scenario describes the conditions under which execution occurs.

It answers:

> Under what world conditions should this execution happen?

A scenario may eventually contain:

- map
- terrain
- obstacles
- objects
- agents
- goals
- environment parameters
- initial state
- random seed
- physics parameters
- sensor configuration
- environmental conditions

The Scenario describes the world.

The Engine executes the world.

---

# 7. Run / Execution Record

A run represents an execution of an experiment.

The distinction is:

    Experiment = intention

    Run = execution

Examples:

- TrainingRun
- EvaluationRun

A run owns lifecycle information such as:

- identity
- status
- start time
- completion time
- scenario reference
- configuration reference
- checkpoint references
- execution metadata

A run should remain a domain record.

It should not become the execution engine.

The Engine Service executes the run.

The Experimentation Service tracks it.

---

# 8. Engine Service

The Engine Service is an execution service.

Its purpose is:

> Execute workloads requested by the control plane.

It should be treated as an execution sandbox.

The Engine Service may eventually execute:

- RL training
- multi-agent training
- simulation
- robotics controllers
- perception pipelines
- planning
- control
- evaluation workloads
- physics simulation

The Engine itself may contain significant complexity.

That complexity is allowed.

The important requirement is that it remains behind a clear service boundary.

---

# 9. Engine Runtime Architecture

The Engine Service should conceptually follow:

    API
      |
    Use Case
      |
    Runtime Registry
      |
    Runtime Worker
      |
    Execution Handle
      |
    Engine


## Runtime Registry

The registry tracks currently running execution contexts.

It may initially be an in-memory dictionary.

It should not be designed around the assumption that there is only one engine.

Example:

    training_run_id -> runtime_worker

Multiple executions should be possible.

---

## Runtime Worker

A Runtime Worker represents the logical runtime identity of one execution.

It may contain:

- training/run identity
- engine state
- execution handle
- runtime metrics
- latest checkpoint metadata
- progress
- execution metadata

It does not represent the business concept of a TrainingRun.

It represents its runtime counterpart.

---

## Execution Handle

Execution Handle represents the actual execution resource.

The implementation may eventually be:

- local process
- thread
- Ray actor
- container
- Kubernetes pod
- remote machine
- GPU worker

The Runtime Worker should not depend on the implementation type.

---

# 10. Engine Domain Boundary

The Engine Service should contain only concepts that have meaning to execution.

Potential domain concepts:

## Entity

- RuntimeWorker

## Value Objects

- RuntimeMetrics
- CheckpointMetadata

## Enums

- EngineStatus
- WorkerStatus
- CheckpointType

Do not introduce domain entities merely because another service has an entity
with the same concept.

For example:

    TrainingRun

belongs to Experimentation.

The Engine should receive a run identifier and execute it.

---

# 11. Shared Contracts

Services communicate through contracts.

Shared contracts may contain:

- request DTOs
- response DTOs
- events
- enums
- data models/value-like transport structures

Examples:

    StartTrainingRequest
    StopTrainingRequest

    TrainingConfiguration
    TrainingProgress
    CheckpointMetadata
    PerformanceMetrics

    CheckpointType
    AgentRole

    TrainingStartedEvent
    TrainingCompletedEvent
    TrainingFailedEvent
    CheckpointCreatedEvent
    EvaluationCompletedEvent

Shared contracts represent communication.

They should NOT become a shared domain model.

Do not put:

- TrainingRun
- Experiment
- Scenario
- RuntimeWorker

into shared contracts merely because several services know their IDs.

---

# 12. AgentRole

AgentRole is part of the cross-service language because evaluation and
performance metrics may require agent-specific information.

Examples may include:

- Tank
- Dealer
- Healer
- Boss

The exact roles should remain extensible.

AgentRole should therefore be treated as a shared contract/domain vocabulary
where required by communication and evaluation.

Do not hard-code AgentRole assumptions into unrelated infrastructure.

---

# 13. Events

An event is an immutable record that something meaningful happened.

Events should contain information required by consumers.

Typical metadata:

- event ID
- occurred-at timestamp
- run identity where applicable
- relevant event payload

Examples:

    TrainingInitializedEvent
    TrainingStartedEvent
    TrainingPausedEvent
    TrainingResumedEvent
    TrainingStoppedEvent
    TrainingCompletedEvent
    TrainingFailedEvent

    CheckpointCreatedEvent

    EvaluationStartedEvent
    EvaluationCompletedEvent

The engine publishes facts.

It does not need to know who consumes them.

---

# 14. Event Flow

Conceptually:

    Engine
      |
      v
    Event Publisher
      |
      v
    Event Channel
      |
      +---- Training Service
      |
      +---- Evaluation
      |
      +---- Telemetry
      |
      +---- Logging
      |
      +---- Future consumers

The initial implementation may be in-memory.

Future implementations may use:

- Redis
- Kafka
- NATS
- RabbitMQ
- gRPC streams
- other messaging systems

Consumers should not require the Engine to know their implementation.

---

# 15. Gateway / SDK

The Engine Gateway is not the Engine.

It is the communication layer between the control plane and Engine Service.

Conceptually:

    Experimentation / Control Plane
             |
        Engine SDK
             |
       HTTP / gRPC
             |
        Engine Service

The SDK should:

- expose typed operations
- serialize DTOs
- communicate over the service boundary
- deserialize responses
- hide transport details from application use cases

The SDK should not contain engine business logic.

---

# 16. Telemetry / Data Collection

Telemetry answers:

> What happened during execution?

It collects raw execution data.

Potential data:

- observations
- actions
- rewards
- trajectories
- sensor data
- logs
- runtime metrics
- events
- timing information
- resource information

TrainingRun should not become a storage container for all execution data.

Raw data should be handled by the appropriate data/telemetry subsystem.

---

# 17. Evaluation

Evaluation answers:

> How well did the execution perform?

Evaluation consumes execution output and produces meaningful measurements.

Examples:

- reward
- win rate
- episode length
- success rate
- collision rate
- trajectory error
- energy consumption
- latency
- agent-specific performance

Evaluation is conceptually separate from execution.

The Engine may execute evaluation workloads, but the meaning and tracking of
evaluation results belong outside the runtime implementation.

---

# 18. Analytics

Analytics operates above raw metrics.

Example:

    Evaluation:
        win_rate = 0.73

    Analytics:
        performance decreased under increased sensor noise

Analytics combines metrics, telemetry, experiment context, and historical
results.

---

# 19. Research Intelligence

Research Intelligence operates above analytics.

It may eventually use multi-agent AI systems to:

- inspect experiments
- compare results
- identify correlations
- summarize findings
- propose hypotheses
- suggest experiments
- modify configurations
- generate reports
- assist with implementation

Research Intelligence must remain outside the execution engine.

The engine executes.

Research Intelligence reasons.

---

# 20. Artifact / Dataset Management

Research artifacts need provenance and reproducibility.

Potential artifacts:

- checkpoints
- datasets
- trajectories
- experiment outputs
- models
- configurations
- evaluation results

Artifact management should be designed independently from execution.

The exact service boundary may evolve.

---

# 21. Robotics Scope

The Engine architecture should eventually support:

1. World and scenario construction
2. Robot/agent construction
3. Physics simulation
4. Simulation execution
5. Perception
6. State estimation
7. Planning
8. Control
9. Learning
10. Multi-agent interaction
11. Experimentation and evaluation

These are capabilities.

They should not automatically become separate services.

Internal Engine modules may contain these capabilities while the Engine remains
one execution boundary.

---

# 22. Extensibility Principle

The architecture must allow new functionality without requiring unrelated
services to change.

For example, adding:

- a new simulator
- a new RL algorithm
- a new controller
- a new evaluation metric
- a new sensor
- a new robot
- a new scenario type
- a new analysis method

should primarily require adding or modifying the responsible component.

It should not require rewriting the entire platform.

Prefer:

    new capability
          |
       interface
          |
    existing boundary

over:

    new capability
          |
    modify every service

---

# 23. Important Boundary Rule

Do not optimize the architecture for hypothetical scale before the boundaries
are correct.

The platform should first support:

- local execution
- local event publishing
- local database
- local Engine Service
- local SDK communication

The interfaces should make future distribution possible.

Actual distributed infrastructure can be introduced later.

---

# 24. Current Implementation Strategy

The existing RL/engine implementation is valuable.

The architecture refactor must preserve working domain logic.

The following are NOT acceptable refactoring strategies:

- deleting working training logic
- replacing real metrics with zero placeholders
- removing existing algorithms
- replacing environment behavior with stubs
- silently changing business behavior
- inventing unsupported domain rules

Refactoring should separate responsibilities while preserving behavior.

---

# 25. Architecture Evolution

This architecture is intentionally evolutionary.

A future requirement may change the implementation.

For example:

    Local Process
          |
          v
    Ray Actor
          |
          v
    Kubernetes Worker

The Runtime Worker abstraction should remain stable.

Similarly:

    In-memory Events
          |
          v
    Redis
          |
          v
    Kafka / NATS

The event contract should remain stable.

Architecture should absorb implementation changes behind boundaries.

---

# 26. Guiding Principle

The goal is not to predict the final architecture.

The goal is to create boundaries that allow the architecture to evolve without
destroying existing work.

When uncertain:

    preserve working logic
    preserve domain meaning
    isolate responsibilities
    introduce abstractions only where useful
    avoid premature infrastructure
    ask before inventing behavior