# PickMeUp

PickMeUp is a production-oriented research platform for building, training, analyzing and deploying autonomous multi-agent systems.

The project began as a custom Multi-Agent Reinforcement Learning environment and is evolving into a complete simulation and AI infrastructure for autonomous agents operating in complex virtual worlds. The long-term goal is to bridge reinforcement learning, simulation, robotics and game AI under a unified engineering platform.

Unlike benchmark implementations that focus only on algorithms, PickMeUp focuses equally on world simulation, scalable training infrastructure, observability, experimentation and future deployment into realistic 3D environments.

---

## Vision

Modern autonomous systems are significantly more than neural networks.

An intelligent system requires

- A physically consistent world
- High performance simulation
- Distributed training
- Reliable telemetry
- Experiment tracking
- Explainable agent behaviour
- Continuous evaluation
- Tooling for debugging and iteration

PickMeUp aims to become an end-to-end platform for developing these systems.

Future domains include

- Multi-Agent Reinforcement Learning
- Autonomous Robotics
- Mobile Robots
- Game AI
- Drone Navigation
- Warehouse Fleet Management
- Multi-Agent Planning
- Intelligent Simulation Systems

---

# Current Architecture

```
                      +-----------------------+
                      |    Training Service   |
                      +-----------+-----------+
                                  |
                    Rollouts / Batches
                                  |
               +------------------+------------------+
               |                                     |
        Shared Workers                     Simulation Engine
               |                                     |
      Shared Memory IPC                    Physics + World Logic
               |                                     |
      Observation Builder                 Action Sequencer
               |                                     |
          MAPPO / CTDE                 Reward Calculation
               |                                     |
         Policy Networks                 Environment State
```

---

# Core Features

## Custom Multi-Agent Environment

A fully custom environment built from scratch without relying on Gym physics.

Features include

- Phase-based action execution
- Turn scheduling
- Multiple character roles
- Custom combat mechanics
- Dynamic action masking
- State transition engine
- Extensible skill system
- Modular environment components

---

## Centralized Training Decentralized Execution (CTDE)

Implements the CTDE paradigm for cooperative multi-agent learning.

Current implementation includes

- Decentralized actors
- Centralized critic
- Role-specific policies
- Shared global observations
- Advantage estimation
- PPO optimization pipeline

---

## High Performance Environment Execution

Training environments execute in parallel using custom infrastructure.

Current features

- Shared Memory IPC
- Parallel environment workers
- Multiprocessing rollout collection
- Low-overhead communication
- Batched inference

Designed for scaling to significantly larger simulations.

---

## Reinforcement Learning Infrastructure

Training pipeline includes

- MAPPO implementation
- Rollout buffer
- GAE
- PPO optimization
- Value normalization
- Action masking
- Entropy scheduling
- Gradient clipping
- Model checkpointing

---

## Reward Framework

The reward engine is modular rather than task-specific.

Current capabilities

- Combat rewards
- Potential-based shaping
- Phase-based reward decomposition
- Reward annealing
- Terminal rewards
- Dense and sparse reward composition

Future work includes automated reward scheduling and curriculum learning.

---

## Simulation Engine

The environment is designed as an independent simulation engine.

Components include

- World State
- Observation Builder
- Action Sequencer
- State Operations
- Physics Layer
- Reward Engine
- Environment API

The goal is to make the simulator reusable across reinforcement learning, robotics and game AI workloads.

---

## Observability

Training is treated as a production system.

Current telemetry includes

- Episode statistics
- Reward tracking
- Entropy
- Policy loss
- Critic loss
- Win rate
- SPS
- Checkpoint metrics

Future additions

- Action timelines
- Agent reasoning visualization
- State replay
- Reward attribution
- Interactive debugging

---

## Distributed System Design

The architecture is intentionally modular.

Major components

- Simulation Engine
- Policy Service
- Training Service
- Rollout Workers
- Shared Memory Layer
- Telemetry Pipeline

The long-term objective is to evolve toward distributed training infrastructure capable of running large-scale autonomous simulations.

---

# Roadmap

## Simulation

- Continuous action spaces
- Physics simulation
- Navigation environments
- Sensor simulation
- Procedural worlds
- Terrain generation

---

## 3D Engine

Planned integration includes

- Three.js visualization
- Modern rendering pipeline
- Interactive simulation playback
- Real-time debugging
- 3D character systems

---

## Robotics

Future robotics capabilities

- ROS2 integration
- LiDAR simulation
- SLAM environments
- Sensor fusion
- Path planning
- Motion planning
- Autonomous navigation

---

## Agentic Training Platform

A long-term goal is building an intelligent reinforcement learning platform where specialized AI agents assist throughout the training lifecycle.

Planned capabilities include

- Training diagnostics
- Automatic hyperparameter suggestions
- Reward analysis
- Failure detection
- Policy regression detection
- Experiment comparison
- Automatic code generation
- Automated bug localization

---

## Research Areas

Current interests driving development include

- Multi-Agent Reinforcement Learning
- World Modeling
- Coordination
- Credit Assignment
- Curriculum Learning
- Autonomous Robotics
- Simulation Systems
- Distributed Reinforcement Learning
- Large Scale Training Infrastructure

---

# Technology Stack

### Languages

- Python
- C++
- TypeScript (planned)

### Machine Learning

- PyTorch
- NumPy

### Backend

- FastAPI
- Shared Memory IPC
- Multiprocessing

### Frontend (Planned)

- React
- Three.js
- WebSockets

### Infrastructure

- Docker
- Linux
- CUDA
- TensorBoard

### Future Integrations

- ROS2
- Isaac Sim
- Gazebo
- OpenCV

---

# Philosophy

The objective of PickMeUp is not only to train agents that solve a benchmark.

The objective is to build the engineering infrastructure required to develop intelligent autonomous systems—from simulation and distributed training to robotics deployment and large-scale experimentation.

The project treats reinforcement learning as one component of a much larger autonomous systems stack.
