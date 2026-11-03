# Embodied AI Sandbox

**An experimental laboratory for exploring embodied intelligence, physical AI, and autonomous agents in simulated environments.**

[![Python](https://img.shields.io/badge/Python-3.14-blue)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/Simulation-MuJoCo-orange)](https://mujoco.org/)
[![Status](https://img.shields.io/badge/Status-Active%20Development-brightgreen)]()

## Overview

The Embodied AI Sandbox is an ongoing hands-on project exploring how intelligent agents can perceive, understand, and interact with physical environments.

Built using MuJoCo, a physics-based simulation engine, the project aims to develop an experimental environment where agents can observe objects, reason about goals, execute actions, respond to feedback, and recover from failures.

The long-term objective is to explore the intersection of artificial intelligence, robotics, simulation, and autonomous decision-making.

## Current Implementation

The initial simulation environment includes:

- A 3D physics-based environment powered by MuJoCo.
- A ground plane and gravity-based physics.
- Three independently simulated coloured cubes.
- A designated target zone.
- A native 3D simulation viewer.

This environment serves as the foundation for subsequent perception, planning, and control experiments.

## Development Roadmap

The project will evolve through the following experimental stages:

- [x] Set up the Python development environment.
- [x] Install and configure MuJoCo.
- [x] Launch the native 3D simulation viewer.
- [x] Create an initial physics environment with objects and a target zone.
- [ ] Implement environment-state observation.
- [ ] Introduce goal specification and task evaluation.
- [ ] Develop an initial autonomous control mechanism.
- [ ] Implement object manipulation and target-reaching tasks.
- [ ] Introduce visual perception.
- [ ] Explore planning and decision-making.
- [ ] Experiment with disturbances, failure detection, and recovery.
- [ ] Measure task success rates and performance.
- [ ] Explore the possibility of transferring selected concepts to physical robotic platforms.

## Technology Stack

- Python
- MuJoCo
- NumPy
- VS Code
- Git and GitHub

## Getting Started

### Prerequisites

- Python 3.14 or a compatible Python version.
- macOS for the current native viewer workflow.
- Git.

### Installation

Clone the repository:

```bash
git clone https://github.com/vsroshan/embodied-ai-sandbox.git
cd embodied-ai-sandbox
```

Create a virtual environment:

```bash
python3 -m venv .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Run the simulation

On macOS, launch the native MuJoCo viewer using:

```bash
./.venv/bin/mjpython main.py
```

## Experimental Philosophy

This project follows an iterative, experiment-driven approach:

1. Build a working environment.
2. Observe and understand its state.
3. Define a measurable objective.
4. Implement an action or control strategy.
5. Evaluate the outcome.
6. Introduce complexity and investigate failures.

The emphasis is on understanding how intelligence can translate into actions within a physical environment, rather than treating AI as a text-generation problem alone.

## Project Status

**Active Development — Experimental Prototype**

This repository documents the development process, experiments, implementation decisions, and findings as the project evolves.

The current implementation is a simulated environment. It does not yet represent a complete autonomous agent or a validated real-world robotic system.

## Author

Developed as an independent learning and experimentation project focused on AI, embodied intelligence, robotics, and emerging physical AI systems.

**More experiments and updates coming soon.**
