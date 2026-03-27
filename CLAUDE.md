# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**NEVER** update this file during a working session, we have other files to track project learnings and documentation references.

## Project Overview

This project investigates whether the standard sigmoid activation function is sufficient for neural network training, or whether a **ScaledSigmoid** variant (`scale * sigmoid(x) + shift`) can improve convergence speed and weight behavior. It compares both activations across multiple test cases (threshold classification, pulse wave detection) using PyTorch, tracking loss convergence and weight magnitude over time.

## Specialized Sub-Agents Available

**ALWAYS**
1. Use the appropriate specialized sub-agents available for the task being worked on.
2. Provide the specialized sub-agents with the current working session goal.
3. Run the code review agent after each code change and have the appropriate coding agents fix any issues found.

## Important Reference Files

- Starting point to understand the project is: [docs/README.md](docs/README.md)
- Important lessons learned and pitfalls to avoid: [docs/LESSONS.md](docs/LESSONS.md)
