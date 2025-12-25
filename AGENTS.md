# Agent Instructions

This project uses **astral uv** for Python package management and script execution.
Use `uv run <script>` to run scripts and `uv add <packages...>` to add dependencies.

## Coding Rules

Avoid duplicate code. Before implementing new functionality, check if similar code or references already exist in the codebase.
Reuse existing functions, classes, or modules instead of creating duplicates.

Avoid hardcoding values when possible. If hardcoding is necessary, prefer creating constants or using configuration files instead of embedding values directly in the code.

## Planning and Time Estimates

When writing plans and estimating time/effort, think from an agent's perspective, not a human's.
Tasks that might take a human 1 hour can often be completed by an agent in 1 minute.
Don't artificially constrain yourself with human-scale time estimates - think more powerfully and efficiently.
