# R2R Documentation

This directory contains all project documentation.

## 📋 Product & Planning

- **[PRD.md](PRD.md)** - Product Requirements Document
  - Full product specification
  - Success metrics and goals
  - Users, use cases, and personas
  - Technical architecture overview

## 🏗️ Architecture & Design

- **[MODULE_STRUCTURE.md](MODULE_STRUCTURE.md)** - Code organization and module layout
- **[PORTING_SUMMARY.md](PORTING_SUMMARY.md)** - Notebook → Production porting details
  - ThesisPipeline architecture
  - Training flow (SFT → RFT)
  - Design decisions and rationale

## 🛠️ Setup & Development

- **[CODE_QUALITY_SETUP.md](CODE_QUALITY_SETUP.md)** - Code quality tools configuration
  - Pre-commit hooks
  - Linting and type checking
  - Testing and coverage
- **[QUICKREF.md](QUICKREF.md)** - Quick reference guide

- **[SETUP_STATUS.md](SETUP_STATUS.md)** - Initial setup tracking

- **[environment.yml](environment.yml)** - Conda environment (archived, using pip)

## 📝 Live Documentation

For the most current information, see the root-level files:

- **[../README.md](../README.md)** - Main project README
- **[../CONTRIBUTING.md](../CONTRIBUTING.md)** - Development workflow
- **[../TODO.md](../TODO.md)** - Current priorities and roadmap

## 📓 Notebooks

Reference notebooks are in [`../notebooks/`](../notebooks/):

- `Tiny_Trading_R1_RankingRL.ipynb` - Pairwise ranking RL implementation
- `Tiny_Trading_R1_SchemaValidation.ipynb` - Schema validation approach

## 🗂️ Configuration

Configuration files are in [`../configs/`](../configs/):

- `base_config.yaml` - Base configuration
- `experiments/` - Experiment-specific configs

## 📐 Schemas

JSON schemas are in [`../schemas/`](../schemas/):

- `thesis.schema.json` - Thesis output format
- `data_manifest.schema.json` - Data manifest format
