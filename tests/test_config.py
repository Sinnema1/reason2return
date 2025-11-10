"""Test configuration loading utilities."""


import yaml


def test_base_config_exists(config_dir):
    """Test that base.yaml exists and is valid."""
    config_path = config_dir / "base.yaml"
    assert config_path.exists(), "base.yaml not found"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    assert config is not None
    assert isinstance(config, dict)


def test_base_config_has_required_sections(config_dir):
    """Test that base config has all required sections."""
    config_path = config_dir / "base.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    required_sections = ["seed", "model", "sft", "rft", "data", "backtest", "paths"]
    for section in required_sections:
        assert section in config, f"Missing required section: {section}"


def test_experiment_configs_exist(config_dir):
    """Test that experiment configs exist."""
    exp_dir = config_dir / "experiments"
    assert exp_dir.exists(), "experiments directory not found"

    expected_experiments = [
        "exp_001_grpo_baseline.yaml",
        "exp_002_pairwise_ranking.yaml",
        "exp_003_schema_validation.yaml",
    ]

    for exp_file in expected_experiments:
        exp_path = exp_dir / exp_file
        assert exp_path.exists(), f"Experiment config not found: {exp_file}"


def test_experiment_config_has_metadata(config_dir):
    """Test that experiment configs have required metadata."""
    exp_path = config_dir / "experiments" / "exp_001_grpo_baseline.yaml"

    with open(exp_path) as f:
        config = yaml.safe_load(f)

    assert "experiment" in config
    assert "name" in config["experiment"]
    assert "description" in config["experiment"]


def test_reward_weights_sum_to_one(config_dir):
    """Test that reward weights approximately sum to 1."""
    config_path = config_dir / "base.yaml"

    with open(config_path) as f:
        config = yaml.safe_load(f)

    weights = config["rft"]["rewards"]
    total = weights["wS"] + weights["wC"] + weights["wD"]

    assert abs(total - 1.0) < 0.01, f"Reward weights sum to {total}, expected ~1.0"
