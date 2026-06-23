import shutil
import pandas as pd
import pytest

from data import DataLoader, DataValidationError


def test_loads_real_data_successfully():
    df = DataLoader().load()
    assert len(df) == 96  # 8 lines of business x 12 weeks
    assert df["lob"].nunique() == 8
    assert df["week_ending"].nunique() == 12


def test_raises_on_missing_column(tmp_path):
    shutil.copytree("data", tmp_path / "data")
    premium_path = tmp_path / "data" / "case4_weekly_premium.csv"
    df = pd.read_csv(premium_path)
    df.drop(columns=["plan_gwp"]).to_csv(premium_path, index=False)
    with pytest.raises(DataValidationError, match="plan_gwp"):
        DataLoader(tmp_path / "data").load()


def test_raises_on_week_mismatch(tmp_path):
    shutil.copytree("data", tmp_path / "data")
    premium_path = tmp_path / "data" / "case4_weekly_premium.csv"
    df = pd.read_csv(premium_path)
    df = df[df.week_ending != df.week_ending.max()]
    df.to_csv(premium_path, index=False)
    with pytest.raises(DataValidationError, match="Week mismatch"):
        DataLoader(tmp_path / "data").load()


def test_raises_on_missing_file(tmp_path):
    with pytest.raises(DataValidationError, match="Missing input file"):
        DataLoader(tmp_path).load()


def test_hit_rate_formula():
    df = DataLoader().load()
    row = df.iloc[0]
    expected = row.bound_count / (row.bound_count + row.quoted_count + row.declined_count + row.ntu_count) * 100
    assert row.hit_rate == pytest.approx(expected)


def test_portfolio_kpis_returns_expected_keys():
    loader = DataLoader()
    loader.load()
    kpis = loader.portfolio_kpis()
    for key in ["as_of_week", "ytd_gwp_actual", "ytd_gwp_plan", "ytd_gwp_vs_plan_pct", "portfolio_hit_rate_pct"]:
        assert key in kpis
