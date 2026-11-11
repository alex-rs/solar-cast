from config.settings import Settings


def test_csv_environment_lists(monkeypatch):
    monkeypatch.setenv("OPENMETEO_MODELS", "alpha,beta,gamma")
    monkeypatch.setenv("OPENMETEO_WEIGHTS", "0.5,0.3,0.2")
    settings = Settings()
    assert settings.openmeteo_models == ["alpha", "beta", "gamma"]
    assert settings.openmeteo_weights == [0.5, 0.3, 0.2]
