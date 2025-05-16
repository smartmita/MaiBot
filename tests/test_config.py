from src.config.config import global_config


class TestConfig:
    def test_load(self):
        config = global_config
        print(config)
