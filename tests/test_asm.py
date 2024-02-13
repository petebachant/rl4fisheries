# Confirm environment is correctly defined:
from stable_baselines3.common.env_checker import check_env


def test_AsmEsc():
    check_env(AsmEsc(), warn=True)
