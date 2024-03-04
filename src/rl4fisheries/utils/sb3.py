import gymnasium as gym

from sb3_contrib import TQC, ARS
from stable_baselines3 import PPO, A2C, DQN, SAC, TD3
from stable_baselines3.common.env_util import make_vec_env

def load_sb3_agent(algo, env, weights, policy = "MlpPolicy"):
    ALGO = algorithm(algo)
    model = ALGO(policy, env)
    agent = model.load(weights)
    return agent

# dictionary look up as a function...
def algorithm(algo):
    algos = {
        'RecurrentPPO': RecurrentPPO,
        'RPPO': RecurrentPPO,
        'recurrentppo': RecurrentPPO,
        'rppo': RecurrentPPO,
        #
        "PPO": PPO,
        "ARS": ARS,
        "TQC": TQC,
        "A2C": A2C,
        "SAC": SAC,
        "DQN": DQN,
        "TD3": TD3,
        "ppo": PPO,
        "ars": ARS,
        "tqc": TQC,
        "a2c": A2C,
        "sac": SAC,
        "dqn": DQN,
        "td3": TD3,
    }
    return algos[algo]

def sb3_train(config_file, **kwargs):
    with open(config_file, "r") as stream:
        options = yaml.safe_load(stream)
        options = {**options, **kwargs}
        # updates / expands on yaml options with optional user-provided input

    if "n_envs" in options:
        env = make_vec_env(
            options["env_id"], options["n_envs"], env_kwargs={"config": options["config"]}
        )
    else:
        env = gym.make(options["env_id"])
    ALGO = algorithm(options["algo"])
    if "id" in options:
        options["id"] = "-" + options["id"]
    model_id = options["algo"] + "-" + options["env_id"]  + options.get("id", "")
    save_id = os.path.join(options["save_path"], model_id)

    model = ALGO(
        options.get("policyType", "MlpPolicy"),
        env,
        verbose=0,
        tensorboard_log=options["tensorboard"],
        **{opt: options[opt] for opt in options if opt in ['use_sde']}, # oof, something nicer soon?
    )

    progress_bar = options.get("progress_bar", False)
    model.learn(total_timesteps=options["total_timesteps"], tb_log_name=model_id, progress_bar=progress_bar)

    os.makedirs(options["save_path"], exist_ok=True)
    model.save(save_id)
    print(f"Saved {options['algo']} model at {save_id}")
    
    return save_id, options
