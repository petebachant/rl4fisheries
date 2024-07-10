# Importing from sub-directories here makes these available as 'top-level' imports
from rl4fisheries.envs.asm_env import AsmEnv
from rl4fisheries.envs.asm_esc import AsmEnvEsc
from rl4fisheries.envs.asm_cr_like import AsmCRLike

from rl4fisheries.agents.cautionary_rule import PrecautionaryPrinciple
from rl4fisheries.agents.const_esc import ConstantEscapement
from rl4fisheries.agents.msy import FMsy
from rl4fisheries.agents.const_act import ConstantAction


from gymnasium.envs.registration import register
# action is fishing intensity
register(id="AsmEnv", entry_point="rl4fisheries.envs.asm_env:AsmEnv")
# action is 'escapement'
register(id="AsmEnvEsc", entry_point="rl4fisheries.envs.asm_esc:AsmEnvEsc")
# CR-like actions
register(id="AsmCRLike", entry_point="rl4fisheries.envs.asm_cr_like:AsmCRLike")

