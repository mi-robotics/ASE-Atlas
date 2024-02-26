import torch 

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
from rl_games.algos_torch import players
import learning.ase_players as ase_player

class CuriASEPlayerContinuous(ase_player.ASEPlayer):
    def __init__(self, config):
        super().__init__(config)
        
        return