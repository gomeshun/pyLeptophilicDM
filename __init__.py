from .model import Model, LeptophilicDM
from .from_taisuke.py_src import par_to_phys as par_to_phys
from .from_taisuke.py_src import g_minus_2 as g_minus_2
from .from_taisuke.py_src import vacuum_stability as vacuum_stability
import os

# Initialize
LeptophilicDM(os.path.dirname(__file__)+"/config.csv")

    
    
    
    
    
    
    
    
    
    
    
    