import sys,os; sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from hybrid import *
import statm
AB=ALLG+['base25']
if __name__=='__main__':
    for ode in sys.argv[1:]:
        hybrid(ode,'evidence',names=statm.CORE2,tag='_b25'); hybrid(ode,'search',names=statm.CORE2,tag='_b25')
        hybrid(ode,'evidence',names=AB,tag='_allb25'); hybrid(ode,'search',names=AB,tag='_allb25')
