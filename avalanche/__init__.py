import os
__version__ = "0.0.1"
os.environ['GLOG_minloglevel'] = '2'
AVALANCHE_BP = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_BP = AVALANCHE_BP + '/extras/artifacts/'
