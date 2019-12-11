import argparse
import gin
from shutil import copyfile
import datetime
import os


#This file documents a very basic setup to configure experiments and achieve some level of traceability.
#Arg parser is used to obtain a configuration file path, a base path for experiments, and a moniker for 
#the experiment being run

#The code being 'configured' is the run_experiment function. Classes can also be configured. 
#What the entry point for our experiments will be like requires a little architecture.

#
def parse_args():
  parser = argparse.ArgumentParser(description='Perform an experiment')
  parser.add_argument('--ginfile', default='./experiment_configurations/config_standard_size_experiment.gin')
  parser.add_argument('--experiment_folder', default='./experiments/')
  parser.add_argument('--experiment_name', default='experiment')
  return parser.parse_args()

#I recommend setting default values for configurable values - otherwise
#code inspection will flag the call sites as missing arguments.
@gin.configurable
def run_experiment(nodes=2, alpha=0.1):  
  return nodes, alpha

if __name__=="__main__":
    args = parse_args()

    #Read experiment parameters    
    gin_file = args.ginfile
    vals = gin.parse_config_file(gin_file)

    #Create experiment folder
    if not os.path.isdir(args.experiment_folder):
      os.mkdir(args.experiment_folder)

    experiment_folder = args.experiment_folder + args.experiment_name + str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '').replace('.', '') + '/'
    os.mkdir(experiment_folder)

    #Copy gin file to experiment folder (for logging)
    #Basename has some quirks on windows, so might want to adjust
    copyfile(gin_file, experiment_folder + os.path.basename(gin_file))
    nodes, alpha = run_experiment()
    
    print('Nodes: %d, alpha: %d' % (nodes, alpha))
