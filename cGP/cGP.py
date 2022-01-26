import os
import pkg_resources
from optparse import OptionParser

usage = "Usage: cGP.py -p <N_PILOT> -s <N_SEQUENTIAL> [optional parameters and options]"

parser = OptionParser(usage)
parser.add_option ("-p", "--N_PILOT", action="store", type="int", dest="N_PILOT",
               help="integer >=2. The number of pilot samples sampled by the designated sampler as pilots. \n compulsory.")
parser.add_option ("-s", "--N_SEQUENTIAL", action="store", type="int", dest="N_SEQUENTIAL",
               help="integer >=2. The number of sequential samples sampled by the cGP surrogate model. \n compulsory.")            
parser.add_option("-c", "--N_COMPONENT", action="store", type="int", dest="N_COMPONENT",default=4,
              help="integer >=1. The *maximal* number of components allowed in cGP surrogate model. \n optional, default=4")
parser.add_option("-C", "--CLUSTER", action="store", type="string", dest="CLUSTER",default=None,
              help="string. Provide the file name of the external cluster definition .py file, no extension. *This overrides -c option. \n optional, default='cluster'")         
parser.add_option("-n", "--N_NEIGHBORS", action="store", type="int", dest="N_NEIGHBORS",default=3,
              help="integer >=1. The number k of neighbors used for k-NN classify step (to partition) in cGP surrogate model. \n optional, default=3")
parser.add_option("-N", "--CLASSIFY", action="store", type="string", dest="CLASSIFY",default=None,
              help="string. Provide the file name of the external classify definition .py file, no extension. *This overrides -n option. \n optional, default='classify'")                  
parser.add_option ("-e", "--EXPLORATION_RATE", action="store", type="float", dest="EXPLORATION_RATE",default=1.0,
               help="float [0,1].The exploration rate for sampling step (probability to maximize the acquisition) in cGP surrogate model. \n optional, default=1.0")        
parser.add_option("-g","--NO_CLUSTER",action="store", type="int", dest="NO_CLUSTER", default=0,
              help= "bool 0/1. If 1(True), then we fit a simple GP surrogate model, ignoring all other options. \n optional, default=0(False)")
parser.add_option ("-f", "--f_truth", action="store", type="string", dest="f_truth_py",default='f_truth',
               help="string. Provide the objective function python script. \n *By default this is defined by f_truth.py in the same folder, only an absolute path is allowed. \n optional, default=f_truth")   
parser.add_option ("-r", "--RND_SEED", action="store", type="int", dest="RND_SEED",default=123,
               help="integer. Provide a random seed. Must be integer\n optional, default=123")
parser.add_option ("-o", "--OUTPUT_NAME", action="store", type="string", dest="OUTPUT_NAME",default=None,
               help="string. Provide the file name that stores the log and resulting samples. \n optional, default=None (automatic generated)")  
parser.add_option ("-d", "--OBJ_NAME", action="store", type="string", dest="OBJ_NAME",default=None,
               help="string. Provide the file name that stores the fitted surrogate model. \n optional, default=None")                
parser.add_option("-F", "--PILOT_FILE", action="store", type="string", dest="PILOT_FILE", default=None,
               help="string(with extension). Provide the file name of the external pilot samples. \n *This overrides -s and -S options. This could be used to resume a paused file. \n optional, default=None (use N_PILOT option)")
parser.add_option("-A", "--ACQUISITION", action="store", type="string", dest="ACQUISITION", default='expected_improvement',
               help="string(with _). Provide the full acquisition function name of the surrogate model. optional, default=expected_improvement")
parser.add_option("-S", "--SAMPLER", action="store", type="string", dest="SAMPLER",default=None,
              help="[NO support for constraints] Choose a sampling method for the pilot samples. \n Currently supports random/latin/sobol.\n optional, default=None (use built-in sampling)")
parser.add_option ("-P", "--N_PROC", action="store", type="int", dest="N_PROC", default=1,
               help="integer. Provide the number of processors used for multiprocessing speed-up.\n optional, default=1")   
parser.add_option ("-V", "--VERSION_DEPENDENCIES", action="store", type="int", dest="VERSION_DEPENDENCIES", default=0,
               help="bool 0/1. If 1(True), then we check the dependencies/packages needed by cGP before we execute the main code.\n optional, default=0(False)")                
parser.add_option("-X", "--QUIET", action='store_true', dest='QUIET',
               help="bool 0/1. If not None, all output would be suppressed.")                                           
           
options, args = parser.parse_args()

if options.VERSION_DEPENDENCIES > 0:
	#testing whether packages are satisfied
	with open('requirements.txt') as f:
	    content = f.readlines()
	dependencies = [x.strip() for x in content] 
	print('cGP dependencies:\n',dependencies)
	pkg_resources.require(dependencies)

#print(options, args)
if options.N_SEQUENTIAL is None:
	raise Exception('ERROR: N_SEQEUNTIAL(int) is a compulsory parameter that must be supplied. \n e.g. -s 20')
if options.PILOT_FILE is None and options.N_PILOT is None:
	raise Exception('ERROR: Either N_PILOT(int) or PILOT_FILE(string, with extension) must be supplied. \n e.g. -p 10 OR -F samples_sobol_10.txt')
#if len(PILOT_NAME)>0:
#	#We provide a pilot sample file.
#else:
#	#We use the chosen sampler for pilot sampling.

#Get a random string stamp for this specific sampler, used for the filename of image export.
if options.SAMPLER is not None: 
	import random
	import string
	def get_random_string(length):
	    return ''.join(random.choice(string.ascii_lowercase) for i in range(length))
	rdstr=get_random_string(24)
	print('random stamp for this sampler:',rdstr)
	import importlib
	sampler_mdl = importlib.import_module(options.SAMPLER)
	sampler_mdl_names = [x for x in sampler_mdl.__dict__ if not x.startswith("_")]
	globals().update({k: getattr(sampler_mdl, k) for k in sampler_mdl_names})
	SAMPLER_NAME = 'sampler_'+rdstr+'.tmp'
	
	f_truth_mdl = importlib.import_module(str(options.f_truth_py))
	f_truth_names = [x for x in f_truth_mdl.__dict__ if not x.startswith("_")]
	globals().update({k: getattr(f_truth_mdl, k) for k in f_truth_names})
	f_truth_bounds = get_bounds(1)	

	get_pilots(options.RND_SEED,options.N_PILOT,SAMPLER_NAME,{'bounds':f_truth_bounds})
	#import os
	from stat import S_IREAD
	# Replace the first parameter with your file name
	os.chmod(SAMPLER_NAME, S_IREAD)
	# Send it into cGP script
	options.PILOT_FILE = SAMPLER_NAME
	print('Successfully generate pilot samples: ',options.PILOT_FILE,' with ',options.N_PILOT,' samples.\n')

if options.CLUSTER is not None: 
	my_CLUSTER = str(options.CLUSTER)
else:
	my_CLUSTER = str(options.N_COMPONENT)
if options.CLASSIFY is not None: 
	my_CLASSIFY = str(options.CLASSIFY)
else:
	my_CLASSIFY = str(options.N_NEIGHBORS)		
#Do we want to use the parallel or single processing version?    	
if options.N_PROC<=1 or options.NO_CLUSTER==1 or options.N_COMPONENT==1:
	#cGP_constrained
	if options.PILOT_FILE is None:
		cmd = 'python cGP_constrained.py '+str(options.RND_SEED)+' '+str(options.N_SEQUENTIAL)+' '+str(options.EXPLORATION_RATE)+' '+str(options.NO_CLUSTER)+' '+str(my_CLUSTER)+' '+str(my_CLASSIFY)+" "+str(options.N_PILOT)+" \'"+str(options.f_truth_py)+"\' "+"\'"+str(options.ACQUISITION)+"\' "
	else:
		cmd = 'python cGP_constrained.py '+str(options.RND_SEED)+' '+str(options.N_SEQUENTIAL)+' '+str(options.EXPLORATION_RATE)+' '+str(options.NO_CLUSTER)+' '+str(my_CLUSTER)+' '+str(my_CLASSIFY)+" "+str(options.PILOT_FILE)+" \'"+str(options.f_truth_py)+"\' "+"\'"+str(options.ACQUISITION)+"\' "
	if options.OUTPUT_NAME is not None:
		cmd = cmd +"\'"+str(options.OUTPUT_NAME)+"\' " 
else:
	#cGP_parallel
	if options.PILOT_FILE is None:
		cmd = 'python cGP_parallel.py '+str(options.RND_SEED)+' '+str(options.N_SEQUENTIAL)+' '+str(options.EXPLORATION_RATE)+' '+str(options.NO_CLUSTER)+' '+str(my_CLUSTER)+' '+str(my_CLASSIFY)+" "+str(options.N_PILOT)+" "+str(options.N_PROC)+" \'"+str(options.f_truth_py)+"\' "+"\'"+str(options.ACQUISITION)+"\' "
	else:
		cmd = 'python cGP_parallel.py '+str(options.RND_SEED)+' '+str(options.N_SEQUENTIAL)+' '+str(options.EXPLORATION_RATE)+' '+str(options.NO_CLUSTER)+' '+str(my_CLUSTER)+' '+str(my_CLASSIFY)+" "+str(options.PILOT_FILE)+" "+str(options.N_PROC)
if options.OBJ_NAME is None and options.OUTPUT_NAME is not None:
	cmd = cmd +"\'"+str(options.OUTPUT_NAME)+"\' "
if options.OBJ_NAME is not None and options.OUTPUT_NAME is not None:
	cmd = cmd +"\'"+str(options.OUTPUT_NAME)+"\' "+"\'"+str(options.OBJ_NAME)+"\'"
if options.OBJ_NAME is not None and options.OUTPUT_NAME is None:
	raise Exception('ERROR: It seems that -d option is supplied but -o option is not suppliled. -d option can only be used with -o option.')
if options.QUIET is not None:
    cmd = cmd + ' >/dev/null 2>&1'
else:
    print(cmd)
os.system(cmd)
#if options.SAMPLER is not None: 
#	from stat import S_IWRITE
#	os.chmod(SAMPLER_NAME, S_IWRITE)	
#	os.remove(SAMPLER_NAME)


