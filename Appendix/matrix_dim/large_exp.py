import sys
import runpy
import numpy as np
file_object = open('recordings_100.txt', 'a')
for RND_SEED in np.linspace(121,220,100):
    for SEQUENTIAL_SIZE in [10,30,50,70,90]:
        file_path2 = 'GP_mod.py'
        sys.argv = ['', RND_SEED,SEQUENTIAL_SIZE,'1',1,'1','1']
        s2 = runpy.run_path(file_path2)
        rec2 = np.array(['GP',RND_SEED,SEQUENTIAL_SIZE,1,1,int(s2['sample_max_x']),s2['sample_max_f'][0],str(s2['rdstr']),int(s2['final_N_COMPONENTS'])])
        print(rec2)
        file_object.write(np.array2string(rec2))
        file_object.write('\n')
        for EXPLORATION_RATE in [0.5,0.8,1]:
                for N_NEIGHBORS in [3]:
                    file_path1 = 'cGP_mod.py'
                    sys.argv = ['', RND_SEED,SEQUENTIAL_SIZE,EXPLORATION_RATE,0,'3',N_NEIGHBORS]
                    #'' means we do not pass argv[0]
                    #'123' means we pass a random seed
                    #'10' means thee number of sequential samples we need
                    #'1' means EXPLORATION_RATE
                    #'False' means we want some clusters, 'cGP'
                    #'3' means how many maximal components we can fit during the fitting-sampling process.
                    #'3' means how many neighbors shall we look into when doing the classification step.
                    s1 = runpy.run_path(file_path1)


                    print(s1['sample_max_f'])
                    print(s1['sample_max_x'])
                    print(int(s1['sample_max_x']))

                    rec1 = np.array(['cGP',RND_SEED,SEQUENTIAL_SIZE,EXPLORATION_RATE,N_NEIGHBORS,int(s1['sample_max_x']),s1['sample_max_f'][0],str(s1['rdstr']),int(s1['final_N_COMPONENTS'])])

                    print(rec1)

                    file_object.write(np.array2string(rec1))
                    file_object.write('\n')

file_object.close()
