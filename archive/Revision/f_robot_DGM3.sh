for i in {1..50..1}
do
    #python cGP.py -p 10 -s 10 -e 1.0 -r $i -f f_robot -o 'robot_'$i'_DGM3_10_GP' -C DGM3 -g 1
    #python cGP.py -p 10 -s 30 -e 1.0 -r $i -f f_robot -o 'robot_'$i'_DGM3_30_GP' -C DGM3 -g 1
    #python cGP.py -p 10 -s 50 -e 1.0 -r $i -f f_robot -o 'robot_'$i'_DGM3_50_GP' -C DGM3 -g 1
    #python cGP.py -p 10 -s 70 -e 1.0 -r $i -f f_robot -o 'robot_'$i'_DGM3_70_GP' -C DGM3 -g 1
    python cGP.py -p 10 -s 90 -e 1.0 -r $i -f f_robot -o 'robot_'$i'_DGM3_90_GP' -C DGM3 -g 1
    
	for exprate in 0.5 0.8 1.0
    do
        
	    #python cGP.py -p 10 -s 10 -e $exprate -r $i -f f_robot -o 'robot_'$i'_DGM3_10_cGP_'$exprate -C DGM3 -g 0
	    #python cGP.py -p 10 -s 30 -e $exprate -r $i -f f_robot -o 'robot_'$i'_DGM3_30_cGP_'$exprate -C DGM3 -g 0
	    #python cGP.py -p 10 -s 50 -e $exprate -r $i -f f_robot -o 'robot_'$i'_DGM3_50_cGP_'$exprate -C DGM3 -g 0
	    #python cGP.py -p 10 -s 70 -e $exprate -r $i -f f_robot -o 'robot_'$i'_DGM3_70_cGP_'$exprate -C DGM3 -g 0
	    python cGP.py -p 10 -s 90 -e $exprate -r $i -f f_robot -o 'robot_'$i'_DGM3_90_cGP_'$exprate -C DGM3 -g 0
	done
done

