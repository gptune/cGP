for i in {131..220..1}
do
    python cGP.py -p 10 -s 30 -r $i -f f_4 -o 'f_4_'$i'_TRUECL' -d f_4 -C TRUECL -g 0 -N TRUECL_classify
    python cGP.py -p 10 -s 30 -r $i -f f_4 -o 'f_4_'$i'_KMeans1' -d f_4 -C KMeans1 -g 0
    
	python cGP.py -p 10 -s 30 -r $i -f f_4 -o 'f_4_'$i'_KMeans2' -d f_4 -C KMeans2 -g 0
	python cGP.py -p 10 -s 30 -r $i -f f_4 -o 'f_4_'$i'_KMeans3' -d f_4 -C KMeans3 -g 0
	python cGP.py -p 10 -s 30 -r $i -f f_4 -o 'f_4_'$i'_KMeans4' -d f_4 -C KMeans4 -g 0
	python cGP.py -p 10 -s 30 -r $i -f f_4 -o 'f_4_'$i'_KMeans5' -d f_4 -C KMeans5 -g 0
	
	python cGP.py -p 10 -s 30 -r $i -f f_4 -o 'f_4_'$i'_DGM2' -d f_4 -C DGM2 -g 0
	python cGP.py -p 10 -s 30 -r $i -f f_4 -o 'f_4_'$i'_DGM3' -d f_4 -C DGM3 -g 0
	python cGP.py -p 10 -s 30 -r $i -f f_4 -o 'f_4_'$i'_DGM4' -d f_4 -C DGM4 -g 0
	
done

