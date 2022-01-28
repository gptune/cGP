for i in {131..220..1}
do
	python cGP.py -p 10 -s 30 -r $i -f f_3 -o 'f_3_'$i'_KMeans1' -d f_3 -C KMeans1 -g 0
	python cGP.py -p 10 -s 30 -r $i -f f_3 -o 'f_3_'$i'_KMeans2' -d f_3 -C KMeans2 -g 0
	python cGP.py -p 10 -s 30 -r $i -f f_3 -o 'f_3_'$i'_KMeans3' -d f_3 -C KMeans3 -g 0
	python cGP.py -p 10 -s 30 -r $i -f f_3 -o 'f_3_'$i'_KMeans4' -d f_3 -C KMeans4 -g 0
	python cGP.py -p 10 -s 30 -r $i -f f_3 -o 'f_3_'$i'_KMeans5' -d f_3 -C KMeans5 -g 0
	
done

