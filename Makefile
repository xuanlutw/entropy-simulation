all: entropy.data plot 

entropy: entropy.c
	gcc -O3 -lm -fopenmp entropy.c -o entropy

entropy.data: entropy.c entropy
	./entropy $(n0) $(eps) $(N) $(R) > entropy.data

plot: plot.py
	python plot.py

clean:
	rm entropy entropy.data
