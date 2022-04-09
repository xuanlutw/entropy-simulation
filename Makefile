all: plot

entropy: entropy.c
	gcc -O3 -lm -fopenmp entropy.c -o entropy

entropy.data: entropy
	./entropy > entropy.data

plot: plot.py entropy.data
	python plot.py
