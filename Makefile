all: draw

entropy: entropy.c
	gcc -O3 -lm -fopenmp entropy.c -o entropy

entropy.data: entropy
	./entropy > entropy.data

draw: draw.py entropy.data
	python draw.py
