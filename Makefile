all: main.o

clean:
	rm -rf *.o

debug:
	g++ main.c -o dbg.o -I/home/jamie/Repos/matplotlib-cpp -I/usr/include/python2.7 -lpython2.7 -std=c++11 -g -fstack-protector-strong

main.o: main.c kalman.h mat_ops.h svd.h
	g++ main.c -o main.o -I/home/jamie/Repos/matplotlib-cpp -I/usr/include/python2.7 -lpython2.7 -std=c++11

run: main.o
	./main.o
