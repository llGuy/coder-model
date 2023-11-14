all:
	g++ -g -std=c++20 main.cc -o gen

clean:
	rm gen
