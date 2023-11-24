src := $(shell find . -name "*.cc")
obj := $(patsubst %.cc, %.o, $(src))

all: inspect gen

$(obj): %.o : %.cc
	g++ --std=c++20 -c $< -o $@ -g

inspect: inspect.o datalib.o
	g++ --std=c++20 $^ -o inspect

gen: main.o datalib.o
	g++ --std=c++20 $^ -o gen

clean:
	rm gen inspect
	rm ./*.o
