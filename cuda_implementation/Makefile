TARGET = tc_cuda
SRC = $(TARGET).cu

build:
	nvcc $(SRC) -o $(TARGET).out

test:
	./$(TARGET).out

run: build test

clean:
	rm -f $(TARGET).out

.PHONY.: build
