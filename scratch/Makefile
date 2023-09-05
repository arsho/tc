TARGET = tc
SRC = $(TARGET).cpp

build:
	icpx -fsycl $(SRC) -o $(TARGET).out

test:
	./$(TARGET).out

run: build test

clean:
	rm -f $(TARGET).out

.PHONY.: build
