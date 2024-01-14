CXX = icpx
CXXFLAGS = -fsycl -std=c++20 -w
#CXXFLAGS = -fsycl -fsycl-targets=nvptx64-nvidia-cuda --cuda-gpu-arch=sm_89 -fdenormal-fp-math=ieee
SRCS := $(wildcard *.cpp)
PRGS := $(patsubst %.cpp,%,$(SRCS))
RUN :=  $(addprefix run_, $(PRGS))

.PHONY: run_all
run_all: $(sort $(RUN))

.PHONY: all
all: $(PRGS)

.PHONY: all
%: %.cpp
	$(CXX) $(CXXFLAGS) -o $@ $^

run_%:%
	./$^

.PHONY: clean
clean:
	rm -f $(PRGS)