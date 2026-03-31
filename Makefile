# Compiler and flags
CXX = g++
CXXFLAGS = -std=c++11 -Wall -O3

# Common object files
COMMON_OBJS = CTRNN.o random.o

# Executables
PROGRAMS = SimpleCTRNN SimpleSearch SimpleWalking SimpleVisual

# Default target - build all programs
all: $(PROGRAMS)

# SimpleCTRNN
SimpleCTRNN: SimpleCTRNNMain.o $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# SimpleSearch
SimpleSearch: SimpleSearchMain.o TSearch.o $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# SimpleWalking
SimpleWalking: SimpleWalkingMain.o LeggedAgent.o $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# SimpleVisual
SimpleVisual: SimpleVisualMain.o VisualAgent.o $(COMMON_OBJS)
	$(CXX) $(CXXFLAGS) -o $@ $^

# Compile .cpp to .o
%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $<

# Clean up compiled files
clean:
	rm -f *.o $(PROGRAMS)

# Phony targets
.PHONY: all clean
