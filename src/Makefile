CXX = g++

CXXFLAGS = -std=c++17 -O3 -DNDEBUG -Wno-deprecated -Wno-deprecated-declarations -finline-functions -foptimize-sibling-calls -ffast-math -fomit-frame-pointer -pipe -I./
%CXXFLAGS = -std=c++17 -O3 -DNDEBUG -W -Wall -Wno-deprecated -Wno-deprecated-declarations -finline-functions -foptimize-sibling-calls -ffast-math -fomit-frame-pointer -pipe -I./
%CXXFLAGS = -std=c++17 -O3 -DNDEBUG -W -Wall -Wno-deprecated
LINKFLAGS = -lm

SRCS = \
	main.cpp \
	sketch_sort.cpp \

OBJS = $(SRCS:%.cpp=%.o)

all: sketchsort

sketchsort: $(OBJS)
	$(CXX) $(CXXFLAGS) $(OTHERFLAGS) $(OBJS) $(LINKFLAGS) -o sketchsort
debug:
	make all CXXFLAGS="-ggdb -W -Wall -pedantic"

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(OTHERFLAGS) -c $<

clean:
	rm -f sketchsort *.o *~
