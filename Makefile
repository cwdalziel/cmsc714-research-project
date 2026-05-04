CXX      = smpicxx
CXXFLAGS = -O2 -Wall
LDFLAGS  = -lfftw3 -lfftw3_mpi -lm
SRCDIR   = src
BINDIR   = bin
RESDIR   = results

SOURCES  = $(shell find $(SRCDIR) -name '*.cpp')
TARGETS  = $(patsubst $(SRCDIR)/%.cpp, $(BINDIR)/%, $(SOURCES))

.PHONY: all clean clean-all

all: $(TARGETS)

$(BINDIR)/%: $(SRCDIR)/%.cpp
	mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) -o $@ $< $(LDFLAGS)

clean-all:
	rm -rf $(BINDIR) $(RESDIR)

clean-results:
	rm -rf $(RESDIR)

clean:
	rm -rf $(BINDIR)