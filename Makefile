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

# ---- Docker (consistent Linux + SimGrid env for all teammates) ----
DOCKER_IMAGE = cmsc714

.PHONY: docker-build docker-shell docker-make

docker-build:
	docker build -t $(DOCKER_IMAGE) .

docker-shell:
	docker run -it --rm -v "$(CURDIR):/work" -w /work $(DOCKER_IMAGE) bash

docker-make:
	docker run --rm -v "$(CURDIR):/work" -w /work $(DOCKER_IMAGE) make