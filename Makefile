CC ?= gcc
CFLAGS ?= -O2 -Wall -Wextra -std=c11
INCLUDES := -Iinclude
LDFLAGS := -lm

SRCDIR := src
TESTDIR := tests
BUILDDIR := build

SOURCES := $(wildcard $(SRCDIR)/*.c)
OBJECTS := $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(SOURCES))

.PHONY: all clean test static example

all: $(BUILDDIR)/libspectral_graph.a

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDES) -c $< -o $@

$(BUILDDIR)/libspectral_graph.a: $(OBJECTS)
	ar rcs $@ $^

test: $(BUILDDIR)/libspectral_graph.a
	$(CC) $(CFLAGS) $(INCLUDES) $(TESTDIR)/spectral_graph_tests.c \
		-L$(BUILDDIR) -lspectral_graph $(LDFLAGS) -o $(BUILDDIR)/test_spectral
	./$(BUILDDIR)/test_spectral

static:
	$(CC) $(CFLAGS) $(INCLUDES) --analyze $(SOURCES)

clean:
	rm -rf $(BUILDDIR)

# Debug build
debug: CFLAGS := -g -O0 -Wall -Wextra -std=c11 -DFSANITIZE
debug: clean all
