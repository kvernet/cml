COMPILER = gcc
CFLAGS   = -O2 -fPIC -Wall -Werror -Wextra
LDFLAGS  = -shared

LIB_NAME = cml
LIB_SRCS = src/cml_matrix.c
LIB_OBJS = $(LIB_SRCS:.c=.o)

all: lib$(LIB_NAME).so

MATRIX_EXAMPLES = alloc det eye inv lu prod solve sum trace transpose zeros
EXAMPLE_SRCS = $(MATRIX_EXAMPLES)

examples: $(EXAMPLE_SRCS)

define build_example
$(COMPILER) $(CFLAGS) -Iinclude examples/$1/$2.c -o bin/examples/$1-$2 -Llib -lcml
endef

$(MATRIX_EXAMPLES): lib$(LIB_NAME).so | bin_examples
	$(call build_example,matrix,$@)

bin_examples:
	mkdir -p bin/examples

lib$(LIB_NAME).so: $(LIB_OBJS) | lib
	$(COMPILER) $(LDFLAGS) $^ -o lib/$@ -lm

lib:
	mkdir -p lib

%.o: %.c
	$(COMPILER) $(CFLAGS) -Iinclude -c $< -o $@

clean:
	rm -rf bin
	rm -rf lib
	rm -f $(LIB_OBJS)