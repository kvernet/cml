COMPILER = gcc
CFLAGS   = -O2 -fPIC -Wall -Werror -Wextra
LDFLAGS  = -shared

LIB_NAME = cml
LIB_SRCS = src/cml_layer.c src/cml_matrix.c src/cml_sequential.c
LIB_OBJS = $(LIB_SRCS:.c=.o)

all: lib$(LIB_NAME).so

LAYER_EXAMPLES  = new
MATRIX_EXAMPLES = alloc det eye inv lu prod solve sum trace transpose zeros
SEQUENTIAL_EXAMPLES = create reg
EXAMPLE_SRCS = $(LAYER_EXAMPLES) $(MATRIX_EXAMPLES) $(SEQUENTIAL_EXAMPLES)

examples: $(EXAMPLE_SRCS)

define build_example
$(COMPILER) $(CFLAGS) -Iinclude examples/$1/$2.c -o bin/examples/$1-$2 -Llib -lcml
endef

$(LAYER_EXAMPLES): lib$(LIB_NAME).so | bin_examples
	$(call build_example,layer,$@)

$(MATRIX_EXAMPLES): lib$(LIB_NAME).so | bin_examples
	$(call build_example,matrix,$@)

$(SEQUENTIAL_EXAMPLES): lib$(LIB_NAME).so | bin_examples
	$(call build_example,sequential,$@)

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