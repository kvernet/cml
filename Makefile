COMPILER = gcc
CFLAGS   = -O2 -fPIC -Wall -Werror -Wextra
LDFLAGS  = -shared

LIB_NAME = cml
LIB_SRCS = src/cml_activation.c src/cml_algorithm.c src/cml_data.c src/cml_layer.c src/cml_loss.c src/cml_matrix.c src/cml_optimizer.c src/cml_prng.c src/cml_sequential.c
LIB_OBJS = $(LIB_SRCS:.c=.o)

all: lib$(LIB_NAME).so

DATA_EXAMPLES = shuffle
LAYER_EXAMPLES  = leaky-relu linear new relu sigmoid softmax tanh
MATRIX_EXAMPLES = alloc det eye inv lu prod solve sum trace transpose zeros
PRNG_EXAMPLES = init normal uniform
SEQUENTIAL_EXAMPLES = and create heart-disease iris lattice-physics linreg or polyreg wdbc wine-quality xor
EXAMPLE_SRCS = $(DATA_EXAMPLES) $(LAYER_EXAMPLES) $(MATRIX_EXAMPLES) $(PRNG_EXAMPLES) $(SEQUENTIAL_EXAMPLES)

examples: $(EXAMPLE_SRCS)

define build_example
$(COMPILER) $(CFLAGS) -Iinclude examples/$1/$2.c -o bin/examples/$1-$2 -Llib -lcml -lm
endef

$(DATA_EXAMPLES): lib$(LIB_NAME).so | bin_examples
	$(call build_example,data,$@)

$(LAYER_EXAMPLES): lib$(LIB_NAME).so | bin_examples
	$(call build_example,layer,$@)

$(MATRIX_EXAMPLES): lib$(LIB_NAME).so | bin_examples
	$(call build_example,matrix,$@)

$(PRNG_EXAMPLES): lib$(LIB_NAME).so | bin_examples
	$(call build_example,prng,$@)

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