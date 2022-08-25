CC = mpiicc
CFLAGS = -DMPI
DEPS = funcoes.h 
OBJ = ising.o funcoes.o

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

ising.x: $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS)

.PHONY: clean

clean:
	-rm -fr *.x *.o *~ core
	#-rm -fr *.o *.pdf *~ core 
