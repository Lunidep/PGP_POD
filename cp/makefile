TARGET = cp
CC = nvcc
CFLAGS = -o

.PHONY: all clean run

all: $(TARGET)

$(TARGET): cp.cu
	$(CC) $(CFLAGS) $@ $<

clean:
	rm -rf res/
	rm -f $(TARGET)

run: $(TARGET)
	mkdir -p res
	./$(TARGET) < test
	cd res && \
	for file in $$(ls *.data); do python ../conv.py $$file $${file%.data}.png; done  && \
  	rm *.data
