import sys
import random

def generate_sequence(length):
    return [random.randint(0, 100) for _ in range(length)]

def save_to_binary_file(sequence, filename):
    with open(filename, 'wb') as file:
        for num in sequence:
            file.write(num.to_bytes(4, byteorder='little'))

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 generate_and_save_sequence.py <length>")
        sys.exit(1)

    try:
        length = int(sys.argv[1])
        if length <= 0:
            raise ValueError("Length must be a positive integer.")
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    sequence = generate_sequence(length)
    save_to_binary_file(sequence, 'test.bin')