import sys

def read_binary_file(filename):
    with open(filename, 'rb') as file:
        content = file.read()
    return content

def convert_bytes_to_integers(content):
    integers = [int.from_bytes(content[i:i+4], byteorder='little') for i in range(0, len(content), 4)]
    return integers

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 read_binary_and_convert.py <filename>")
        sys.exit(1)

    filename = sys.argv[1]
    
    binary_content = read_binary_file(filename)
    integers = convert_bytes_to_integers(binary_content)

    print(integers)