import random

num_arrays = 2
array_length = 10000000

with open("output4.txt", "w") as file:
    file.write(f"{array_length}\n")

    for _ in range(num_arrays):
        random_numbers = [random.randint(-100, 100) for _ in range(array_length)]
        array_str = " ".join(map(str, random_numbers))
        file.write(f"{array_str}\n")