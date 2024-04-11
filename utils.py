import random

class imgGen:
    """
    Utility function to generate matrix of given size,
    By default generates 28*28
    """
    def __init__(self, size=[28,28]):
        self.size = size
    def generate_random_decimal_list(self):
        return [round(random.uniform(0, 1), 2) for i in range(self.size[0])]
    def randDecMatrix (self):
        random_decimal_matrix = []
        for i in range(self.size[1]):
            random_decimal_list = self.generate_random_decimal_list()
            random_decimal_matrix.append(random_decimal_list)
        return random_decimal_matrix



# Generate a list of size 28x28
# mainfunc = imgGen()

# Example output
# print(mainfunc.randDecMatrix())