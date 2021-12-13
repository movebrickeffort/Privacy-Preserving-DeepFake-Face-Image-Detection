import random
def random_c():
    c = random.uniform(0, 1)
    while(c - 0<1e-6):
        c = random.uniform(0, 1)

    return c