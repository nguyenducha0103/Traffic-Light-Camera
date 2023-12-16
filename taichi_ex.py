import time

N = 1000000

def is_prime(n: int):
    result = True
    for k in range(2, int(n**0.5) + 1):
        if n % k == 0:
            result = False
            break
    return result

def count_primes(n: int) -> int:
    count = 0
    for k in range(2, n):
        if is_prime(k):
            count += 1

    return count
for i in range(10):
    start = time.perf_counter()
    print(f"Number of primes: {count_primes(N)}")
    print(f"time elapsed: {time.perf_counter() - start}/s")

import taichi as ti
ti.init(arch=ti.gpu)

@ti.func
def is_prime(n: int):
    result = True
    for k in range(2, int(n**0.5) + 1):
        if n % k == 0:
            result = False
            break
    return result

@ti.kernel
def count_primes(n: int) -> int:
    count = 0
    for k in range(2, n):
        if is_prime(k):
            count += 1

    return count

# for i in range(10000):
#     start = time.perf_counter()
#     print(f"Number of primes: {count_primes(N)}")
#     print(f"time elapsed: {time.perf_counter() - start}/s")