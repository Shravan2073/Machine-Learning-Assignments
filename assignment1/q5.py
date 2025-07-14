import random

def randomstats():
    nums = [random.randint(1, 10) for _ in range(25)]
    
    mean = sum(nums) / 25
    median = sorted(nums)[12]
    
    counts = {}
    for n in nums:
        counts[n] = counts.get(n, 0) + 1
    mode = max(counts, key=counts.get)
    
    return nums, mean, median, mode

nums, mean, median, mode = randomstats()
print(f"Numbers: {nums}")
print(f"Mean: {mean}, Median: {median}, Mode: {mode}")