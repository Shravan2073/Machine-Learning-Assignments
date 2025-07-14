

def range(numbers):
    if len(numbers) < 3:
        return "Range determination not possible"
    smallest = min(numbers)
    largest = max(numbers)
    return largest - smallest

my_list = [5, 3, 8, 1, 0, 4]
print("rrange of list:", range(my_list))
