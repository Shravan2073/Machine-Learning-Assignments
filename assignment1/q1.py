

listt = [2,7,4,1,3,6]
target = 10



def count(numbers):
    count = 0
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if numbers[i] + numbers[j] == 10:
                count += 1
    return count

list = [2, 7, 4, 1, 3, 6]
print("pair count: ", count(list))