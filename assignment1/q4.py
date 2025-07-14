def maxchar(text):
    counts = {}
    for char in text.lower():
        if char.isalpha():
            counts[char] = counts.get(char, 0) + 1
    
    maxcount = max(counts.values())
    maxchar = max([c for c, count in counts.items() if count == maxcount])
    return maxchar, maxcount

text = "hippopotamus"
char, count = maxchar(text)
print(f"'{char}' appears {count} times")