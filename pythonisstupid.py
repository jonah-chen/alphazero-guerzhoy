string = "[["
for i in range(8):
    for j in range(8):
        string += f"Button(root, command=lambda: move({i}, {j}), padx=50, pady=50)"
        string += ", " if j != 7 else ""
    string += "], [" if i != 7 else "]]"

print(string)