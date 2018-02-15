data = {1 : [2,3],   # Directed acyclic graph adjacency list
        2 : [4],
        3 : [5, 6],
        4 : [7],
        5 : [8],
        6 : [7],
        9 : [4, 5]
        }

def dfs(data, path, paths=[]):
    datum = path[-1]
    if datum in data:
        for val in data[datum]:
            new_path = path + [val]
            paths = dfs(data, new_path, paths)
    else:
        paths += [path]
    return paths

print(dfs(data = data, path = [1], paths = []))
print(dfs(data = data, path = [9], paths = []))
