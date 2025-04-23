import heapq

def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan distance

def astar(grid, start, goal):
    cols, rows = len(grid), len(grid[0])
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current = heapq.heappop(open_set)[1]

        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        neighbours = get_neighbours(current, cols, rows)
        for neighbour in neighbours:
            if grid[neighbour[0]][neighbour[1]] == 1:
                continue

            tentative_g = g_score[current] + 1
            if tentative_g < g_score.get(neighbour, float('inf')):
                came_from[neighbour] = current
                g_score[neighbour] = tentative_g
                f_score[neighbour] = tentative_g + heuristic(neighbour, goal)
                heapq.heappush(open_set, (f_score[neighbour], neighbour))

    return []

def get_neighbours(node, cols, rows):
    x, y = node
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]  # 4-way movement
    result = []
    for dx, dy in directions:
        nx, ny = x + dx, y + dy
        if 0 <= nx < cols and 0 <= ny < rows:
            result.append((nx, ny))
    return result
