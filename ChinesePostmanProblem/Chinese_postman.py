import itertools
import heapq
import matplotlib.pyplot as plt
import networkx as nx
import random

def dijkstra(graph, start):
    """Finds the shortest paths from start vertex to all other vertices using Dijkstra's algorithm."""
    n = len(graph)
    distances = [float('inf')] * n
    distances[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        if current_distance > distances[current_vertex]:
            continue

        for neighbor, weight in graph[current_vertex]:
            distance = current_distance + weight

            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    return distances

def is_eulerian(graph):
    """Checks if the graph is Eulerian (all vertices have even degree)."""
    return all(len(edges) % 2 == 0 for edges in graph)

def find_minimum_weight_matching(odd_vertices, pair_distances):
    """Finds the minimum weight matching for a set of odd-degree vertices and returns the matching pairs."""
    n = len(odd_vertices)
    dp = [float('inf')] * (1 << n)
    dp[0] = 0
    parent = [-1] * (1 << n)

    pairings = {}
    for i, j in itertools.combinations(range(n), 2):
        mask = (1 << i) | (1 << j)
        pairings[mask] = pair_distances[(odd_vertices[i], odd_vertices[j])]

    for mask in range(1 << n):
        if bin(mask).count('1') % 2 != 0:
            continue

        for submask, cost in pairings.items():
            if (mask & submask) == 0:
                new_mask = mask | submask
                if dp[new_mask] > dp[mask] + cost:
                    dp[new_mask] = dp[mask] + cost
                    parent[new_mask] = mask

    # Recover the matching pairs from the dp table
    matching_pairs = []
    mask = (1 << n) - 1
    while mask > 0:
        submask = mask ^ parent[mask]
        bits = [i for i in range(n) if (submask & (1 << i)) > 0]
        u, v = odd_vertices[bits[0]], odd_vertices[bits[1]]
        matching_pairs.append((u, v, pair_distances[(u, v)]))
        mask = parent[mask]

    return dp[-1], matching_pairs

def add_matching_edges(graph, matching_edges):
    """Adds the matching edges to the original graph."""
    modified_graph = [edges[:] for edges in graph]
    for u, v, weight in matching_edges:
        modified_graph[u].append((v, weight))
        modified_graph[v].append((u, weight))
    return modified_graph

def find_eulerian_circuit(graph, start):
    """Finds the Eulerian circuit in the graph using Hierholzer's algorithm."""
    stack = [start]
    path = []
    local_graph = [edges[:] for edges in graph]  # Make a copy of the graph

    while stack:
        v = stack[-1]

        if local_graph[v]:
            u, _ = local_graph[v].pop()
            local_graph[u].remove((v, _))
            stack.append(u)
        else:
            path.append(stack.pop())

    return path

def chinese_postman(graph):

    n = len(graph)

    # Step 1: Check if the graph is Eulerian
    if is_eulerian(graph):
        # If the graph is Eulerian, find the sum of all edge weights and the Eulerian circuit
        tour_length = sum(weight for edges in graph for _, weight in edges) / 2
        eulerian_circuit = find_eulerian_circuit(graph, 0)
        return tour_length, eulerian_circuit, graph

    # Step 2: Find all vertices with odd degree
    odd_vertices = [v for v in range(n) if len(graph[v]) % 2 == 1]

    # Step 3: Find the shortest paths between all odd degree vertices
    pair_distances = {}
    for i, u in enumerate(odd_vertices):
        distances = dijkstra(graph, u)
        for j, v in enumerate(odd_vertices):
            if i != j:
                pair_distances[(u, v)] = distances[v]
                pair_distances[(v, u)] = distances[v]  # Add the reverse pair as well

    # Step 4: Find the minimum weight perfect matching
    matching_cost, matching_edges = find_minimum_weight_matching(odd_vertices, pair_distances)

    # Step 5: Create a modified graph by adding matching edges
    modified_graph = add_matching_edges(graph, matching_edges)

    # Step 6: The graph is now Eulerian, find the Eulerian circuit
    eulerian_circuit = find_eulerian_circuit(modified_graph, 0)

    # Step 7: Calculate the total length of the Chinese Postman Tour
    tour_length = sum(weight for edges in graph for _, weight in edges) / 2 + matching_cost

    return tour_length, eulerian_circuit, modified_graph

def visualize_graph(graph, eulerian_circuit=None):
    """Visualize the graph with the Eulerian circuit using matplotlib."""
    G = nx.Graph()
    for u, edges in enumerate(graph):
        for v, weight in edges:
            G.add_edge(u, v, weight=weight)

    pos = nx.spring_layout(G)  # Positions for all nodes

    # Draw the graph
    nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, edge_color='blue')
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): f"{d['weight']}" for u, v, d in G.edges(data=True)})

    if eulerian_circuit:
        # Highlight the Eulerian circuit in red
        circuit_edges = [(eulerian_circuit[i], eulerian_circuit[i + 1]) for i in range(len(eulerian_circuit) - 1)]
        circuit_edges += [(eulerian_circuit[-1], eulerian_circuit[0])]  # Close the circuit
        nx.draw_networkx_edges(G, pos, edgelist=circuit_edges, edge_color='red', width=2)

    plt.title("Graph Visualization")
    plt.axis('off')  # Turn off the axis
    plt.show()

def generate_random_graph(n, p):
    """Generates a random graph with n vertices and edge probability p."""
    graph = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                weight = random.randint(1, 10)
                graph[i].append((j, weight))
                graph[j].append((i, weight))
    return graph

# New feature: Graph statistics
def graph_statistics(graph):
    """Calculates and returns graph statistics: number of vertices, number of edges, and average degree."""
    n = len(graph)
    m = sum(len(edges) for edges in graph) // 2
    avg_degree = sum(len(edges) for edges in graph) / n
    return n, m, avg_degree

# New feature: Graph connectivity check
def is_connected(graph):
    """Checks if the graph is connected using DFS."""
    visited = [False] * len(graph)
    stack = [0]

    while stack:
        v = stack.pop()
        if not visited[v]:
            visited[v] = True
            for neighbor, _ in graph[v]:
                if not visited[neighbor]:
                    stack.append(neighbor)

    return all(visited)

# Example usage
if __name__ == "__main__":
    # Generate a random graph
    n = 10
    p = 0.5
    G = generate_random_graph(n, p)

    # Calculate graph statistics
    n, m, avg_degree = graph_statistics(G)
    print(f"Graph Statistics: Vertices={n}, Edges={m}, Average Degree={avg_degree:.2f}")

    # Check if the graph is connected
    if is_connected(G):
        print("Graph is connected.")
    else:
        print("Graph is not connected.")

    # Solve the Chinese Postman Problem
    tour_length, eulerian_circuit, modified_graph = chinese_postman(G)

    print("Chinese Postman Tour Length:", tour_length)
    print("Eulerian Circuit (Chinese Postman Tour):", eulerian_circuit)

    # Visualize the original graph
    print("\nOriginal Graph:")
    visualize_graph(G)

    # Visualize the modified graph with the Eulerian circuit
    print("\nModified Graph with Eulerian Circuit:")
    visualize_graph(modified_graph, eulerian_circuit)