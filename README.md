# Eulerian Pathfinder: The Chinese Postman Solver

This project implements a solution to the **Chinese Postman Problem** using **Dijkstra's algorithm** for shortest paths, **Hierholzer's algorithm** for finding Eulerian circuits, and a **minimum weight perfect matching** approach to handle odd-degree vertices.

## Features
- Generates a **random weighted graph**.
- Computes **graph statistics** (number of vertices, edges, and average degree).
- Checks whether the graph is **connected**.
- Identifies if a graph is **Eulerian**.
- Uses **Dijkstra's algorithm** to compute shortest paths between odd-degree vertices.
- Finds the **minimum weight perfect matching** using dynamic programming.
- Constructs an **Eulerian circuit** by adding matching edges.
- **Visualizes** the original and modified graphs using `matplotlib` and `networkx`.

## Dependencies
Ensure you have the following Python libraries installed:

```sh
pip install matplotlib networkx
```

## How It Works
1. **Generate a Random Graph**: The program creates a weighted undirected graph with a given number of vertices and edge probability.
2. **Graph Statistics & Connectivity Check**: Displays graph properties and verifies if the graph is connected.
3. **Solve the Chinese Postman Problem**:
   - If the graph is **already Eulerian**, it finds an Eulerian circuit directly.
   - If not, it identifies odd-degree vertices and finds the **minimum weight perfect matching**.
   - It adds these matching edges to the graph and then computes an Eulerian circuit.
4. **Graph Visualization**: The original and modified graphs are displayed, with the Eulerian circuit highlighted in red.

## Usage
Run the script directly:

```sh
python chinese_postman.py
```

This will generate a random graph, analyze its properties, and find an optimal Eulerian circuit for the **Chinese Postman Problem**.

## Example Output
```sh
Graph Statistics: Vertices=10, Edges=23, Average Degree=4.6
Graph is connected.
Chinese Postman Tour Length: 52.0
Eulerian Circuit (Chinese Postman Tour): [0, 3, 1, 6, 8, 7, 2, 9, 5, 4, 0]
```
The circuit is then visualized using `matplotlib`.

## File Structure
```
/ChinesePostmanProblem
│── chinese_postman.py  # Main script
│── README.md           # Project documentation
```

## Future Improvements
- Allow user input for custom graphs.
- Implement a more efficient perfect matching algorithm.
- Support directed graphs.

## License
This project is licensed under the MIT License.
Contact
## Contact
For any inquiries, reach out at parikshitvel0210@gmail.com or open an issue on GitHub.
