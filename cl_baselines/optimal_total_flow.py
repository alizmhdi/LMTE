import numpy as np
import gurobipy as grb


class TotalFlowOptimal:
    """
    Optimal total-flow traffic engineering solver using linear programming.

    Maximises the total routed demand subject to per-edge capacity constraints,
    mirroring the MetaRL PathOptimalSolver for the 'total_flow' objective.

    The LP formulation:
        Variables : f_{i}_{j}_{k} >= 0  (actual flow on path k for commodity (i,j))
        Constraints:
            sum_k f_{i}_{j}_{k}  <= demand[i][j]          (demand upper-bound)
            sum_{(i,j,k): e in path} f_{i}_{j}_{k} <= cap(e)  (link capacity)
        Objective : maximise sum_{i,j,k} f_{i}_{j}_{k}
    """

    def __init__(self, topology, candidate_path, edge_to_path):
        """
        Args:
            topology: NetworkX DiGraph with 'capacity' edge attribute.
            candidate_path: dict {(i, j): [path_0, path_1, ...]} where each path
                is a list of node ids.
            edge_to_path: dict {(u, v): [(src, dst, path_index), ...]} mapping
                each directed edge to the (src, dst, k) triples whose k-th path
                traverses that edge.
        """
        self.topology = topology
        self.candidate_path = candidate_path
        self.edge_to_path = edge_to_path
        self._n = topology.number_of_nodes()

        # Pre-compute the ordered variable name list (same order for every call).
        self._var_names = [
            f'f_{i}_{j}_{k}'
            for i in range(self._n)
            for j in range(self._n)
            if j != i
            for k in range(len(self.candidate_path[(i, j)]))
        ]

    def maximize_total_flow(self, demand):
        """
        Maximise total routed demand for a single traffic matrix.

        Args:
            demand: np.ndarray, shape (N, N) or (N*N,)
                Traffic demand matrix in raw capacity units (same units as
                the 'capacity' edge attribute).

        Returns:
            tuple: (total_flow, path_flow_routing)
                total_flow (float): optimal total routed flow (same units as demand).
                path_flow_routing (dict): maps 'f_{i}_{j}_{k}' -> actual flow value;
                    compatible with get_weight_dict_tensor for downstream use.
                Returns (None, None) if the LP has no optimal solution.
        """
        n = self._n
        demand = np.asarray(demand, dtype=float)
        if demand.shape != (n, n):
            demand = demand.reshape((n, n))

        m = grb.Model('total_flow_opt')
        m.Params.OutputFlag = 0

        # Decision variables: actual flow on each path
        path_flow = m.addVars(self._var_names, lb=0.0, vtype=grb.GRB.CONTINUOUS,
                               name='path_flow')

        # Demand constraints: total flow per commodity <= demand[i][j]
        m.addConstrs(
            grb.quicksum(
                path_flow[f'f_{i}_{j}_{k}']
                for k in range(len(self.candidate_path[(i, j)]))
            ) <= float(demand[i][j])
            for i in range(n)
            for j in range(n)
            if j != i
        )

        # Capacity constraints: total flow on each directed edge <= capacity
        m.addConstrs(
            grb.quicksum(
                path_flow[f'f_{src}_{dst}_{k}']
                for (src, dst, k) in self.edge_to_path[edge]
            ) <= float(self.topology.edges[edge]['capacity'])
            for edge in self.topology.edges
        )

        # Objective: maximise total routed flow
        m.setObjective(
            grb.quicksum(path_flow[name] for name in self._var_names),
            grb.GRB.MAXIMIZE,
        )
        m.optimize()

        if m.status == grb.GRB.Status.OPTIMAL:
            solution = m.getAttr('x', path_flow)
            path_flow_routing = {name: solution[name] for name in self._var_names}
            return m.objVal, path_flow_routing

        print('[TotalFlowOptimal] No optimal solution found (status={}).'.format(m.status))
        return None, None
