from qubots.base_problem import BaseProblem
import math, random
import os

class LocationRoutingProblem(BaseProblem):
    """
    Location Routing Problem (LRP) for Qubots.
    
    In the LRP, a fleet of vehicles (with uniform capacity) must serve a set of customers using one of several depots.
    Each depot has an opening cost and a capacity, and each customer has a demand.
    Vehicles (trucks) follow routes that start and end at their assigned depot.
    
    A candidate solution is represented as a dictionary with:
      - "routes": a list (of length nb_trucks) where each element is a list of customer indices (0-indexed)
                  indicating the order in which a truck visits customers.
      - "depot_assignment": a list (of length nb_trucks) where each element is a depot index (0-indexed) 
                  assigned to the corresponding truck.
    
    The objective is to minimize the total cost, defined as the sum of:
      - For each used route: the fixed opening route cost plus the travel distance (from the depot to the first customer,
        between consecutive customers, and from the last customer back to the depot).
      - For each depot that serves at least one route: its opening cost.
    
    Additionally, the total demand served by any truck must not exceed the vehicle capacity,
    and the sum of demands served by trucks assigned to a depot must not exceed that depot's capacity.
    """
    
    def __init__(self, instance_file: str, **kwargs):
        (self.nb_customers,
         self.nb_depots,
         self.vehicle_capacity,
         self.opening_route_cost,
         self.demands,
         self.capacity_depots,
         self.opening_depots_cost,
         self.dist_matrix,
         self.dist_depots) = self._read_instance(instance_file)
        
        # Compute minimum number of trucks needed (based on total demand)
        min_nb_trucks = math.ceil(sum(self.demands) / self.vehicle_capacity)
        # Increase by a factor of 1.5 for feasibility
        self.nb_trucks = math.ceil(1.5 * min_nb_trucks)
    
    def _read_instance(self, filename: str):
        """
        Reads the instance file in the S. Barreto format.
        
        Expected format:
          - The first two numbers: number of customers and number of depots.
          - Next, for each depot: two integers for its x and y coordinates.
          - Then, for each customer: two integers for its x and y coordinates.
          - Next: the vehicle capacity.
          - Then, for each depot: its capacity.
          - Then, for each customer: its demand.
          - Then, for each depot: its opening cost (float).
          - Next: the opening cost for a route (integer).
          - Finally: an integer (0 or 1) indicating if costs should be treated as doubles.
        
        The method also computes:
          - A distance matrix among customers.
          - A matrix of distances from each customer to each depot.
        """

        # Resolve relative path with respect to this module’s directory.
        if not os.path.isabs(filename):
            base_dir = os.path.dirname(os.path.abspath(__file__))
            filename = os.path.join(base_dir, filename)

        with open(filename, "r") as f:
            tokens = f.read().split()
        it = iter(tokens)
        
        nb_customers = int(next(it))
        nb_depots = int(next(it))
        
        # Read depot coordinates.
        x_depot = []
        y_depot = []
        for i in range(nb_depots):
            x_depot.append(int(next(it)))
            y_depot.append(int(next(it)))
        
        # Read customer coordinates.
        x_customer = []
        y_customer = []
        for i in range(nb_customers):
            x_customer.append(int(next(it)))
            y_customer.append(int(next(it)))
        
        vehicle_capacity = int(next(it))
        
        capacity_depots = []
        for i in range(nb_depots):
            capacity_depots.append(int(next(it)))
        
        demands = []
        for i in range(nb_customers):
            demands.append(int(next(it)))
        
        temp_opening_cost_depot = []
        for i in range(nb_depots):
            temp_opening_cost_depot.append(float(next(it)))
        temp_opening_route_cost = int(next(it))
        are_cost_double = int(next(it))
        
        if are_cost_double == 1:
            opening_depots_cost = temp_opening_cost_depot
            opening_route_cost = temp_opening_route_cost
        else:
            opening_route_cost = round(temp_opening_route_cost)
            opening_depots_cost = [round(c) for c in temp_opening_cost_depot]
        
        dist_matrix = self._compute_distance_matrix(x_customer, y_customer, are_cost_double)
        dist_depots = self._compute_distance_depot(x_customer, y_customer, x_depot, y_depot, are_cost_double)
        
        return nb_customers, nb_depots, vehicle_capacity, opening_route_cost, demands, capacity_depots, opening_depots_cost, dist_matrix, dist_depots
    
    def _compute_distance_matrix(self, customers_x, customers_y, are_cost_double):
        nb = len(customers_x)
        matrix = [[0]*nb for _ in range(nb)]
        for i in range(nb):
            for j in range(nb):
                d = math.sqrt((customers_x[i]-customers_x[j])**2 + (customers_y[i]-customers_y[j])**2)
                if are_cost_double == 0:
                    d = math.ceil(100*d)
                matrix[i][j] = d
        return matrix
    
    def _compute_distance_depot(self, customers_x, customers_y, depot_x, depot_y, are_cost_double):
        nb_c = len(customers_x)
        nb_d = len(depot_x)
        matrix = [[0]*nb_d for _ in range(nb_c)]
        for i in range(nb_c):
            for d in range(nb_d):
                d_val = math.sqrt((customers_x[i]-depot_x[d])**2 + (customers_y[i]-depot_y[d])**2)
                if are_cost_double == 0:
                    d_val = math.ceil(100*d_val)
                matrix[i][d] = d_val
        return matrix
    
    def evaluate_solution(self, solution) -> float:
        penalty = 1e9
        # Expect solution to be a dict with keys "routes" and "depot_assignment".
        if not isinstance(solution, dict):
            return penalty
        if "routes" not in solution or "depot_assignment" not in solution:
            return penalty
        routes = solution["routes"]
        depot_assignment = solution["depot_assignment"]
        
        if not isinstance(routes, list) or len(routes) != self.nb_trucks:
            return penalty
        if not isinstance(depot_assignment, list) or len(depot_assignment) != self.nb_trucks:
            return penalty
        
        # Verify that every customer is assigned exactly once.
        all_assigned = []
        for route in routes:
            if not isinstance(route, list):
                return penalty
            all_assigned.extend(route)
        if sorted(all_assigned) != list(range(self.nb_customers)):
            return penalty
        
        total_route_cost = 0
        quantity_served = [0] * self.nb_trucks
        # Evaluate each truck's route.
        for r in range(self.nb_trucks):
            route = routes[r]
            if len(route) > 0:
                q = sum(self.demands[i] for i in route)
                if q > self.vehicle_capacity:
                    return penalty
                quantity_served[r] = q
                # Get depot assignment for route r.
                depot = depot_assignment[r]
                if not isinstance(depot, int) or depot < 0 or depot >= self.nb_depots:
                    return penalty
                # Compute route travel distance:
                # distance = from depot to first customer + sum(distance between consecutive customers)
                #            + from last customer back to depot.
                d_route = self.dist_depots[route[0]][depot]
                for i in range(len(route)-1):
                    d_route += self.dist_matrix[route[i]][route[i+1]]
                d_route += self.dist_depots[route[-1]][depot]
                # Add fixed route opening cost.
                total_route_cost += self.opening_route_cost + d_route
            else:
                # Empty route: no cost.
                quantity_served[r] = 0
        
        # For each depot, sum the quantities served by routes assigned to it.
        depot_total = [0] * self.nb_depots
        depot_used = [False] * self.nb_depots
        for r in range(self.nb_trucks):
            depot = depot_assignment[r]
            if len(routes[r]) > 0:
                depot_used[depot] = True
                depot_total[depot] += quantity_served[r]
        # Check depot capacity constraints.
        for d in range(self.nb_depots):
            if depot_total[d] > self.capacity_depots[d]:
                return penalty
        
        depot_cost = 0
        for d in range(self.nb_depots):
            if depot_used[d]:
                depot_cost += self.opening_depots_cost[d]
        
        totalCost = total_route_cost + depot_cost
        return totalCost
    
    def random_solution(self):
        # Generate a random candidate solution.
        # Randomly assign each customer to one of the trucks.
        routes = [[] for _ in range(self.nb_trucks)]
        for customer in range(self.nb_customers):
            r = random.randrange(self.nb_trucks)
            routes[r].append(customer)
        # Shuffle each route.
        for r in range(self.nb_trucks):
            random.shuffle(routes[r])
        # For each route, assign a random depot.
        depot_assignment = [random.randrange(self.nb_depots) for _ in range(self.nb_trucks)]
        return {"routes": routes, "depot_assignment": depot_assignment}
