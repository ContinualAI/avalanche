import numpy as np
import quadprog
import qpsolvers


if __name__ == "__main__":

    M = np.array([[1.0, 2.0, 0.0], [-8.0, 3.0, 2.0], [0.0, 1.0, 1.0]])
    P = M.T @ M  # this is a positive definite matrix
    q = np.array([3.0, 2.0, 3.0]) @ M
    G = np.array([[1.0, 2.0, 1.0], [2.0, 0.0, 1.0], [-1.0, 2.0, -1.0]])
    h = np.array([3.0, 2.0, -2.0])
    
    A = np.array([0.0, 0.0, 0.0])
    b = np.array([0.0])

    problem = qpsolvers.Problem(P=P, q=-q, G=-G.T, h=-h)
    solution = qpsolvers.solve_problem(problem, solver="quadprog")
    print(f"Primal: x = {solution.x}")
    # print(f"Dual (Gx <= h): z = {solution.z}")
    # print(f"Dual (Ax == b): y = {solution.y}")
    # print(f"Dual (lb <= x <= ub): z_box = {solution.z_box}")
    # print(solution.dual_residual())
    # print(solution.primal_residual())
    # print(solution.duality_gap())

    v = qpsolvers.solve_qp(P=P, q=-q, G=-G.transpose(), h=-h, solver="quadprog")
    print("qpsolvers: ", v)

    # P, q, G, h
    # v = quadprog.solve_qp(P, q, G, h)[0]
    # print("quadprog: ", v)
    v = quadprog.solve_qp(P, q, G, h, factorized=False)
    print("quadprog: ", v)
