
import numpy as np
import matplotlib.pyplot as plt
import time

def create_set_of_equations(a1, a2, a3, n, lam):  
    arr = np.zeros((n, n))

    d1 = np.repeat(a1, n)
    d2 = np.repeat(a2, n-1)
    d3 = np.repeat(a3, n-2)

    arr += np.diag(d1)
    arr += np.diag(d2, -1)
    arr += np.diag(d3, -2)
    arr += np.diag(d2, 1)
    arr += np.diag(d3, 2)

    b = np.fromfunction(lam, (n, 1))

    return arr, b

def task_A(c, d, e, f):

    print("task A")
    a1 = 5 + e
    a2 = a3 = -1
    n = 9 * c * d   
    print(f"\ta1 = {a1}, n = {n}")

    arr, b = create_set_of_equations(a1, a2, a3, n, (lambda n, i : np.sin(n * (f + 1))))

    return arr, b

def jacobi_method(a, b, max_iter = float('inf'), show_iter = False):

    start = time.time()
    n = a.shape[0]
    low_triangle = np.tril(a, -1)
    up_triangle = np.triu(a, 1)
    diag = np.diag(np.diag(a))
    iteration = 0
    x = np.ones(n).reshape((n, 1))
    goal_residuum_norm = 1e-9
    
    residuum_norm = float('inf')

    
    while True:
        iteration += 1
        new_x = np.add(low_triangle, up_triangle)
        new_x = np.linalg.lstsq(diag, new_x, rcond=None)[0]
        new_x = np.negative(new_x)
        new_x = np.matmul(new_x, x)
        new_x = np.add(new_x, np.linalg.lstsq(diag, b, rcond=None)[0])
        x = new_x
        
        residuum = np.subtract(np.matmul(a, x), b)
        new_residuum_norm = np.linalg.norm(residuum)
        if show_iter:
            print(f"{iteration}) Residuum norm = {new_residuum_norm}")
        if new_residuum_norm < residuum_norm:
            residuum_norm = new_residuum_norm
        if iteration > max_iter:
            break
        if residuum_norm <= goal_residuum_norm:
            break
    end = time.time()
    ret_time = end - start
    return residuum_norm, iteration, ret_time

def gauss_seidel_method(a, b, max_iter = float('inf'), show_iter = False):
    start = time.time()
    n = a.shape[0]
    low_triangle = np.tril(a, -1)
    up_triangle = np.triu(a, 1)
    diag = np.diag(np.diag(a))
    iteration = 0
    x = np.ones(n).reshape((n, 1))
    goal_residuum_norm = 1e-9
    residuum_norm = float('inf')

    while True:
        iteration += 1

        first = np.add(diag, low_triangle)
        first = np.negative(first)
        first = np.linalg.lstsq(first, np.matmul(up_triangle, x), rcond=None)[0]
        second = np.add(diag, low_triangle)
        second = np.linalg.lstsq(second, b, rcond=None)[0]
        x = np.add(first, second)



        residuum = np.subtract(np.matmul(a, x), b)
        new_residuum_norm = np.linalg.norm(residuum)
        if show_iter:
            print(f"{iteration}) Residuum norm = {new_residuum_norm}")
        if new_residuum_norm < residuum_norm:
            residuum_norm = new_residuum_norm
        if iteration > max_iter:
            break
        if residuum_norm <= goal_residuum_norm:
            break
    end = time.time()
    ret_time = end - start
    return residuum_norm, iteration, ret_time

def task_B(a, b):

    print("Task B")
    jacobi_residuum_norm, jacobi_iteration_count, jacobi_time = jacobi_method(a,b)

    print(f"\tJacobi method: \n \
        \t\ttime = {jacobi_time}  [s]\n \
        \t\titerations = {jacobi_iteration_count}\n \
        \t\tresiduum norm = {jacobi_residuum_norm}")

    gauss_seidel_resiuum_norm, gauss_seidel_iteration_count, gauss_seidel_time = gauss_seidel_method(a,b)

    print(f"\tGauss-Seidel method: \n \
        \t\ttime = {gauss_seidel_time}  [s]\n \
        \t\titerations = {gauss_seidel_iteration_count}\n \
        \t\tresiduum norm = {gauss_seidel_resiuum_norm}")


def task_C(c, d, e, f):

    a1 = 3
    a2 = a3 = -1
    n = 9*c*d
    arr, b = create_set_of_equations(a1, a2, a3, n, (lambda n, i : np.sin(n * (f + 1))))

    print("Task C")

    jacobi_residuum_norm, jacobi_iteration_count, jacobi_time = jacobi_method(arr,b, max_iter=30, show_iter=True)
    print(f"\tJacobi method: \n \
    \t\titerations = {jacobi_iteration_count}\n \
    \t\tresiduum norm = {jacobi_residuum_norm}")

    gauss_seidel_resiuum_norm, gauss_seidel_iteration_count, gauss_seidel_time = gauss_seidel_method(arr,b, max_iter=30,  show_iter=True)
    print(f"\tJacobi method: \n \
    \t\titerations = {gauss_seidel_iteration_count}\n \
    \t\tresiduum norm = {gauss_seidel_resiuum_norm}")

    print("\tBoth iteration methos do not coincide")


    return arr, b


def simple_lu_factorization(A):

    n = A.shape[0]

    U = A.copy()
    L = np.eye(n, dtype=np.double)

    for i in range(n):
        factor = U[i+1:, i] / U[i, i]
        L[i+1:, i] = factor
        U[i+1:] -= factor[:, np.newaxis] * U[i]

    return L, U

def forward_substitution(L, b):

    n = L.shape[0]

    y = np.zeros_like(b, dtype=np.double)

    y[0] = b[0] / L[0, 0]

    for i in range(1, n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i,i]

    return y

def back_substitution(U, y):

    n = U.shape[0]

    x = np.zeros_like(y, dtype=np.double)

    x[-1] = y[-1] / U[-1, -1]

    for i in range(n-2, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i:], x[i:])) / U[i,i]

    return x

def lu_method(a, b):

    start = time.time()
    L, U = simple_lu_factorization(a)
    y = forward_substitution(L, b)
    x = back_substitution(U, y)
    residuum = np.subtract(np.matmul(a, x), b)
    residuum_norm = np.linalg.norm(residuum)
    end = time.time()
    ret_time = end - start

    return residuum_norm, ret_time

def task_D(a, b):

    residuum_norm, time = lu_method(a, b)
    

    print(f"Task D - factorization LU\n \
        residuum norm = {residuum_norm}\n \
        time = {time}  [s]")
        
    return

def task_E(c, d, e, f):
    a1 = 5 + e
    a2 = a3 = -1
    N = [100, 500, 1000, 2000, 3000]
    jacobi_times = []
    gauss_seidel_times = []
    lu_times = []

    for n in N:
        A, b = create_set_of_equations(a1, a2, a3, n, (lambda n, i : np.sin(n * (f + 1))))
        jacobi_residuum_norm, jacobi_iteration_count, jacobi_time = jacobi_method(A,b)
        gauss_seidel_resiuum_norm, gauss_seidel_iteration_count, gauss_seidel_time = gauss_seidel_method(A,b)
        lu_resiuum_norm, lu_time = lu_method(A, b)
        print(f"N = {n}\n \
            Jacobi method: time = {jacobi_time}  [s], iterations = {jacobi_iteration_count}, residuum norm = {jacobi_residuum_norm}\n \
            Gauss-Seidel method: time = {gauss_seidel_time}  [s], iterations = {gauss_seidel_iteration_count}, residuum norm = {gauss_seidel_resiuum_norm}\n \
            LU factorization method: time = {lu_time}  [s], residuum norm = {lu_resiuum_norm}\n")
        jacobi_times.append(jacobi_time)
        gauss_seidel_times.append(gauss_seidel_time)
        lu_times.append(lu_time)

    plt.figure(1)
    plt.plot(N, jacobi_times)
    plt.xlabel('N')
    plt.ylabel('time [s]')
    plt.title('Jacobi method')
    plt.savefig('jacobi.png')

    plt.figure(2)
    plt.plot(N, gauss_seidel_times)
    plt.xlabel('N')
    plt.ylabel('time [s]')
    plt.title('Gauss-Seidel method')
    plt.savefig('gauss_seidel.png')

    plt.figure(3)
    plt.plot(N, lu_times)
    plt.xlabel('N')
    plt.ylabel('time [s]')
    plt.title('LU factorization method')
    plt.savefig('lu_factorization.png')

    return

def main():
    c = 3
    d = 8
    e = 5
    f = 0
    a, b = task_A(c, d, e, f)
    task_B(a, b)
    a, b = task_C(c, d, e, f)
    task_D(a, b)
    task_E(c,d,e,f)
    return

if __name__ == '__main__':
    main()