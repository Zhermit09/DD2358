# Plain Python version — no Cython type declarations yet.
# The annotation report will show every line that touches the Python C-API.

def gauss_seidel(f):
    N = len(f)
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            f[i][j] = 0.25 * (f[i][j+1] + f[i][j-1] +
                               f[i+1][j] + f[i-1][j])
    return f
