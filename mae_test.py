import numpy as np
import matplotlib.pyplot as plt
from time import time
from mars_module import Earth
import pyearth as pye
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

iteration = 1

def fit_assert(X, y, tol, max_terms=1, ind=0):
    global iteration
    pye_Earth = pye.Earth(max_terms=max_terms, max_degree=2)

    start_time = time()
    pye_Earth.fit(X, y)
    fit_time = time() - start_time

    y_pred = pye_Earth.predict(X)
    
    mae_pye = np.mean(np.abs(y - y_pred))
    # тут разбиваем и номер теста выше, там же ловим ошибки
    print(f"Test {iteration} pyearth, fit time={fit_time:.3f}s, MAE={mae_pye:.2e}")




    model = Earth(max_terms=max_terms)
    start_time = time()
    model.fit(X, y)
    fit_time = time() - start_time

    y_pred = model.predict(X)
    
    mae = np.mean(np.abs(y - y_pred))
    # тут разбиваем и номер теста выше, там же ловим ошибки
    print(f"Test {iteration}: max_terms={max_terms}, fit time={fit_time:.3f}s, MAE={mae:.2e} -- ", end="")

    try:
        assert(mae < mae_pye)
    except AssertionError:
        print("failed")
        print("MAE > MAE_pyearth")
    except Exception as e:
        print("failed")
        print("Catched exception:", e)
    else:
        print("passed")

    print()
    iteration += 1

def next_test():
    global iteration
    iteration = 1
    print()

# ABS
print("Testing y = |x| - 4:")

np.random.seed(0)
m = 100
n = 1
X = 80 * np.random.uniform(size=(m, n)) - 40
y = np.abs(X[:, 0] - 4.0)

fit_assert(X, y, 0.1, max_terms=3, ind=0)

fit_assert(X, y, 1e-5, max_terms=5, ind=0)

fit_assert(X, y, 1e-5, max_terms=15, ind=0)

#LINEAR
next_test()
print("Testing y = x - 4:")

np.random.seed(0)
m = 100
n = 1
X = 80 * np.random.uniform(size=(m, n)) - 40
y = X[:, 0] - 4.0


fit_assert(X, y, 1e-5, max_terms=3, ind=0)


fit_assert(X, y, 1e-5, max_terms=5, ind=0)


fit_assert(X, y, 1e-5, max_terms=15, ind=0)

# Random linear abs and hinge sum
next_test()
print("Testing y = x - 4 + |x + 5| + min(0, -x + 20):")

np.random.seed(0)
m = 100
n = 1
X = 80 * np.random.uniform(size=(m, n)) - 40
y = X[:, 0] - 4.0 + np.abs(X[:, 0] + 5) + np.minimum(0, -1 * X[:, 0] + 20)


fit_assert(X, y, 2, max_terms=3, ind=0)


fit_assert(X, y, 1, max_terms=5, ind=0)


fit_assert(X, y, 1e-5, max_terms=11, ind=0)

# 5D
next_test()
print("Testing y = x0 - 4 + |x1 + 5| + max(0, x2 + 20) + min(0, -x3) + 10x4:")

np.random.seed(0)
m = 100
n = 5
X = 80 * np.random.uniform(size=(m, n)) - 40
y = X[:, 0] - 4 + np.abs(X[:, 1] + 5) + np.maximum(0, X[:, 2] + 20) + np.minimum(0, -X[:, 3]) + 10 * X[:, 4]


fit_assert(X, y, 100, max_terms=3, ind=0)


fit_assert(X, y, 50, max_terms=5, ind=0)


fit_assert(X, y, 0.1, max_terms=15, ind=0)


fit_assert(X, y, 1e-5, max_terms=20, ind=0)


# mult
next_test()
print("Testing y = x0 * (x1 - 10) + 5x2:")

np.random.seed(0)
m = 100
n = 3
X = 80 * np.random.uniform(size=(m, n)) - 40
y = X[:, 0] * (X[:, 1] - 10) + 5 * X[:, 2]


fit_assert(X, y, 500, max_terms=1, ind=0)


fit_assert(X, y, 500, max_terms=3, ind=0)


fit_assert(X, y, 400, max_terms=5, ind=0)


fit_assert(X, y, 1e-5, max_terms=15, ind=0)


fit_assert(X, y, 1e-5, max_terms=20, ind=0)