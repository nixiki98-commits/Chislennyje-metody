import time
import math
import sys
import numpy as np

rng = np.random.default_rng(42)


def measure(fn, arr, repeat=3):
    best = float("inf")
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn(arr)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    return result, best


def show(t_loop, t_np):
    print(f"цикл:   {t_loop*1000:10.2f} мс")
    print(f"NumPy:  {t_np*1000:10.2f} мс")
    print(f"ускорение: {t_loop / t_np:.1f}×")


# Задание 1
print("\nЗадание 1")


def loop_sum(arr):
    s = 0.0
    for x in arr:
        s += x
    return s


def numpy_sum(arr):
    return arr.sum()


for n in [10**5, 10**6, 10**7]:
    a = rng.normal(size=n)
    r1, t1 = measure(loop_sum, a)
    r2, t2 = measure(numpy_sum, a)
    print(n, f"{t1*1000:.2f} мс", f"{t2*1000:.3f} мс", f"{t1/t2:.1f}×", np.isclose(r1, r2))


# Задание 2
print("\nЗадание 2")


def loop_mean_std(arr):
    n = len(arr)
    s = 0.0
    s2 = 0.0
    for x in arr:
        s += x
        s2 += x * x
    mean = s / n
    var = s2 / n - mean * mean
    if var < 0:
        var = 0.0
    return mean, math.sqrt(var)


def numpy_mean_std(arr):
    return arr.mean(), arr.std()


a = rng.normal(size=10**6)
(m1, sd1), t1 = measure(loop_mean_std, a)
(m2, sd2), t2 = measure(numpy_mean_std, a)
show(t1, t2)
print("разница mean:", abs(m1 - m2))
print("разница std:", abs(sd1 - sd2))
print("совпадают до 1e-12:", abs(m1 - m2) < 1e-12 and abs(sd1 - sd2) < 1e-12)

b = rng.normal(loc=1e9, scale=1.0, size=10**6)
print("среднее 1e9, цикл:", loop_mean_std(b)[1])
print("среднее 1e9, NumPy:", b.std())


# Задание 3
print("\nЗадание 3")


def loop_func(arr):
    res = []
    for x in arr:
        res.append(math.sin(x) + x * x)
    return res


def numpy_func(arr):
    return np.sin(arr) + arr ** 2


a = rng.normal(size=10**6)
r1, t1 = measure(loop_func, a)
r2, t2 = measure(numpy_func, a)
show(t1, t2)
print("совпали:", np.allclose(r1, r2))

mem_list = sys.getsizeof(r1)
for x in r1:
    mem_list += sys.getsizeof(x)
print("память списка:", round(mem_list / 1e6), "МБ")
print("память массива:", round(r2.nbytes / 1e6), "МБ")


# Задание 4
print("\nЗадание 4")


def loop_clip(arr):
    res = []
    for x in arr:
        if x < 0:
            res.append(0.0)
        elif x > 1:
            res.append(1.0)
        else:
            res.append(x)
    return res


def numpy_clip(arr):
    return np.clip(arr, 0, 1)


a = rng.normal(size=10**6)
r1, t1 = measure(loop_clip, a)
r2, t2 = measure(numpy_clip, a)
show(t1, t2)
print("совпали:", np.allclose(r1, r2))
print("изменено элементов:", np.sum((a < 0) | (a > 1)))


# Задание 5
print("\nЗадание 5")
m = rng.normal(size=(100000, 10))


def loop_sum_rows(mat):
    res = [0.0] * mat.shape[0]
    for i in range(mat.shape[0]):
        s = 0.0
        for j in range(mat.shape[1]):
            s += mat[i, j]
        res[i] = s
    return res


def numpy_sum_rows(mat):
    return mat.sum(axis=1)


r1, t1 = measure(loop_sum_rows, m)
r2, t2 = measure(numpy_sum_rows, m)
show(t1, t2)
print("совпали:", np.allclose(r1, r2))


# Задание 6
print("\nЗадание 6")


def loop_center(mat):
    n = mat.shape[0]
    k = mat.shape[1]
    res = [[0.0] * k for _ in range(n)]
    for j in range(k):
        s = 0.0
        for i in range(n):
            s += mat[i, j]
        mean = s / n
        for i in range(n):
            res[i][j] = mat[i, j] - mean
    return res


def numpy_center(mat):
    return mat - mat.mean(axis=0)


r1, t1 = measure(loop_center, m)
r2, t2 = measure(numpy_center, m)
show(t1, t2)
print("совпали:", np.allclose(np.array(r1), r2))
print("макс. среднее столбца после вычитания:", np.abs(r2.mean(axis=0)).max())


# Задание 7
print("\nЗадание 7")


def loop_max(arr):
    max_val = arr[0]
    max_idx = 0
    for i in range(1, len(arr)):
        if arr[i] > max_val:
            max_val = arr[i]
            max_idx = i
    return max_val, max_idx


def numpy_max(arr):
    return arr.max(), arr.argmax()


a = rng.normal(size=10**6)
r1, t1 = measure(loop_max, a)
r2, t2 = measure(numpy_max, a)
show(t1, t2)
print("совпали:", r1[0] == r2[0] and r1[1] == r2[1])

b = a.copy()
b[0] = np.nan
c = a.copy()
c[500000] = np.nan
print("NaN в начале: цикл", loop_max(b), "| a.max()", b.max(),
      "| nanmax", np.nanmax(b), "| nanargmax", np.nanargmax(b))
print("NaN в середине: цикл", loop_max(c), "| a.max()", c.max())


# Задание 8
print("\nЗадание 8")


def loop_filter(arr):
    return [x for x in arr if x > 0]


def numpy_filter(arr):
    return arr[arr > 0]


a = rng.normal(size=10**6)
r1, t1 = measure(loop_filter, a)
r2, t2 = measure(numpy_filter, a)
show(t1, t2)
print("совпали:", np.allclose(np.array(r1), r2))

mem_list = sys.getsizeof(r1)
for x in r1:
    mem_list += sys.getsizeof(x)
print("память списка:", round(mem_list / 1e6, 1), "МБ")
print("память массива:", round(r2.nbytes / 1e6, 1), "МБ")
print("массив компактнее в", round(mem_list / r2.nbytes, 1), "раз")


# Задание 9
print("\nЗадание 9")


def loop_abs(arr):
    for i in range(len(arr)):
        if arr[i] < 0:
            arr[i] = -arr[i]


def numpy_abs_mask(arr):
    arr[arr < 0] *= -1


def numpy_abs_out(arr):
    np.abs(arr, out=arr)


base = rng.normal(size=10**6)

times = []
for fn in [loop_abs, numpy_abs_mask, numpy_abs_out]:
    best = float("inf")
    for _ in range(3):
        x = base.copy()
        t0 = time.perf_counter()
        fn(x)
        t1 = time.perf_counter()
        best = min(best, t1 - t0)
    times.append(best)
    print(fn.__name__, f"{best*1000:.2f} мс", "верно:", np.array_equal(x, np.abs(base)))

print(f"ускорение a[a<0] *= -1: {times[0]/times[1]:.1f}×")
print(f"ускорение np.abs(out=a): {times[0]/times[2]:.1f}×")

x = base.copy()
y = x
numpy_abs_out(x)
print("тот же массив в памяти:", np.shares_memory(x, y))


# Задание 10
print("\nЗадание 10")


def loop_mask(arr):
    return [x for x in arr if -1 < x < 1]


def numpy_mask(arr):
    return arr[(arr > -1) & (arr < 1)]


a = rng.normal(size=10**6)
r1, t1 = measure(loop_mask, a)
r2, t2 = measure(numpy_mask, a)
show(t1, t2)
print("совпали:", np.allclose(np.array(r1), r2))

try:
    a[:5] > -1 & a[:5] < 1
except Exception as e:
    print("без скобок, float:", type(e).__name__)
try:
    ints = np.arange(5)
    ints > -1 & ints < 1
except Exception as e:
    print("без скобок, int:", type(e).__name__)


# Задание 11
print("\nЗадание 11")
X = rng.normal(size=(1000, 3))
Y = rng.normal(size=(1000, 3))
X_list = X.tolist()
Y_list = Y.tolist()


def loop_dist(data):
    X, Y = data
    n = len(X)
    m = len(Y)
    D = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            dx = X[i][0] - Y[j][0]
            dy = X[i][1] - Y[j][1]
            dz = X[i][2] - Y[j][2]
            D[i][j] = math.sqrt(dx * dx + dy * dy + dz * dz)
    return D


def numpy_dist(data):
    X, Y = data
    return np.sqrt(((X[:, None, :] - Y[None, :, :]) ** 2).sum(axis=2))


def numpy_dist_matmul(data):
    X, Y = data
    sq = (X ** 2).sum(axis=1)[:, None] + (Y ** 2).sum(axis=1)[None, :] - 2 * X @ Y.T
    return np.sqrt(np.maximum(sq, 0))


r1, t1 = measure(loop_dist, (X_list, Y_list), repeat=2)
r2, t2 = measure(numpy_dist, (X, Y))
r3, t3 = measure(numpy_dist_matmul, (X, Y))
show(t1, t2)
print("совпали:", np.allclose(np.array(r1), r2))
print(f"через X @ Y.T: {t3*1000:.2f} мс, совпал: {np.allclose(r3, r2)}")
print("временный массив (1000,1000,3):", 1000 * 1000 * 3 * 8 / 1e6, "МБ")


# Задание 12
print("\nЗадание 12")
k = 10
a = rng.normal(size=10**5)


def loop_moving(arr):
    res = []
    for i in range(k - 1, len(arr)):
        s = 0.0
        for j in range(i - k + 1, i + 1):
            s += arr[j]
        res.append(s / k)
    return res


def numpy_moving(arr):
    return np.convolve(arr, np.ones(k) / k, mode="valid")


r1, t1 = measure(loop_moving, a)
r2, t2 = measure(numpy_moving, a)
show(t1, t2)
print("совпали:", np.allclose(r1, r2))


# Задание 13
print("\nЗадание 13")
a = rng.normal(size=10**5)


def loop_cumsum(arr):
    res = []
    s = 0.0
    for x in arr:
        s += x
        res.append(s)
    return res


def numpy_cumsum(arr):
    return np.cumsum(arr)


r1, t1 = measure(loop_cumsum, a)
r2, t2 = measure(numpy_cumsum, a)
show(t1, t2)
diff = np.max(np.abs(np.array(r1) - r2))
print("макс. разница:", diff, "| меньше 1e-9:", diff < 1e-9)


# Задание 14
print("\nЗадание 14")
ints = rng.integers(0, 100, size=10**6)


def loop_count(arr):
    d = {}
    for v in arr:
        if v in d:
            d[v] += 1
        else:
            d[v] = 1
    res = []
    for i in range(100):
        res.append(d.get(i, 0))
    return res


def numpy_count(arr):
    return np.bincount(arr, minlength=100)


r1, t1 = measure(loop_count, ints)
r2, t2 = measure(numpy_count, ints)
show(t1, t2)
print("частоты совпали:", list(r1) == list(r2))


# Задание 15
print("\nЗадание 15")
N = 100000
raw = rng.normal(size=(N, 3))
raw[rng.integers(0, N, 100), rng.integers(0, 3, 100)] = 1e6


def step1_loop(raw):
    n = len(raw)
    mask = [[False] * 3 for _ in range(n)]
    for j in range(3):
        s = 0.0
        for i in range(n):
            s += raw[i][j]
        mean = s / n
        v = 0.0
        for i in range(n):
            v += (raw[i][j] - mean) ** 2
        std = (v / n) ** 0.5
        for i in range(n):
            if abs(raw[i][j] - mean) > 5 * std:
                mask[i][j] = True
    return mask


def step1_numpy(raw):
    return np.abs(raw - raw.mean(axis=0)) > 5 * raw.std(axis=0)


def step2_loop(data):
    raw, mask = data
    n = len(raw)
    res = [row[:] for row in raw]
    for j in range(3):
        s = 0.0
        c = 0
        for i in range(n):
            if not mask[i][j]:
                s += raw[i][j]
                c += 1
        mean = s / c
        for i in range(n):
            if mask[i][j]:
                res[i][j] = mean
    return res


def step2_numpy(data):
    raw, mask = data
    tmp = np.where(mask, np.nan, raw)
    return np.where(np.isnan(tmp), np.nanmean(tmp, axis=0), tmp)


def step3_loop(data):
    n = len(data)
    res = [[0.0] * 3 for _ in range(n)]
    for j in range(3):
        s = 0.0
        for i in range(n):
            s += data[i][j]
        mean = s / n
        v = 0.0
        for i in range(n):
            v += (data[i][j] - mean) ** 2
        std = (v / n) ** 0.5
        for i in range(n):
            res[i][j] = (data[i][j] - mean) / std
    return res


def step3_numpy(data):
    return (data - data.mean(axis=0)) / data.std(axis=0)


def step4_loop(data):
    n = len(data)
    means = []
    stds = []
    for j in range(3):
        s = 0.0
        for i in range(n):
            s += data[i][j]
        mean = s / n
        v = 0.0
        for i in range(n):
            v += (data[i][j] - mean) ** 2
        means.append(mean)
        stds.append((v / n) ** 0.5)
    return means, stds


def step4_numpy(data):
    return data.mean(axis=0), data.std(axis=0)


def step5_loop(data):
    n = len(data)
    means = [0.0, 0.0, 0.0]
    for j in range(3):
        for i in range(n):
            means[j] += data[i][j] / n
    cov = [[0.0] * 3 for _ in range(3)]
    for j in range(3):
        for k in range(3):
            for i in range(n):
                cov[j][k] += (data[i][j] - means[j]) * (data[i][k] - means[k])
    corr = [[0.0] * 3 for _ in range(3)]
    for j in range(3):
        for k in range(3):
            corr[j][k] = cov[j][k] / (cov[j][j] * cov[k][k]) ** 0.5
    return corr


def step5_numpy(data):
    return np.corrcoef(data.T)


mask_np = step1_numpy(raw)
clean_np = step2_numpy((raw, mask_np))
norm_np = step3_numpy(clean_np)

steps = [
    ("1 выбросы", step1_loop, step1_numpy, raw.tolist(), raw, 16, 2),
    ("2 замена", step2_loop, step2_numpy, (raw.tolist(), mask_np.tolist()), (raw, mask_np), 15, 3),
    ("3 нормировка", step3_loop, step3_numpy, clean_np.tolist(), clean_np, 15, 2),
    ("4 mean/std", step4_loop, step4_numpy, norm_np.tolist(), norm_np, 15, 2),
    ("5 корреляция", step5_loop, step5_numpy, norm_np.tolist(), norm_np, 16, 2),
]

print("шаг            цикл, мс   NumPy, мс   ускорение   строк цикл   строк NumPy")
total_loop = 0
total_np = 0
lines_loop = 0
lines_np = 0
for name, f_loop, f_np, d_loop, d_np, l1, l2 in steps:
    r1, t1 = measure(f_loop, d_loop, repeat=2)
    r2, t2 = measure(f_np, d_np)
    total_loop += t1
    total_np += t2
    lines_loop += l1
    lines_np += l2
    print(name.ljust(14), str(round(t1 * 1000, 1)).rjust(8), str(round(t2 * 1000, 1)).rjust(10),
          str(round(t1 / t2, 1)).rjust(10), str(l1).rjust(12), str(l2).rjust(13))
print("ИТОГО".ljust(14), str(round(total_loop * 1000, 1)).rjust(8),
      str(round(total_np * 1000, 1)).rjust(10), str(round(total_loop / total_np, 1)).rjust(10),
      str(lines_loop).rjust(12), str(lines_np).rjust(13))

means, stds = step4_numpy(norm_np)
print("найдено выбросов:", mask_np.sum())
print("после нормировки mean:", means, "std:", stds)
print("корреляционная матрица:")
print(np.round(step5_numpy(norm_np), 4))
print("цикл и NumPy дали одинаковую корреляцию:",
      np.allclose(np.array(step5_loop(norm_np.tolist())), step5_numpy(norm_np)))

