def bisection_method(f, a, b, eps=1e-6, max_iter=1000):
    if f(a) * f(b) >= 0:
        raise ValueError("Функция должна принимать значения разных знаков на концах отрезка [a, b]!")

    iteration = 0
    while (b - a) >= eps and iteration < max_iter:
        c = (a + b) / 2.0
        f_c = f(c)
        
        if f_c == 0:
            return c, iteration
        elif f(a) * f_c < 0:
            b = c
        else:
            a = c
            
        iteration += 1
        
    root = (a + b) / 2.0
    return root, iteration


if __name__ == "__main__":
    print("--- ТЕСТ 1: Уравнение x^2 - 4 = 0 ---")
    f1 = lambda x: x**2 - 4
    root1, iters1 = bisection_method(f1, a=0, b=5, eps=1e-6)
    print(f"Найденный корень: {root1:.8f}")
    print(f"Количество итераций: {iters1}")
    print(f"Проверка f(root): {f1(root1):.2e}\n")

    print("--- ТЕСТ 2: Уравнение cos(x) - x = 0 ---")
    import math
    f2 = lambda x: math.cos(x) - x
    root2, iters2 = bisection_method(f2, a=0, b=1, eps=1e-6)
    print(f"Найденный корень: {root2:.8f}")
    print(f"Количество итераций: {iters2}")
    print(f"Проверка f(root): {f2(root2):.2e}")