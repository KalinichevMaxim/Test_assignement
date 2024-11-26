import numpy as np
import matplotlib.pyplot as plt

# Функция для нахождения LU-разложения матрицы
def lu_decomp(a):
    n = len(a[0, :])
    u = np.copy(a)
    l = np.zeros((n, n))
            
    for k in range(1, n):
        for i in range(k - 1, n):
            for j in range(i, n):
                l[j, i] = u[j, i] / u[i, i]
                
        for i in range(k, n):
            for j in range(k - 1, n):
                u[i, j] = u[i, j] - l[i,k-1] * u[k-1, j]

    return l , u

# Функция для нахождения обратной матрицы

def invert_matrix(a):
    n = len(a[:,0])
    l, u = lu_decomp(a)
    res = np.zeros((n, n))
    temp = np.zeros(n)
    e = np.eye(n)
    for i in range(n):
        for j in range(n):
            temp[j] = e[j, i] - np.dot(temp[:j], l[j, :j])
        for j in range(n - 1, -1, -1):
            res[j, i] = (temp[j] - np.dot(res[j:, i], u[j, j:])) / u[j, j]
    return res

#Вычисление производной
def derivative(t, i, j, y, z, n_func, flag):
    eps = 0.001
    
    if flag == 0:
        y1 = np.copy(y)
        y2 = np.copy(y)
        y1[j] = y[j] + eps
        y2[j] = y[j] - eps
        if n_func == 1:
            return (cond(t, y1, z) - cond(t, y2, z))/ eps * 0.5
        if n_func != 1:
            return (right_impl(t, y1, z)[i] - right_impl(t, y2, z)[i])/ eps * 0.5
        
    if flag == 1:
        if n_func == 1:
            return (cond(t, y, z + eps) - cond(t, y, z - eps))/ eps * 0.5
        if n_func != 1:
            return (right_impl(t, y, z + eps)[i] - right_impl(t, y, z - eps)[i])/ eps * 0.5

#Вычисление матрицы Якоби
def jacobi(t, y, z, n_func, flag):
    if flag == 0:
        n_y = len(y)
    if flag == 1:
        n_y = 1
    res = np.zeros((n_func, n_y))
    for i in range(n_func):
        for j in range(n_y):
            res[i, j] = derivative(t, i, j, y, z, n_func, flag)
            
    return res

#Переменное ускорение свободного падения
def g(t):
    return 9.81 + 0.05 * np.sin(2 * np.pi * t)

#Функции, соответствующие правым частям дифференциальных уравнений
def right_impl(t, y, z):
    return y[1], -y[0] / l / m * z, y[3], -y[2] / l / m * z - g(t)

#Функция условия
def cond(t, y, z):
    c = 10         #Переменная для стабилизации уравнения
    return (y[1]**2 + y[3]**2 - y[2] * g(t)) * l * m - z * (y[0]**2 + y[2]**2) + 2*c * (y[0] * y[1] + y[2] * y[3]) + c**2 *(y[0]**2 + y[2]**2 - l**2) 



# Параметры системы
m = 1                            # масса маятника
l = 5                            # длина маятника
x_0 = 3                          # начальная координата по х
y_0 = 4                          # начальная координата по у
v_0 = 1                          # начальная скорость

n_eq = 4                         # Число дифференциальных уравнений
n_point = 10000                   # Число расчетных точек
n_iter = 10                      # Число итераций в методе Ньютона
t_max = 100                        # Максимальное значение времени

# Инициализация массивов
t = np.linspace(0, t_max, n_point)                  # Задание сетки по времени с эквидистантным интевалом
h = t[1] - t[0]                                     # Временной интервал
diff_var = np.zeros((n_point, n_eq, n_iter))        # Переменные по которым происходит дифференцирование (x; v_х; y; v_y)
free_var = np.zeros((n_point, n_iter))              # Недифференцируемая переменная (Сила сопротивления стержня)  
right_part = np.zeros((n_point, n_eq, n_iter))      # Правая часть дифференциальных уравнений
eq = np.zeros((n_point, n_iter))                    # Функция, соответствующая условию

A = np.zeros((n_eq + 1, n_eq + 1))      # Вспомогательная переменная
df = np.zeros(5)                        # Вспомогательная переменная

# Задание начальных условий

diff_var[0, 0, n_iter-1] = x_0
diff_var[0, 1, n_iter-1] = v_0 * y_0 / l
diff_var[0, 2, n_iter-1] = y_0
diff_var[0, 3, n_iter-1] = -v_0 * x_0 / l
free_var[0, n_iter-1] = 0#-m / l * (g(0) * y_0 - v_0**2)

right_part[0, :, n_iter-1] = right_impl(t[0], diff_var[0, :, n_iter-1], free_var[0, n_iter-1])
eq[0, n_iter-1] = cond(t[0], diff_var[0, :, n_iter-1], free_var[0, n_iter-1])



# Задаем цикл по всей временной сетке
for i in range(1, n_point):
    
    #Задаем стандартную схему прогноза значений в начале итерации 
    if i == 1:
        diff_var[i, :, 0] = diff_var[i-1, :, n_iter-1]
        free_var[i, 0] = free_var[i - 1, n_iter-1]
        right_part[i, :, 0] = right_part[i-1, :, n_iter-1]
        eq[i, 0] = eq[i - 1, n_iter-1]
    if i > 1:
        diff_var[i, :, 0] = 2 * diff_var[i-1, :, n_iter-1] - diff_var[i-2, :, n_iter-1]
        free_var[i, 0] = 2 * free_var[i-1, n_iter-1] - free_var[i-2, n_iter-1]
        right_part[i, :, 0] = right_impl(t[i], diff_var[i, :, 0], free_var[i, 0])
        eq[i, 0] = cond(t[i], diff_var[i, :, 0], free_var[i, 0]) 
        
    #Находим матрицы Якоби и преобразуем их соответсвенно методу
    
    #Матрица Якоби для правых частей дифф уравнений и дифф переменных
    matrix_right_diff = np.eye(n_eq) - h * 0.5 * jacobi(t[i - 1], diff_var[i, :, 0], free_var[i, 0], n_eq, 0)
    
    #Матрица Якоби для правых частей дифф уравнений и недифф переменной
    matrix_right_free = -h * 0.5 * jacobi(t[i - 1], diff_var[i, :, 0], free_var[i, 0], n_eq, 1)
    
    #Матрица Якоби для функции условия и дифф переменных
    matrix_cond_diff = -jacobi(t[i - 1], diff_var[i, :, 0], free_var[i, 0], 1, 0)
    
    # Обозначим за А итоговую матрицу для решения системы алгебраических уравнений и заполним ее
    for j in range(n_eq + 1):
        for k in range(n_eq + 1):      
            if (j < 4 and k < 4):
                A[j, k] = matrix_right_diff[j, k]
            if (j == 4 and k < 4):
                A[j, k] = matrix_cond_diff[0, k]
            if (k == 4 and j < 4):
                A[j, k] = matrix_right_free[j, 0]            
            if (k == 4 and j == 4):
                # последний элемент матрицы находим как производную функции условия по недифф переменной
                A[4, 4] = -derivative(t[i-1], 0, 0, diff_var[i, :, 0], free_var[i, 0], 1, 1)
    
    #Найдем матрицу, обратную матрице А
    A = invert_matrix(A)
    
    # Задаем цикл по всем итерациям
    for k in range(1, n_iter):
        
        #Находим правую часть системы алгебраических уравнений df
        df[:n_eq] = h * 0.5 * (right_part[i-1, :, n_iter-1] + right_part[i, :, k-1]) - (diff_var[i, :, k-1] - diff_var[i-1, :, n_iter-1])
        df[-1] = eq[i, k-1]
        
        #Решение системы алгебраических уравнений
        diff_var[i, :, k] = diff_var[i, :, k - 1] + np.dot(A, df)[:n_eq]
        free_var[i, k] = free_var[i, k-1] + np.dot(A, df)[-1]
                
        # Определяем правые части дифф уравнений и функцию условия, как функции от ранее найденных переменных    
        right_part[i, :, k] = right_impl(t[i], diff_var[i, :, k], free_var[i, k])
        eq[i, k] = cond(t[i], diff_var[i, :, k], free_var[i, k])
        
    #Задаем стандартную схему прогноза значений для последней итерации         
    right_part[i, :, n_iter-1] = 2 / h * (diff_var[i, :, n_iter-1] - diff_var[i - 1, :, n_iter-1]) - right_part[i - 1, :, n_iter-1] 

#Построение графиков
plt.title('Зависимость координат от времени')
plt.plot(t, diff_var[:, 0, n_iter - 1], c = 'r', label = 'x(t)')
plt.plot(t, diff_var[:, 2, n_iter - 1], c = 'b', label = 'y(t)')
plt.plot(t, np.sqrt(diff_var[:, 2, n_iter - 1]**2 + diff_var[:, 0, n_iter - 1]**2), c = 'k', label = '$\sqrt{x^{2}(t) + y^{2}(t)}$')
plt.xlabel('Время, c', fontsize = 15)
plt.ylabel('Координата, м', fontsize = 15)
plt.legend(bbox_to_anchor = (1, 0.6))
