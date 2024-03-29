"""
Модуль реализует функциональность алгоритма машинного обучения MARS (Multivariate Adaptive Regression Splines),
предложенного Jerome H. Friedman в 1991 г.
### TODO дописать
"""

import copy
import numpy as np
import scipy.optimize
from numpy import linalg as LA
from sklearn.base import RegressorMixin, BaseEstimator, TransformerMixin
from itertools import compress


class Earth(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    Класс, реализующий MARS.

    Параметры
    ----------
    max_terms: int
        Максимальное число базисных функций.

    penalty: float, optional (default=3.0)
        Параметр сглаживания d в GCV. Чем больше, тем меньше узлов создаётся.
        C_new(M) = C(M) + d*M

    thresh: float
        Условия остановки прямого прохода.
        Если RSQ > 1-tresh или если RSQ увеличивается меньше, чем на tresh после очередной итерации.
        ### TODO Можно будет предложить ещё какие-то условия.

    smooth: bool, optional (default=False)
        Использовать ли вместо срезки кубические усечённые сплайны с непрерывными 1ми производными?

    allow_linear: bool, optional (default=False)
        Допускать ли добавление линейных ф-ций?

    allow_indicator: bool, optional (default=False)
        Допускать ли добавление кусочно-постоянных ф-ций?

    degree: int, optional(default=None)
        Использовать ли усечённые сплайны заданной степени.

    feature_importance_type: string or list of strings, optional (default=None)
        Критерии важности признаков ('gcv', 'rss', 'nb_subsets').
        По умолчанию не вычисляется.

    endspan_alpha: float, optional, probability between 0 and 1 (default=0.05)
        Вероятности сделать больше endspan отклонений.

    endspan: int, optional (default=-1)
        Ручное задание интервала между узлом и краем.

    minspan_alpha: float, optional, probability between 0 and 1 (default=0.05)
        Вероятности сделать больше minspan отклонений.

    minspan: int, optional (default=-1)
        Ручное задание минимального интервала между соседними узлами.

    enable_pruning: bool, optional (default=True)
        Делать ли обратный проход?

    verbose: bool, optional(default=False)
        Выводить ли дополнительную информацию?

    atol : float, optional (default=1e-12)
        Абсолютная точность. Нужен, в частности, для встречающихся в коде сравнений, а также для
        корректировки положительно определённых матриц, которые иногда в силу ошибок округления могут таковыми не оказаться.
        При появлении подобной ошибки уменьшить значения абсоютной и относительной точности.

    rtol : float, optional (default=1e-05)
        Относительная точность. Ситуации применения аналогичны atol.
        

        
    Атрибуты
    ----------
    ... TODO


    Методы
    ----------
    ... TODO
    ...
    """

    def __init__(
            self, max_terms=None, penalty=3,
            endspan_alpha=0.05, endspan=None,
            minspan_alpha=0.05, minspan=None,
            atol=1e-04, rtol=1e-05,
            allow_linear=False, enable_pruning=True,
            dropout=None, dropout_type='both',
            verbose=0):

        self.max_terms = max_terms ### TODO число б.ф. должно быть меньше, чем размерность объектов d, проверка
        self.penalty = penalty
        self.endspan_alpha = endspan_alpha
        self.endspan = endspan
        self.minspan_alpha = minspan_alpha
        self.minspan = minspan
        self.allow_linear = allow_linear
        self.enable_pruning = enable_pruning
        self.verbose = verbose ### TODO сделать похожий вывод на pyearth
        self.atol = atol
        self.rtol = rtol

        
        self.dropout = dropout # вероятность базисной функции быть выкинутой
        if not (dropout_type == 'both' or dropout_type == 'solo'):
            raise AttributeError
        self.dropout_type = dropout_type


        self.term_list_forward_ = None
        self.term_list_backward_ = None
        self.coeffs_forward_ = None
        self.coeffs_backward_ = None
        self.lof_value_forward_ = float('inf')
        self.lof_value_backward_ = float('inf')
        self.lof_value_predict_ = float('inf')
        self.metrics2value_forward_ = None
        self.metrics2value_backward_ = None
        self.metrics2value_predict_ = None


        # term_list = [B_1, ..., B_M] - список б.ф. (term)
        # B = [mult_1 , ... , mult_K] - список множителей (mult) б.ф. B
        self.term_list = None
        self.coeffs = None
        self.lof_func = self.mse_func
        self.lof_value = float('inf')
        self.train_data_count = None
        self.data_dim = None
        self.optimization_method = 'nelder-mead'
        self.metrics2value = None
        self.metrics2func = {
            'MSE': self.mse_func,
            'MAE': self.mae_func,
            #'GCV': self.gcv_func,
            #'RSQ': self.rsq_func,
            #'GRSQ': self.grsq_func,
            }


    # =====================================Вспомогательные классы и ф-ции============================


    class TermClass(list):
        """
        Класс, реализующий представление б.ф.
        Обёртка над списком множителей, составляющих б.ф.
        TODO дописать
        """

        def __init__(self, term, valid_coords=[]):
            super().__init__(term)
            self.valid_coords = valid_coords # список валидных координат

        def __repr__(self):
            s = ''
            for mult in self:
                s += '[' + str(mult) + '], '
            return s
        
        def add_mult(self, mult):
            """
            Метод добавления множителя к базисной ф-ции.
            Кроме простого добавления в список, происходит удаление
                уже использованной координаты из мн-ва валидных коорд-т.


            Параметры
            ----------
            mult: добавляемый множитель
            """
            self.append(mult)
            self.valid_coords.remove(mult.v)


    class TermListClass(list):
        """
        Класс, реализующий представление мн-ва б.ф.
        Обёртка над списком б.ф.
        TODO дописать
        """

        def __init__(self, term_list, coeffs=None):
            super().__init__(term_list)
            self.coeffs = coeffs

        def __repr__(self):
            s = ''
            for term_num, term in enumerate(self):
                s += str(term) + f'coeff={self.coeffs[term_num]}\n'
            return s

    
    def g_calculation(self, B, coeffs):
        '''
        Ф-ция, реализующая вычисление g(x):
            g(x) = a_1 * B_1(x) + ... + a_M * B_M(x)
            B_i - i-ая б.ф.
            a_i - коэфф-т при i-ой б.ф.

        Параметры
        ----------
        B: матрица, объекты-б.ф.
        coeffs: вектор, коэффициенты при б.ф.


        Выход
        ----------
        вектор, значения ф-ции g(x) на объектах выборки
        '''
        g = B @ coeffs
        return g


    def term_calculation(self, X, term):
        '''
        Ф-ция, реализующая вычисление б.ф. B_m:
            B_m(x) = mult_{m,1}(x) * ... * mult_{m,K_m}(x)

        Параметры
        ----------
        X: матрица, выборка
        term: TermClass, б.ф. B_m

        
        Выход
        ----------
        вектор, значения б.ф. B_m на объектах выборки
        '''
        term_values = 1
        for mult in term:
            term_values *= mult.calculate(X)
        return term_values


    def b_calculation(self, X, term_list):
        '''
        Ф-ция, реализующая вычисление матрицы B размера NxM:
            B[i,m] = B_m(x_i)
            N - кол-во объектов
            M - кол-во б.ф.

        Параметры
        ----------
        X: матрица, выборка
        term_list: TermListClass, [B_1, ..., B_M] - набор б.ф.


        Выход
        ----------
        матрица B
        '''
        data_count = X.shape[0]
        term_count = len(term_list)
        B = np.empty((data_count, term_count))

        for ind in range(term_count):
            B[:, ind] = self.term_calculation(X, term_list[ind])
        return B


    def c_calculation(self, V_norm, B, b_mean=0.):
        '''
        Ф-ция, вычисляющая поправочный коэффициент C_correct(M).
            C(M) = trace(B @ (B.T @ B)^(-1) @ B.T) + 1
            C_correct(M) = C(M) + d*M
            B[i,m] = B_m(x_i)
        TODO добавить простую эвристику вычисления C из книжки
        TODO правильно ли я понял про усреднение мат. B?
            
        Параметры
        ----------
        B: матрица, объекты-б.ф.
        V_norm: матрица, скорректированная симм. полож. опред. V_norm = B.T @ (B - b_mean)
        b_mean: вектор, среднее по объектам мат. B (по умолчанию = 0)


        Выход
        ----------
        float, поправочный коэффициент C_correct(M)
        '''

        # Т.к. в V уже использована нормализованная мат. B (V = B^T @ (B - b_mean)) =>
        #   мат. B, которая используется в формуле для C(M), также должна быть нормирована
        #   и только она (т.е. B.T без изменений)
        
        term_count = B.shape[1]
        B_norm = B - b_mean

        C = np.trace(B_norm @ LA.inv(V_norm) @ B.T) + 1
        C_correct = C + self.penalty * term_count
        return C_correct
    

    def metrics_calculation(self, y, B, coeffs, metrics2func, **kwargs):
        """
        Ф-ция, вычисляющая значения метрик.

        Параметры
        ----------
        y: вектор, ответы на объектах
        B: матрица, объекты-б.ф.
        coeffs: вектор, коэфф-ты при б.ф.
        metrics2func: dict(str2func), словарь название метрики - соотв. ф-ция


        Выход
        ----------
        dict(str2float), словарь название метрики - её значение
        """
        metrics2value = {}

        for name_metric, func in metrics2func.items():
            metrics2value[name_metric] = func(y, B, coeffs, **kwargs)

        return metrics2value
    

    def minspan_endspan_calculation(self, nonzero_count, data_dim):
        '''
        Ф-ция, вычисляющая L(alpha) и Le(alpha):
            L(alpha)  - задаёт шаг из порогов между соседними узлами (t)
            Le(alpha) - задаёт отступ из порогов для граничных узлов
        Смысл - сглаживание скользящим окном.
            
        Параметры
        ----------
        nonzero_count: int, число объектов, на которых б.ф. B_m != 0
        data_dim: int, размерность объектов


        Выход
        ----------
        (minspan, endspan): (int, int), значения L и Le соотв.
        '''
        minspan = self.minspan
        endspan = self.endspan

        if minspan == None:
            minspan = int(-np.log2(-1 / (data_dim * nonzero_count) *
                          np.log(1 - self.minspan_alpha)) / 2.5)

        if endspan == None:
            endspan = int(3 - np.log2(self.endspan_alpha / data_dim))

        return (minspan, endspan)


    def mse_func(self, y, B, coeffs, **kwargs):
        """
        Mean Squared Error (MSE):
            MSE = 1/N * sum([y_i - f(x_i)]^2)
            f(x) = g(x) = B @ a

        Параметры
        ----------
        y: вектор, ответы на объектах
        B: матрица, объекты-б.ф.
        coeffs: вектор, коэфф-ты при б.ф.


        Выход
        ----------
        float, значение MSE
        """
        y_pred = self.g_calculation(B, coeffs)
        mse = np.mean((y - y_pred) ** 2)
        return mse
    

    def mae_func(self, y, B, coeffs, **kwargs):
        """
        Mean Absolute Error (MAE):
            MAE = 1/N * sum(|y_i - f(x_i)|)
            f(x) = g(x) = B @ a

        Параметры
        ----------
        y: вектор, ответы на объектах
        B: матрица, объекты-б.ф.
        coeffs: вектор, коэфф-ты при б.ф.


        Выход
        ----------
        float, значение MAE
        """
        y_pred = self.g_calculation(B, coeffs)
        mae = np.mean(np.abs(y - y_pred))
        return mae
    

    def rss_func(self, y, B, coeffs, **kwargs):
        """
        Residual Sum of Squares (RSS):
           RSS = sum([y_i - f(x_i)]^2)
           f(x) = g(x) = B @ a

        Параметры
        ----------
        y: вектор, ответы на объектах
        B: матрица, объекты-б.ф.
        coeffs: вектор, коэфф-ты при б.ф.


        Выход
        ----------
        float, значение RSS
        """
        y_pred = self.g_calculation(B, coeffs)
        rss = np.sum((y - y_pred) ** 2)
        return rss
    

    def rsq_func(self, y, B, coeffs, **kwargs):
        """
        TODO дописать
        R SQuared (RSQ):
           RSQ = 
           f(x) = g(x) = B @ a

        Параметры
        ----------
        y: вектор, ответы на объектах
        B: матрица, объекты-б.ф.
        coeffs: вектор, коэфф-ты при б.ф.


        Выход
        ----------
        float, значение RSQ
        """
        y_pred = self.g_calculation(B, coeffs)
        rsq = None
        return rsq
    

    def grsq_func(self, y, B, coeffs, **kwargs):
        """
        TODO дописать
        General R SQuared (GRSQ):
           GRSQ = 
           f(x) = g(x) = B @ a

        Параметры
        ----------
        y: вектор, ответы на объектах
        B: матрица, объекты-б.ф.
        coeffs: вектор, коэфф-ты при б.ф.


        Выход
        ----------
        float, значение GRSQ
        """
        y_pred = self.g_calculation(B, coeffs)
        grsq = None
        return grsq
        

    def gcv_func(self, y, B, coeffs, V_norm, b_mean=0., **kwargs):
        '''
        TODO реализовать в виде класса
        Generalized Cross-Validation criterion (GCV):
            GCV(M) = 1/N * sum([y_i - f(x_i)]^2) / [1 - C_correct(M)/N]^2 <=>
            GCV(M) = MSE / const_correct(M)
            C_correct(M) = C(M) + d*M
            C(M) = trace(B @ (B.T @ B)^(-1) @ B.T) + 1
            f(x) = g(x) = B @ a
            Смысл C: число лин. нез. б.ф.
            Смысл GCV: скорректированный MSE, учитывающий возрастание дисперсии, вызванное увеличением кол-ва б.ф. 
        
        Параметры
        ----------
        y: вектор, ответы на объектах
        B: матрица, объекты-б.ф.
        coeffs: вектор, коэфф-ты при б.ф.
        V_norm: матрица, скорректированная симм. полож. опред. V_norm = B.T @ (B - b_mean)
        b_mean: вектор, усреднённая по объектам мат. B (по умолчанию = 0)


        Выход
        ----------
        float, значение GCV
        '''
        ### TODO исправить с учётом передачи усреднённых значений
        data_count = y.size
        mse = self.mse_func(B, y, coeffs)
        C = self.c_calculation(V_norm, B, b_mean=b_mean)
        gcv = mse / (1 - C / data_count)**2
        return gcv


    def minimize(self, f, x0, args, method='nelder-mead', options={'disp': True}):
        """
        Ф-ция численной минимизации. Обёртка над scipy.optimize.minimize

        Параметры
        ----------
        ### TODO посмотреть другие параметры
        f: минимизируемая ф-ция вида f(x, *args) -> float
        x0: начальное приближение
        args: доп. аргументы ф-ции f
        method: метод оптимизации
        options: доп. опции


        Выход
        ----------
        """
        argmin = scipy.optimize.minimize(f, x0, args=args, method=method, options=options)
        return argmin


    def analytically_pseudo_solves_slae(self, V, c):
        """
        Ф-ция, аналитически вычисляющая псевдорешение СЛАУ
          методом наименьших квадратов на основе метода Холецкого.
        Используется для нахождения коэфф-ов.
        V @ a = c, V = L @ L^T
            V - матрица, скорректированная симм. полож. опред.
            L - матрица, нижн. треуг.

        Параметры
        ----------
        V: матрица, симм. полож. опред. (подразумевается, что V = B.T @ B)
        c: вектор (подразумевается, что c = B.T @ y)
            B - матрица, объекты-б.ф.
            y - вектор, ответы на объектах


        Выход
        ----------
        вектор, коэфф-ты при б.ф.
        """
        # V @ a = (L @ L^T) @ a = L @ (L^T @ a) = L @ b = c
        #   b = L^T @ a

        # Разложение методом Холецкого
        #L = self.modified_cholesky(V, c, L)
        L = LA.cholesky(V)

        # Решение системы L @ b = c
        b = LA.solve(L, c)

        # Решение системы L^T @ a = b
        a = LA.solve(L.T, b)

        ### TODO Мб есть оптимизатор, в котором можно явно выбрать метод Холецкого?
        return a
    

    def coeffs_and_lof_calculation(self, A, y, term_list, for_X=True, need_lof=True):
        """
        Ф-ция, вычисляющая коэфф-ты при б.ф. и итоговое зн-ие lof.

        Параметры
        ----------
        A: матрица
            если for_X=True  => выборка X
            если for_X=False => матрица объекты-б.ф. B
        y: вектор, ответы на объектах
        term_list: TermListClass, набор б.ф.
        for_X: bool, явл-ся ли матрица A выборкой X? (по умолчанию True)
               Иначе считается, что A - матрица объекты-б.ф. B.
        need_lof:  bool, нужно ли вычислять lof? (по умолчанию True)


        Выход
        ----------
        (coeffs, lof): (вектор, float), коэфф-ты при б.ф. и зн-ие lof
        """
        if for_X:
            X = A
            B = self.b_calculation(X, term_list)
        else:
            B = A

        V = self.symmetric_positive_matrix_correct(B.T @ B)
        c = B.T @ y

        coeffs = self.analytically_pseudo_solves_slae(V, c)

        lof = None
        if need_lof:
            lof = self.lof_func(y, B, coeffs, V=V)

        return (coeffs, lof)
    

    def symmetric_positive_matrix_correct(self, A):
        """
        Ф-ция, корректирующая потенциально симметричную положительно определённую матрицу,
          добавляя к ней диагональную матрицу, определяемую атрибутами точности: atol rtol.
        Далее происходит проверка достижения желаемого рез-та.
        В противном случае выбрасывается исключение и рекомендуется уменьшить точность.
        
        Параметры
        ----------
        A: матрица, потенциально симм. полож. опред.


        Выход
        ----------
        матрица, симм. полож. опред.
        """

        A += np.diag(self.atol + np.diag(A) * self.rtol)

        # проверка на симметричность
        if not np.allclose(A, A.T, rtol=self.rtol, atol=self.atol):
            raise LA.LinAlgError('Asymmetric matrix')

        # проверка на положительную определённость
        if not np.all(LA.eigvalsh(A) > 0):
            raise LA.LinAlgError('Matrix is not positive definite')

        return A
    

    def modified_cholesky(self, V, L_prev):
        """
        Ф-ция, реализующая разложение Холецкого с частичным пересчётом элементов диагональной матрицы L.

        Параметры
        ----------
        V: матрица, симм. полож. опред.
        L_prev: матрица, треуг. с прошлой итерации


        Выход
        ----------
        матрица, треуг.
        """
        pass


    class ConstantFunc():
        """
        Класс, реализующий константную ф-цию.
            f(x) = const

        Атрибуты
        ----------
        value: значение
        """

        def __init__(self, value=1.):
            self.value = value

        def __repr__(self):
            ### TODO во всех классах переопределить __class__:
            #return f'{self.__class__} value={self.value}'
            return f'<ConstantFunc> value={self.value}'

        def calculate(self, X=None):
            '''
            Ф-ция, вычисляющая константную ф-цию.

            Параметры
            ----------
            X: матрица, выборка (не используется)
            '''
            return self.value

    
    class BaseSplineFunc():
        """
        Класс-родитель для сплайновых ф-ций, использующихся в качестве множителей в б.ф.

        Атрибуты
        ----------
        s: знак
        v: координата
        t: порог
        """

        def __init__(self, s, v, t):
            self.s = s
            self.v = v
            self.t = t

        def __repr__(self):
            return f'<BaseSplineFunc> s={self.s}, v={self.v}, t={self.t:.3f}'


    class IndicatorFunc(BaseSplineFunc):
        """
        Класс, реализующий индикаторную ф-цию.
            f(x) = s * [x[v] - t]
                [...] - скобка Айверсона

        Атрибуты
        ----------
        s: знак
        v: координата
        t: порог
        """

        def __init__(self, s, v, t):
            super.__init__(s, v, t)

        def __repr__(self):
            return f'<IndicatorFunc> s={self.s}, v={self.v}, t={self.t:.3f}'

        def calculate(self, X):
            '''
            Ф-ция, вычисляющая индикаторную ф-цию.

            Параметры
            ----------
            X: матрица, выборка
            '''
            indicator = ((self.s * (X[:, self.v] - self.t)) > 0.) * 1.
            return indicator


    class LinearFunc(BaseSplineFunc):
        """
        Класс, реализующий линейную ф-цию.
            f(x) = s * (x[v] - t)

        Атрибуты
        ----------
        s: знак
        v: координата
        t: порог
        """

        def __init__(self, s, v, t):
            super().__init__(s, v, t)

        def __repr__(self):
            return f'<LinearFunc> s={self.s}, v={self.v}, t={self.t:.3f}'

        def calculate(self, X):
            '''
            Ф-ция, вычисляющая линейную ф-цию.

            Параметры
            ----------
            X: матрица, выборка
            '''
            linear = self.s * (X[:, self.v] - self.t)
            return linear


    class ReluFunc(BaseSplineFunc):
        """
        Класс, реализующий положительную срезку.
            f(x) = [s * (x[v] - t)]_+ = ReLU(x[v] - t)
                [...]_+ - положительная срезка

        Атрибуты
        ----------
        s: знак
        v: координата
        t: порог
        """

        def __init__(self, s, v, t):
            super().__init__(s, v, t)

        def __repr__(self):
            return f'<ReluFunc> s={self.s}, v={self.v}, t={self.t:.3f}'

        def calculate(self, X):
            '''
            Ф-ция, вычисляющая положительную срезку.

            Параметры
            ----------
            X: матрица, выборка
            '''
            linear = self.s * (X[:, self.v] - self.t)
            relu = np.maximum(linear, 0)
            return relu
        

    class PowReluFunc(BaseSplineFunc):
        """
        Класс, реализующий степенную положительную срезку.
            f(x) = ([s * (x[v] - t)]_+)^q

        Атрибуты
        ----------
        s: знак
        v: координата
        t: порог
        q: степень
        """

        def __init__(self, s, v, t, q):
            super().__init__(s, v, t)
            self.q = q

        def __repr__(self):
            return f'<PowReluFunc> s={self.s}, v={self.v}, t={self.t:.3f}, q={self.q}'

        def calculate(self, X):
            '''
            Ф-ция, вычисляющая степенную положительную срезку.

            Параметры
            ----------
            X: матрица, выборка
            '''
            linear = self.s * (X[:, self.v] - self.t)
            relu = np.maximum(linear, 0)
            pow_relu = relu ** self.q
            return pow_relu


    class CubicFunc(BaseSplineFunc):
        """
        Класс, реализующий кубический усечённый сплайн с непрерывной 1ой производной.
            TODO Привести вид такой ф-ции или хотя бы её концепцию.
        
        Атрибуты
        ----------
        s: знак
        v: координата
        t: порог
        t_minus:
        t_plus:
        """

        def __init__(self, s, v, t, t_minus, t_plus):
            super().__init__(s, v, t)
            self.t_minus = t_minus
            self.t_plus  = t_plus

        def __repr__(self):
            return (f'<CubicFunc> s={self.s}, v={self.v}, t={self.t:.3f},'
                    f't_minus={self.t_minus:.3f}, t_plus={self.t_plus:.3f}')

        def calculate(self, X):
            '''
            Ф-ция, вычисляющая кубический усечённый сплайн.

            Параметры
            ----------
            X: матрица, выборка
            '''
            # точки, в которых X[v] удовлетворяет условиям:
            # X[v] <= t_minus:
            mask_less = X[:, self.v] <= self.t_minus
            # t_minus < X[v] < t_plus:
            mask_between = self.t_minus < X[:, self.v] < self.t_plus
            # X[v] >= t_plus:
            mask_greater = X[:, self.v] >= self.t_plus

            # знак положительный
            if self.s > 0:
                p_plus = (2*self.t_plus + self.t_minus - 3*self.t) / (self.t_plus - self.t_minus) ** 2
                r_plus = (2*self.t - self.t_plus - self.t_minus) / (self.t_plus - self.t_minus) ** 3

                return mask_greater * (X[:, self.v] - self.t) + \
                       mask_between * (p_plus * (X[:, self.v] - self.t_minus) ** 2 +
                                       r_plus * (X[:, self.v] - self.t_minus) ** 3)
            
            # знак отрицательный
            p_minus = (3*self.t - 2*self.t_minus - self.t_plus) / (self.t_minus - self.t_plus) ** 2
            r_minus = (self.t_minus + self.t_plus - 2*self.t) / (self.t_minus - self.t_plus) ** 3

            return mask_less * (self.t - X[:, self.v]) + \
                   mask_between * (p_minus * (X[:, self.v] - self.t_plus) ** 2 +
                                   r_minus * (X[:, self.v] - self.t_plus) ** 3)


    def classic_knot_optimization(self, X, y, B, fixed_values, **kwargs):
        """
        Ф-ция находит лучший узел t при фиксированной б.ф. B_m и координате v.
        Классический переборный алгоритм.

        Параметры
        ----------
        X: матрица, обучающая выборка
        y: вектор, ответы на объектах
        B: матрица, объекты-б.ф.
        fixed_values: (m - int, v - int) - фикс. параметры: номер фикс. б.ф. и номер фикс. коор-ты v


        Выход
        ----------
        (best_t, best_lof): (float, float), лучший порог t и лучшее значение lof при фикс. параметрах
        """
        ### TODO дописать
        pass


    def knot_optimization(self, X, y, B, fixed_values, sorted_features, splines_type='default', **kwargs):
        """
        Ф-ция находит лучший узел t при фиксированной б.ф. B_m и координате v.

        Параметры
        ----------
        X: матрица, обучающая выборка
        y: вектор, ответы на объектах
        B: матрица, объекты-б.ф.
        fixed_values: (m - int, v - int) - фикс. параметры: номер фикс. б.ф. и номер фикс. коор-ты v
        sorted_features: вектор, ненулевые зн-я объектов по фикс. коорд., отсортированные по возрастанию
        splines_type: ...


        Выход
        ----------
        (best_t, best_lof): (float, float), лучший порог t и лучшее значение lof при фикс. параметрах
        """
        N, d = X.shape # N - кол-во объектов, d - размерность объектов
        m, v = fixed_values

        x_v = X[:, v]
        B_m = B[:, m]

        # B_extnd - расширенная матрица B, она не явл-ся матрицей объекты-б.ф.
        #   т.к. в последних двух столбцах используются не б.ф.
        B_extnd = np.hstack([B, np.empty((N, 2))])

        # Заполняем последние 2 столбца матрицы B_extnd 2-мя последними слагаемыми из g' (они не б.ф.)
        B_extnd[:, -2] = B_m * x_v  # B_m(x) * x[v]
        B_extnd[:, -1] = 0          # B_m(x) * [x[v] - t]_+ == 0 (t - самый крайний правый порог)
        

        # Прореживание мн-ва перебираемых порогов
        N_m = np.count_nonzero(B_m)
        minspan, endspan = self.minspan_endspan_calculation(N_m, d)
        thin_sorted_features = sorted_features[endspan:-endspan:minspan]
        if thin_sorted_features.size == 0:
            return (None, float('inf'))
        t = thin_sorted_features[-1]


        # Нормализация:
        #   B @ a = y => {нормализация} => (B - b_mean) @ a = y - y_mean --->
        #   ---> {B.T @ (...)} ---> [B.T @ (B - b_mean)] @ a = B.T @ (y - y_mean) <=>
        #   <=> V @ a = c
        y_mean = np.mean(y)
        b_extnd_mean = np.mean(B_extnd, axis=0)

        V_extnd = self.symmetric_positive_matrix_correct(B_extnd.T @ (B_extnd - b_extnd_mean))
        c_extnd = B_extnd.T @ (y - y_mean)


        # Находим коэфф-ты - решение нормализованной задачи V @ a = c
        coeffs_extnd = self.analytically_pseudo_solves_slae(V_extnd, c_extnd)

        best_lof = self.lof_func(y - y_mean, B_extnd - b_extnd_mean, coeffs_extnd, V=V_extnd, b_mean=b_extnd_mean)
        best_t = t

    
        # Цикл по порогам от больших к меньшим, t <= u
        u = t
        s_u = 0
        for t in thin_sorted_features[-2::-1]:
            between_inds = np.nonzero((t <= x_v) & (x_v < u))[0]
            greater_inds = np.nonzero(x_v >= u)[0]

            # Обновляем последнее слагаемое в g', которое одно зав-т от порога t
            B_extnd[:, -1] = B_m * np.maximum(x_v - t, 0)  # B_m(x) * [x[v] - t]_+
            b_extnd_mean[-1] = np.mean(B_extnd[:, -1])

            # c[M+1]
            c_extnd[-1] += np.sum((y[between_inds] - y_mean) * B_m[between_inds] *
                                  (x_v[between_inds] - t)) + \
                           (u - t) * np.sum((y[greater_inds] - y_mean) * B_m[greater_inds])

            # V[i,M+1], i = 1..M => {симметричность матрицы V} => V[M+1,i], i = 1..M
            V_extnd[:-1, -1] += np.sum((B_extnd[between_inds, :-1] - b_extnd_mean[:-1]) * 
                                       B_m[between_inds][:, np.newaxis] *
                                       (x_v[between_inds] - t)[:, np.newaxis], axis=0) + \
                                (u - t) * np.sum((B_extnd[greater_inds, :-1] - b_extnd_mean[:-1]) *
                                                 B_m[greater_inds][:, np.newaxis], axis=0)
            V_extnd[-1, :-1] = V_extnd[:-1, -1]
            
            # V[M+1,M+1]
            s_greater_inds = np.nonzero(x_v >= t)[0]
            s_t = np.sum(B_m[s_greater_inds] * (x_v[s_greater_inds] - t))

            V_extnd[-1, -1] += np.sum((B_m[between_inds] * (x_v[between_inds] - t)) ** 2) + \
                               (u - t) * np.sum(B_m[greater_inds] ** 2 *
                                                (2 * x_v[greater_inds] - t - u)) + \
                               (s_u ** 2 - s_t ** 2) / N


            V_extnd = self.symmetric_positive_matrix_correct(V_extnd)

            # Находим коэфф-ты - решение нормализованной задачи V @ a = c
            # Они отвечают ф-ции g' и отличаются от оптимальных для набора б.ф.
            # При этом зн-я lof* моделей g и g' на своих оптимальных наборах коэфф-тов совпадают =>
            #   => совпадают оптимальные пороги t*
            coeffs_extnd = self.analytically_pseudo_solves_slae(V_extnd, c_extnd)

            # Находим lof*
            ### Раньше передавал коэфф-ты нормализованной задачи в НЕ нормализаованную, а это, оказывается, не экв-но:
            ###   lof = self.lof_func(y, B_extnd, coeffs_extnd, V=V_extnd, b_mean=b_extnd_mean)
            lof = self.lof_func(y - y_mean, B_extnd - b_extnd_mean, coeffs_extnd, V=V_extnd, b_mean=b_extnd_mean)

            if lof < best_lof:
                best_lof = lof
                best_t = t
            
            #print(f'v: {v}, m: {m}, t: {t}, lof: {lof}, best_lof: {best_lof}')

            u = t
            s_u = s_t

        return (best_t, best_lof)


    # ==================================================Функциональность==========================================================


    def fit(self, X, y=None):
        """
        Обучение модели.


        Параметры
        ----------
        X: array-like, shape = [m, n], где m - кол-во объектов, n - кол-во признаков
            Обучающая выборка.

        y: array-like, optional (default=None), shape = [m], где m - кол-во объектов
            Ответы на объектах.
        """
        self.forward_pass(X, y)

        if self.enable_pruning:
            self.pruning_pass(X, y)
            
        return self
        
        
    def forward_pass(self, X, y=None):
        """
        Проход вперёд.


        Параметры
        ----------
        X: array-like, shape = [m, n], где m - кол-во объектов, n - кол-во признаков
            Обучающая выборка.

        y: array-like, optional (default=None), shape = [m], где m - кол-во объектов
            Ответы на объектах.
        """

        # Обозначения из оригинальной статьи:
        # g(x) - модель:
        #   g(x) = a_1 * B_1(x) + ... + a_M * B_M(x)
        #     x = (x_1 , ... , x_d) - объект из обуч. выборки
        #       N - кол-во объектов ---> data_count
        #       d - размерность объекта ---> data_dim
        #     B_i - i-ая б.ф. ---> term
        #     M - кол-во б.ф. ---> term_count
        #     a_i - коэф-т при i-ой б.ф.
        #     B(x) = mult_1 * ... * mult_K
        #     B_1 = const_func
        #       mult - множитель б.ф.
        #       Виды множителей:
        #       1) [s * (x[v] - t)]_+     - положительная срезка
        #       2) ([s * (x[v] - t)]_+)^q - степенная положительная срезка
        #       3) [s * (x[v] - t)]       - индикаторная ф-ция
        #       4) s * (x[v] - t)         - линейная ф-ция
        #       5) const_func             - константная ф-ция
        #       6) cubic_func             - кубический сплайн с 1ой непр-ой произв-ой
        #         [...]   - скобка Айверсона
        #         [...]_+ - положительная срезка
        #         K - кол-во множителей в б.ф.
        #         s - знак множителя
        #         v - координата x множителя
        #         t - порог множителя

        self.train_data_count, self.data_dim = X.shape

        # Создание первой константной б.ф.
        const_term = self.TermClass([self.ConstantFunc(1.)], valid_coords=list(range(self.data_dim)))
        self.term_list = self.TermListClass([const_term, ])

        # Создание матрицы объекты-б.ф., её заполнение для константной б.ф.
        B = np.ones((self.train_data_count, 1))

        best_lof = None
        term_count = 2  # M <- 2
        is_dropped = self.dropout is not None

        # создаём б.ф. пока не достигнем макс. кол-ва
        while term_count <= self.max_terms:
            best_lof = float('inf')  # lof* <- +inf
            best_term_num = None
            best_term = None
            best_v = None
            best_t = None

            # перебираем уже созданные б.ф.
            for term_num, term in enumerate(self.term_list):

                # перебираем все ещё не занятые координаты v в б.ф. term
                for v in term.valid_coords:

                    # сортируем мн-во порогов, соотв. объекты которых не равны 0 на б.ф. term
                    x_sorted = np.sort(X[B[:, term_num] != 0, v])

                    # находим лучший порог t и значение lof при фикс. б.ф. term и координате v
                    fixed_values = (term_num, v)
                    t, lof = self.knot_optimization(X, y, B, fixed_values, x_sorted)
                    #t, lof = self.classic_knot_optimization(X, y, B, fixed_values, x_sorted)

                    if lof < best_lof:
                        best_lof  = lof
                        best_term_num = term_num
                        best_term = term
                        best_v = v
                        best_t = t
            
            # создаём новые множители ReLU
            mult_plus  = self.ReluFunc(+1, best_v, best_t)
            mult_minus = self.ReluFunc(-1, best_v, best_t)

            # создаём новые б.ф.
            term_plus  = copy.deepcopy(best_term)  # B_M
            term_minus = copy.deepcopy(best_term)  # B_{M+1}
            term_plus.add_mult(mult_plus)
            term_minus.add_mult(mult_minus)

            # добавляем новые б.ф. к уже имеющимся
            ### TODO а если макс. кол-во функций будет равно 3, то что тогда?
            self.term_list.append(term_plus)
            self.term_list.append(term_minus)

            # добавляем к матрице B новые столбцы
            b_term_plus  = (B[:, best_term_num] * np.maximum(+1 * (X[:, best_v] - best_t), 0))[:, np.newaxis]
            b_term_minus = (B[:, best_term_num] * np.maximum(-1 * (X[:, best_v] - best_t), 0))[:, np.newaxis]
            B = np.hstack((B, b_term_plus, b_term_minus))

            term_count += 2  # M <- M + 2

            # блок с применением dropout'a
            if (term_count > self.max_terms and is_dropped
                and 0. <= self.dropout <= 1.):
                is_dropped = False

                if self.dropout_type == 'both':
                    mask = np.random.binomial(n=1, p=self.dropout, size=B.shape[1] // 2)
                    mask = np.repeat(mask, 2)
                else:
                    mask = np.random.binomial(n=1, p=self.dropout, size=B.shape[1]-1)

                mask = np.insert(mask, 0, 0)
                mask = (mask == 0)
                B = B[:, mask]
                term_count -= np.sum(~mask)
                self.term_list = self.TermListClass(self, compress(self.term_list, mask))


        # Нахождение коэфф-ов модели, построенной на шаге вперёд
        self.coeffs, self.lof_value = self.coeffs_and_lof_calculation(B, y, self.term_list, for_X=False)
        self.term_list.coeffs = self.coeffs
        self.metrics2value = self.metrics_calculation(y, B, self.coeffs, self.metrics2func)


        # Заполняем информационные атрибуты для прохода вперёд
        self.term_list_forward_ = self.term_list
        self.lof_value_forward_ = self.lof_value
        self.coeffs_forward_ = self.coeffs
        self.metrics2value_forward_ = self.metrics2value

        return self


    def pruning_pass(self, X, y=None):
        """
        Проход назад.


        Параметры
        ----------
        X: array-like, shape = [m, n], где m - кол-во объектов, n - кол-во признаков
            Обучающая выборка.

        y: array-like, optional (default=None), shape = [m], где m - кол-во объектов
            Ответы на объектах.
        """
        # названия переменных взяты из статьи
        M_max = len(self.term_list)

        best_J = copy.deepcopy(self.term_list)  # глобально лучший набор б.ф.
        best_K = copy.deepcopy(best_J)  # локально лучший набор б.ф.
        best_lof = self.lof_value  # глобально лучшее значение LOF
        best_coeffs = self.coeffs

        for M in range(M_max, 1, -1): # M_max, M_max-1, ..., 2
            b = float('inf')  # локально лучшее значение LOF
            L = copy.deepcopy(best_K)

            # для очередного прореженного списка поочерёдно удаляем входящие в него б.ф.
            #   для определения кандидата на удаление
            for m in range(1, M):
                K = copy.deepcopy(L)
                K.pop(m)

                # находим оптимальные коэфф-ты, считаем lof без очередной б.ф.
                coeffs, lof = self.coeffs_and_lof_calculation(X, y, K)

                if lof < b:
                    # локальное улучшение
                    b = lof
                    best_K = K

                if lof <= best_lof:
                    # глобальное улучшение
                    best_J = K
                    best_lof = lof
                    best_coeffs = coeffs


        self.term_list = best_J
        self.lof_value = best_lof
        self.coeffs = best_coeffs
        self.term_list.coeffs = self.coeffs

        B = self.b_calculation(X, self.term_list)
        self.metrics2value = self.metrics_calculation(y, B, self.coeffs, self.metrics2func)


        # Заполняем информационные атрибуты для прохода назад
        self.term_list_backward_ = self.term_list
        self.lof_value_backward_ = self.lof_value
        self.coeffs_backward_ = self.coeffs
        self.metrics2value_backward_ = self.metrics2value
        
                    
        return self
    

    def linear_fit(self, X, y=None):
        """
        Определение коэффициентов для линеной модели методом наименьших квадратов.

        Параметры
        ----------
        X: array-like, shape = [m, n], где m - кол-во объектов, n - кол-во признаков
            Обучающая выборка.

        y: array-like, optional (default=None), shape = [m], где m - кол-во объектов
            Ответы на объектах.
        """
        ### TODO реализовать (обёртка над обычной sklearn регр)
        pass


    def predict(self, X):
        """
        Предсказание модели на входных данных.

        Параметры
        ----------
        X: array-like, shape = [m, n], где m - кол-во объектов, n - кол-во признаков
            Входные данные, по которым требуется сделать прогноз.


        Выход
        ----------
        y: array of shape = [m], где m - кол-во объектов
            Прогноз.
        """
        B = self.b_calculation(X, self.term_list)
        y_pred = self.g_calculation(B, self.coeffs)
        return y_pred


    def transform(self, X):
        """
        Переход в пространство базисных функций.

        Параметры
        ----------
        X: array-like, shape = [m, n], где m - кол-во объектов, n - кол-во признаков
            Входные непреобразованные данные.


        Выход
        ----------
        B: array of shape [m, nb_terms], где m - кол-во объектов, nb_terms - кол-во
        получившихся базисных функций.
            Преобразованные данные.
        """
        B = self.b_calculation(X, self.term_list)
        return B


    # ========================================================Вывод информации=======================================================================


    class InfoClass():
        """
        Класс, реализующий удобный отладочный интерфейс.

        TODO Подумать над структурой
        """
        pass


    def forward_trace(self):
        '''
        Вывод информации о проходе вперёд.
        '''
        ### TODO реализовать
        pass


    def pruning_trace(self):
        '''
        Вывод информации о проходе назад.
        '''
        ### TODO реализовать
        pass


    def trace(self):
        '''
        Вывод информации о проходе вперёд и назад.
        '''
        self.forward_trace()
        self.pruning_trace()


    def summary(self):
        """
        Описание модели в виде строки.
        """
        ### TODO дописать
        return self.term_list


    def summary_feature_importances(self, sort_by=None):
        """
        Важность признаков в виде строки.


        Параметры
        ----------
        sory_by: string, optional
            Сортировка, если поддерживается, по feature_importance_type ('rrs', 'gcv', 'nb_subsets').
        """
        pass
