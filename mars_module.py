import copy
import numpy as np
import scipy.optimize
from numpy import linalg as LA
from sklearn.base import RegressorMixin, BaseEstimator, TransformerMixin


class Earth(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    Класс, реализующий MARS.


    ### Вся функциональность из py-earth должна быть сюда перенесена.
    ### По возможности надо сохранить оригинальные названия аргументов, атрибутов, методов и т.д.


    Параметры
    ----------
    max_terms : int
        Максимальное число базисных функций.


    max_degree : int
        Максимальная степень множителей базисных функций.


    penalty : float, optional (default=3.0)
        Параметр сглаживания d в GCV (чем больше, тем меньше узлов создаётся).
        C_new(M) = C(M) + d*M


    thresh : float
        Условия остановки прямого прохода.
        Если RSQ > 1-tresh или если RSQ увеличивается меньше, чем на tresh после очередной итерации.
        ### TODO Можно будет предложить ещё какие-то условия.


    smooth : bool, optional (default=False)
        Использовать ли вместо срезки кубические усечённые сплайны с непрерывными 1ми производными?


    allow_linear : bool, optional (default=True)
         Допускать ли добавление линейных ф-ций?


    allow_ : bool, optional (default=False)
    ### TODO Допускать ли добавление кусочно-постоянных ф-ций?


    feature_importance_type: string or list of strings, optional (default=None)
        Критерии важности признаков ('gcv', 'rss', 'nb_subsets').
        По умолчанию не вычисляется.
        ### TODO Можно будет добавить свои критерии важности.


    endspan_alpha : float, optional, probability between 0 and 1 (default=0.05)
        Вероятности сделать больше endspan отклонений.


    endspan : int, optional (default=-1)
        Ручное задание интервала между узлом и краем.


    minspan_alpha : float, optional, probability between 0 and 1 (default=0.05)
        Вероятности сделать больше minspan отклонений.


    minspan : int, optional (default=-1)
        Ручное задание минимального интервала между соседними узлами.


    enable_pruning : bool, optional (default=True)
        Делать ли обратный проход?


    verbose : bool, optional(default=False)
        Выводить ли дополнительную информацию?



    Атрибуты
    ----------
    `coef_` : array, shape = [pruned basis length, number of outputs]
        Веса несокращённых базисных функций.


    `mse_` : float
        MSE итоговой модели.


    `rsq_` : float
        RSQ итоговой модели.


    `gcv_` : float
        GCV итоговой модели.


    `feature_importances_`: array of shape [m] or dict, где m - кол-во признаков
        Важность всех объектов по одному из критериев.


    `forward_pass_record_` : ???
        Информация по прямому проходу.
        ### TODO Тип выхода определяем сами, хотя можно глянуть как делали авторы библиотеки.


    `pruning_pass_record_` : ???
        Информация по обратному проходу.
        ### TODO Тип выхода определяем сами, хотя можно глянуть как делали авторы библиотеки.



    Методы
    ----------
    ### TODO
    ...
    """

    def __init__(
            self, max_terms=None, max_degree=None, allow_missing=False,
            penalty=None, endspan_alpha=None, endspan=None,
            minspan_alpha=None, minspan=None,
            thresh=None, zero_tol=None, min_search_points=None,
            check_every=None, allow_linear=None, use_fast=None, fast_K=None,
            fast_h=None, smooth=None, enable_pruning=True,
            feature_importance_type=None, verbose=0):

        ### TODO число б.ф. должно быть меньше, чем размерность объектов d
        self.max_terms = max_terms
        self.max_degree = max_degree
        self.penalty = penalty
        self.endspan_alpha = endspan_alpha
        self.endspan = endspan
        self.minspan_alpha = minspan_alpha
        self.minspan = minspan
        self.thresh = thresh
        self.min_search_points = min_search_points
        self.allow_linear = allow_linear
        self.smooth = smooth
        self.enable_pruning = enable_pruning
        self.feature_importance_type = feature_importance_type
        self.verbose = verbose

        if self.penalty == None:
            self.penalty = 3

        ### Пока не реализуем
        self.allow_missing = allow_missing
        self.use_fast = use_fast
        self.fast_K = fast_K
        self.fast_h = fast_h
        self.zero_tol = zero_tol
        self.check_every = check_every

        # Нами введённые параметры
        ### TODO добавить поддержку не только GCV.
        ### В частности - поддержку произвольной ф-ции. Для оптимизации использовать численные методы.
        self.method = 'nelder-mead'  # метод оптимизации
        self.lof = self.gcv  # LOF ф-ция
        self.best_lof = float('inf')  # зн-ие lof на обученных коэфф-ах
        # term_list = [B_1, ..., B_M] - список б.ф. (term)
        # B = [mult_,1 , ... , mult_K] - список множителей (mult) б.ф. B
        self.term_list = self.TermListClass([[self.ConstantFunc(1.)], ])
        self.coeffs = None  # коэфф-ты при б.ф.
        self.B = None  # матрица объекты-б.ф. (чтобы не пересчитывать лишний раз)


    ### Множества нужны для verbose, trace и т.д.
    ### Из py-earth пока не добавлены сюда: allow_missing, zero_tol, use_fast, fast_K, fast_h, check_every.
    ### После реализации соответсвующих возможностей, добавлять сюда соответствующие параметры.
    forward_pass_arg_names = set([
        'max_terms', 'max_degree', 'allow_missing', 'penalty',
        'endspan_alpha', 'endspan',
        'minspan_alpha', 'minspan',
        'thresh', 'min_search_points', 'allow_linear',
        'feature_importance_type',
        'verbose'
    ])
    pruning_pass_arg_names = set([
        'penalty',
        'feature_importance_type',
        'verbose'
    ])


    # =====================================Вспомогательные классы и ф-ции============================
    ### TODO Уменьшить кол-во пересчётов матрицы B.

    ### Мб можно хранить в более удобном виде?
    class TermListClass(list):
        """
        Класс, реализующий представление мн-ва б.ф. в виде списка б.ф.
        б.ф. также представляется обёрткой над списком составляющих его множителей.
        TODO дописать
        """

        def __init__(self, term_list):
            super().__init__(term_list)

        def __repr__(self):
            s = ''
            for term in self:
                for mult in term:
                    s += '[' + str(mult) + '], '
                s += '\n'
            return s
        

    class TermClass(list):
        """
        Класс, реализующий представление б.ф. в виде списка составляющих его множителей.
        TODO дописать
        """

        def __init__(self, term):
            super().__init__(term)

    
    def g_calculation(self, B, coeffs):
        '''
        Ф-ция, реализующая ф-цию g(x):
            g(x) = a_1 * B_1(x) + ... + a_M * B_M(x)

        Параметры
        ----------
        B: матрица объекты-б.ф.
        coeffs: вектор коэффициентов


        Выход
        ----------
        вектор значений ф-ции g(x) на эл-тах обуч.
        '''
        g = B @ coeffs
        return g


    def term_calculation(self, X, term):
        '''
        Ф-ция, реализующая вычисление б.ф. B_m(x):
            B_m(x) = mult_{m,1}(x) * ... * mult_{m,K_m}(x)

        Параметры
        ----------
        X: матрица объектов
        term: б.ф. B_m(x)

        
        Выход
        ----------
        значения б.ф. B_m
        '''
        term_values = 1
        for mult in term:
            term_values *= mult.calculate(X)
        return term_values


    def b_calculation(self, X, term_list):
        '''
        Ф-ция, вычисляющая матрицу B:
            B[i,m] = B_m(x[i])

        Параметры
        ----------
        X: матрица объектов
        term_list: [B_1, ..., B_M] - список б.ф.


        Выход
        ----------
        матрица B
        '''
        data_count, term_count = X.shape[0], len(term_list)
        B = np.empty((data_count, term_count))

        for ind in range(term_count):
            B[:, ind] = self.term_calculation(X, term_list[ind])
        return B


    def c_calculation(self, B):
        '''
        Ф-ция, вычисляющая поправочный коэффициент C_correct(M).
            C(M) = trace(B @ (B^T @ B)^(-1) @ B^T) + 1
            C_correct(M) = C(M) + d*M
            B[i,m] = B_m(x_i)
        ### TODO: добавить простую эвристику вычисления C из книжки
            
        Параметры
        ----------
        B: матрица объекты-б.ф.


        Выход
        ----------
        поправочный коэффициент C_correct(M)
        '''
        term_count = B.shape[1]

        # сдвигаем СЗ матрицы V на половину машинной точности float32
        V = B.T @ B
        V += np.finfo(np.float32).eps / 2

        C = np.trace(B @ LA.inv(V) @ B.T) + 1
        C_correct = C + self.penalty * term_count
        return C_correct
    

    def minspan_endspan_calculation(self, b, data_dim):
        '''
        Ф-ция, вычисляющая L(alpha) и Le(alpha):
            L(alpha) задаёт шаг из порогов между соседними узлами
            Le(alpha) задаёт отступ из порогов для граничных узлов 
        Смысл - сглаживание, скользящее окно.
            
        Параметры
        ----------
        b: вектор - значения б.ф. B_m на объектах из обуч. выборки
        data_dim: размерность объектов
        '''
        term_nonzero_count = np.count_nonzero(b)

        if self.minspan == None:
            self.minspan = -np.log2(-1 / (data_dim * term_nonzero_count) *
                               np.log(1 - self.minspan_alpha)) / 2.5

        if self.endspan == None:
            self.endspan = 3 - np.log2(self.endspan_alpha / data_dim)
        

    def gcv(self, B, y, coeffs):
        '''
        ### TODO: реализовать в виде класса
        Generalized Cross-Validation criterion (GCV):
            GCV(M) = 1/N * sum([y_i - f(x_i)]^2) / [1 - C_correct(M)/N]^2
            C_correct(M) = C(M) + d*M
            C(M) = trace(B @ (B^T @ B)^(-1) @ B^T) + 1
            Смысл C - число лин. нез. б.ф.
            Смысл GCV - скорректированный MSE, учитывающий возрастание дисперсии, вызванное увеличением кол-ва б.ф. 
        
        Параметры
        ----------
        B: матрица объекты-б.ф.
        y: вектор ответов на объектах
        coeffs: вектор коэфф-ов


        Выход
        ----------
        значение GCV
        '''
        data_count = y.size
        y_pred = self.g_calculation(B, coeffs)
        mse = np.mean((y - y_pred) ** 2)
        c = self.c_calculation(B)
        mse_correct = mse / (1 - c / data_count) ** 2
        return mse_correct


    def minimize(self, f, x0, args, method='nelder-mead', options={'disp': True}):
        """
        Ф-ция численной минимизации. Обёртка над scipy.optimize.minimize

        Параметры
        ----------
        ### TODO посмотреть другие параметры, 
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


    def coeffs_calculation(self, V, c):
        """
        Ф-ция, аналитически вычисляющая коэфф-ты МНК методом Холецкого:
        V @ a = c, V = L @ L^T
        Где:
            V - симметричная, положительно определённая матрица
            L - нижн. треуг. матрица

        Параметры
        ----------
        V: матрица B.T @ B
        c: вектор B.T @ y


        Выход
        ----------
        коэфф-ты при б.ф. в модели
        """
        # V @ a = (L @ L^T) @ a = L @ (L^T @ a) = L @ b = c
        # b = L^T @ a

        # проверка на симметричность матрицы V
        if not np.allclose(V, V.T):
            raise LA.LinAlgError('Asymmetric matrix!')

        # сдвигаем СЗ на половину машинной точности float32
        ### хотя тип матрицы float64, но такой добавки не хватает
        ### TODO: всё равно была проблема. мб дело не в этом, а мб и потому что не хватило
        half_eps = np.finfo(np.float32).eps / 2
        ### TODO: добавлять диагональную матрицу
        V += half_eps

        # проверка на положительную определённость
        if not np.all(LA.eigvalsh(V) > 0):
            raise LA.LinAlgError('Matrix is not positive definite!')
        
        L = LA.cholesky(V)

        # Решение системы L @ b = c
        b = LA.solve(L, c)

        # Решение системы L^T @ a = b
        a = LA.solve(L.T, b)

        ### TODO: Мб есть оптимизатор, в котором можно явно выбрать метод Холецкого?
        return a


    class ConstantFunc():
        """
        Класс, реализующий константную ф-цию.
            f(x) = const

        Атрибуты
        ----------
        value: значение
        """

        ### TODO Какие методы, кроме совсем тривиальных, должны быть в этом классе?

        def __init__(self, value=1.):
            self.value = value

        def __repr__(self):
            ### TODO во всех классах переопределить __class__:
            ### return f'{self.__class__} value={self.value}'
            return f'<ConstantFunc> value={self.value}'

        def calculate(self, X=None):
            '''
            Ф-ция, вычисляющая константную ф-цию.

            Параметры
            ----------
            X: матрица объектов (не используется)
            '''
            return self.value

    
    class BaseFunc():
        """
        Класс-родитель для ф-ций, использующихся в качестве множителей в б.ф.

        Атрибуты
        ----------
        s: знак
        v: координата
        t: порог
        """

        ### TODO
        ### 1. Подумать, что общего, кроме (s, v, t), можно найти у этих и мб будущих функций?
        ###    Вспомнить статьи, которые читали. Мб там использовались какие-то ещё ф-ции?
        ###    Если да, то что у них общего с нашими?
        ### 2. Подумать, какие ещё методы и атрибуты могли бы пригодиться?

        def __init__(self, s, v, t):
            self.s = s
            self.v = v
            self.t = t

        def __repr__(self):
            return f'<BaseFunc> s={self.s}, v={self.v}, t={self.t:.3f}'


    class IndicatorFunc(BaseFunc):
        """
        Класс, реализующий индикаторную ф-цию.
            f(x) = s * [x_v - t]

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
            X: матрица объектов
            '''
            indicator = self.s * (X[:, self.v] - self.t > 0.)
            return indicator


    class LinearFunc(BaseFunc):
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
            X: матрица объектов
            '''
            linear = self.s * (X[:, self.v] - self.t)
            return linear


    class ReluFunc(BaseFunc):
        """
        Класс, реализующий положительную срезку (она же ReLU).
            f(x) = [s * (x[v] - t)]_+

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
            X: матрица объектов
            '''
            linear = self.s * (X[:, self.v] - self.t)
            relu = np.maximum(linear, 0)
            return relu
        

    class PowReluFunc(BaseFunc):
        """
        Класс, реализующий степенную положительную срезку.
            f(x) = [s * (x[v] - t)]^q_+

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
            X: матрица объектов
            '''
            linear = self.s * (X[:, self.v] - self.t)
            relu = np.maximum(linear, 0)
            pow_relu = relu ** self.q
            return pow_relu


    class CubicFunc(BaseFunc):
        """
        Класс, реализующий кубический усечённый сплайн с непрерывной 1ой производной.
            ### TODO : Привести вид такой ф-ции или хотя бы её концепцию.
        
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
            self.t_plus = t_plus

        def __repr__(self):
            return (f'<CubicFunc> s={self.s}, v={self.v}, t={self.t:.3f},'
                    f't_minus={self.t_minus:.3f}, t_plus={self.t_plus:.3f}')

        def calculate(self, X):
            '''
            Ф-ция, вычисляющая кубический усечённый сплайн.

            Параметры
            ----------
            X: матрица объектов
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


    def knot_optimization(self, X, y, terms, new_basis, sorted_features, splines_type='default', **kwargs):
        """
        Ф-ция находит лучший узел при фиксированной б.ф. B_m и координате v.

        Параметры
        ----------
        X : обучающая выборка
        y : отлклики
        terms : список уже созданных б.ф.
        new_basis : (term_num, term, v) - фиксированная б.ф. и координата v
        sorted_features : отсортированный по возрастанию список перебираемых координат
        splines_type : ?
        ### Не знаю зачем нам нужно передавать sorted_features, если мы и так передаём X?
        ### Мб чтобы не сортировать лишний раз?


        Выход
        ----------
        (best_thres, best_lof) : лучший порог и значение LOF на нём
        """
        data_count, data_dim = X.shape
        term_num, term, v = new_basis
        B = self.b_calculation(X, terms)
        # B_extnd - расширенная матрица B, она не явл-ся матрицей объекты-б.ф.,
        #  т.к. в последних двух столбцах используются не б.ф.
        B_extnd = np.hstack([B, np.empty((data_count, 2))])


        # Прореживание мн-ва перебираемых порогов.
        self.minspan_endspan_calculation(B[:, term_num], data_dim)
        thin_sorted_features = sorted_features[self.endspan:-self.endspan:self.minspan]
        if thin_sorted_features.size == 0:
            return (None, float('inf'))
        

        # заполняем последние 2 столбца матрицы B_extnd 2-мя последними слагаемыми из g' (они не б.ф.)
        thres = thin_sorted_features[-1]
        B_extnd[:, -2] = B_extnd[:, term_num] * X[:, v]                         # B_m(x) * x[v]
        B_extnd[:, -1] = B_extnd[:, term_num] * np.maximum(X[:, v] - thres, 0)  # B_m(x) * (x[v] - t)_+
        b_extnd_mean = np.mean(B_extnd, axis=0)
        y_mean = np.mean(y)

        # B @ a = y ---> (B^T @ B) @ a = (B^T @ y) = V @ a = c
        ### TODO: Попробовать без нормализации и с полной нормализацией.
        V_extnd = (B_extnd - b_extnd_mean).T @ B_extnd
        c_extnd = (y - y_mean).T @ B_extnd

        s_greater_ind = np.nonzero(X[:, v] >= thres)
        s_prev = np.sum(B_extnd[s_greater_ind, term_num] * (X[s_greater_ind, v] - thres))  # s(u)

        coeffs_extnd = self.coeffs_calculation(V_extnd, c_extnd)
        best_lof = self.lof(B_extnd, y, coeffs_extnd)
        best_thres = thres


        # цикл по порогам                             t <= u
        thres_prev = thres                          # thres_prev -> u
        for thres in thin_sorted_features[-2::-1]:  # thres -> t
            if self.term_calculation(thres, term) == 0: ###?
                continue
            between_ind = np.nonzero(thres < X[:, v] < thres_prev)
            greater_ind = np.nonzero(X[:, v] >= thres_prev)
            s_greater_ind = np.nonzero(X[:, v] >= thres)
            s = np.sum(B_extnd[s_greater_ind, term_num] * (X[s_greater_ind, v] - thres))  # s(t)

            # c[M+1]
            c_extnd[-1] += np.sum((y[between_ind] - y_mean) * B_extnd[between_ind, term_num] *
                                  (X[between_ind, v] - thres)) + \
                           (thres_prev - thres) * \
                           np.sum((y[greater_ind] - y_mean) *
                                  B_extnd[greater_ind, term_num])

            # V[i,M+1], i = 1..M => {симметричность матрицы V} => V[M+1,i], i = 1..M
            V_extnd[:-1, -1] += np.sum((B_extnd[between_ind, :-1] - b_extnd_mean[:-1]) * 
                                       B_extnd[between_ind, term_num][:, np.newaxis] *
                                       (X[between_ind, v] - thres)[:, np.newaxis], axis=0) + \
                                (thres_prev - thres) * \
                                np.sum((B_extnd[greater_ind, :-1] - b_extnd_mean[:-1]) *
                                       B_extnd[greater_ind, term_num][:, np.newaxis], axis=0)
            V_extnd[-1, :-1] = V_extnd[:-1, -1]

            # V[M+1,M+1]
            V_extnd[-1, -1] += np.sum((B_extnd[between_ind, term_num] * (X[between_ind, v] - thres)) ** 2) + \
                               (thres_prev - thres) * \
                               np.sum(B_extnd[greater_ind, term_num] ** 2 *
                                      (2 * X[greater_ind, v] - thres - thres_prev)) + \
                               (s_prev ** 2 - s ** 2) / data_count

            # находим коэфф-ты, отвечающие ф-ции g' и отличающиеся от оптимальных для соотв-го набора б.ф.
            # при этом зн-я lof* моделей g и g' (со своими наборами коэфф-тов) совпадают =>
            # => совпадают оптимальные пороги t*
            coeffs_extnd = self.coeffs_calculation(V_extnd, c_extnd)
            lof = self.lof(B_extnd, y, coeffs_extnd)

            if lof < best_lof:
                best_lof = lof
                best_thres = thres

            s_prev = s
            thres_prev = thres

        return (best_thres, best_lof)


    ### Дополнительные ф-ции, которые использовались в py-earth.
    ### def __eq__(self, other):
    ### def __ne__(self, other):
    ### def _pull_forward_args(self, **kwargs): "Pull named arguments relevant to the forward pass"
    ### def _pull_pruning_args(self, **kwargs): "Pull named arguments relevant to the pruning pass"
    ### def _scrape_labels(self, X): "Try to get labels from input data (for example, if X is a
    ### pandas DataFrame).  Return None if no labels can be extracted"
    ### def _scrub_x(self, X, missing, **kwargs): "Sanitize input predictors and extract column names if appropriate"
    ### def _scrub(self, X, y, sample_weight, output_weight, missing, **kwargs): "Sanitize input data"


    ### ==================================================Реализация основной функциональности==========================================================


    ### Если какие-то параметры в последющих функциях не потребуется - ну значит не потребуются.
    ### Но для наглядности пусть будут все.
    def fit(
            self, X, y=None,
            sample_weight=None,
            output_weight=None,
            missing=None,
            xlabels=None,
            linvars=[]):
        """
        Обучение модели.


        Параметры
        ----------
        X : array-like, shape = [m, n], где m - кол-во объектов, n - кол-во признаков
            Обучающие данные.


        y : array-like, optional (default=None), shape = [m, p], где m - кол-во объектов, p - кол-во разных выходов
            Ответы на обучающих данных.


        sample_weight : array-like, optional (default=None), shape = [m], где m - кол-во объектов
            Пообъектное взвешивание. Веса >= 0. Полезно при несбалансированных дисперсиях распределений над объектами.


        output_weight : array-like, optional (default=None), shape = [p], где p - кол-во выходов
            Взвешивание всех ответов модели для каждого из выходовов после обучения.


        ### Пока игнорируем
        missing : array-like, shape = [m, n], где m - кол-во объектов, n - кол-во признаков.
            ...


        linvars : iterable of strings or ints, optional (empty by default)
            Перечисление тех признаков (по номерам или именам), которые могут входить в базисные функции только линейно.


        xlabels : iterable of strings , optional (empty by default)
           Явное задание имён признаков (столбцов). Кол-во имён должно быть равно кол-ву признаков.
        """
        ### TODO Дописать + реализовать вывод доп. инф-ции.
        self.forward_pass(X, y,
                          sample_weight,
                          output_weight,
                          missing,
                          xlabels,
                          linvars)

        if self.enable_pruning:
            self.pruning_pass(X, y,
                              sample_weight,
                              output_weight,
                              missing)
            
        return self
        
        
    def forward_pass(
            self, X, y=None,
            sample_weight=None,
            output_weight=None,
            missing=None,
            xlabels=None,
            linvars=[],
            skip_scrub=False):
        """
        Проход вперёд.


        Параметры
        ----------
        X : array-like, shape = [m, n], где m - кол-во объектов, n - кол-во признаков
            Обучающие данные.


        y : array-like, optional (default=None), shape = [m, p], где m - кол-во объектов, p - кол-во разных выходов
            Ответы на обучающих данных.


        sample_weight : array-like, optional (default=None), shape = [m], где m - кол-во объектов
            Пообъектное взвешивание. Веса >= 0. Полезно при несбалансированных дисперсиях распределений над объектами.

        ### как это понимать?
        output_weight : array-like, optional (default=None), shape = [p], где p - кол-во выходов
            Взвешивание всех ответов модели для каждого из выходовов после обучения.


        ### Пока игнорируем
        missing : array-like, shape = [m, n], где m - кол-во объектов, n - кол-во признаков.
            ...


        linvars : iterable of strings or ints, optional (empty by default)
            Перечисление тех признаков (по номерам или именам), которые могут входить в базисные функции только линейно.


        xlabels : iterable of strings , optional (empty by default)
           Явное задание имён признаков (столбцов). Кол-во имён должно быть равно кол-ву признаков.

        ### Что такое skip_scrub?
        """

        # Обозначения из оригинальной статьи:
        # g(x) - модель:
        #   g(x) = a_1 * B_1(x) + ... + a_M * B_M(x)
        #     M - кол-во б.ф. ---> term_count
        #     B - б.ф. ---> term
        #     B(x) = mult_1 * ... * mult_K
        #     B_1 = const_func
        #       Виды множителей в б.ф. ---> mult:
        #       1) [s_1 * (x[v_1] - t_1)]_+     - положительная срезка
        #       2) [(s_1 * (x[v_1] - t_1))^q]_+ - степенная положительная срезка
        #       3) [s_1 * (x[v_1] - t_1)]       - индикаторная ф-ция
        #       4) s_1 * (x[v_1] - t_1)         - линейная ф-ция
        #       5) const_func                   - константная ф-ция
        #       6) cubic_func                   - кубический сплайн с 1ой непр-ой произв-ой
        #         [...]   - скобка Айверсона
        #         [...]_+ - положительная срезка
        #         K       - кол-во множителей в б.ф.
        #         s_j     - знак j-ого множителя
        #         v_j     - координата x j-ого множителя
        #         t_j     - порог j-ого множителя
        #   a_i - коэф-т при i-ой б.ф.
        #   N   - кол-во объектов ---> data_count
        #   x = (x_1 , ... , x_d)
        #     d - размерность ---> data_dim

        data_count, data_dim = X.shape
        term_count = 2  # M <- 2

        final_coeffs = None
        final_lof = float('inf')

        # создаём б.ф. пока не достигнем макс. кол-ва
        while term_count <= self.max_terms:
            best_lof = float('inf')  # lof* <- +inf

            # перебираем уже созданные б.ф.
            for term_num, term in enumerate(self.term_list):
                # формируем мн-во уже использованных (невалидных) координат
                ### TODO: можно хранить для каждой б.ф. мн-во неиспользованных координат, будет ли ускорение?
                not_valid_coords = []
                # если это не константная б.ф. B_1
                ### TODO сделать соответствующую ф-цию добавления в классе б.ф.
                for mult in term:
                    if type(mult) != self.ConstantFunc:
                        not_valid_coords.append(mult.v)
                # формируем мн-во ещё не занятых (валидных) координат
                valid_coords = [coord for coord in range(0, data_dim) if coord not in not_valid_coords]

                # перебираем все ещё не занятые координаты
                for v in valid_coords:
                    ### TODO:
                    ### t_plus и t_minus предлагается выбрать как среднее между 
                    ###  t и соседними узлами справа и слева

                    # перебираем пороги t (обучающие данные)
                    for ind in range(data_count):
                        # учитываем только нетривиальные пороги
                        x = X[ind][np.newaxis, :]
                        if self.term_calculation(x, term) == 0:
                            continue
                        t = x[0, v]

                        # создаём новые множители
                        mult_with_plus  = self.ReluFunc(-1, v, t)
                        mult_with_minus = self.ReluFunc(+1, v, t)

                        # создаём новые б.ф.
                        ### мб нужно copy.deepcopy
                        term_with_plus  = list(term)
                        term_with_minus = list(term)
                        term_with_plus.append(mult_with_plus)    # B'_M  = B_m * mult_+
                        term_with_minus.append(mult_with_minus)  # B'_{M+1} = B_m * mult_-

                        # добавляем в список б.ф. новые б.ф.
                        term_list = list(self.term_list)
                        term_list.append(term_with_plus)
                        term_list.append(term_with_minus)

                        # находим оптимальные коэфф-ты МНК методом Холецкого, считаем lof
                        B = self.b_calculation(X, term_list)
                        V = B.T @ B
                        c = B.T @ y
                        coeffs = self.coeffs_calculation(V, c)
                        lof = self.lof(B, y, coeffs)
                        if lof < best_lof:
                            best_lof  = lof
                            best_term = term
                            best_v = v
                            best_t = t
                            final_coeffs = coeffs
                            final_lof = lof
        

            # создаём лучшие множители
            mult_with_plus  = self.ReluFunc(-1, best_v, best_t)
            mult_with_minus = self.ReluFunc(+1, best_v, best_t)

            # создаём лучшие б.ф.
            term_with_plus  = list(best_term)
            term_with_minus = list(best_term)
            term_with_plus.append(mult_with_plus)    # B_M
            term_with_minus.append(mult_with_minus)  # B_{M+1}

            # добавляем в список б.ф. лучшие б.ф.
            self.term_list.append(term_with_plus)
            self.term_list.append(term_with_minus)
            term_count += 2  # M <- M + 2

        # зн-е lof и набор коэфф-ов после прохода вперёд
        self.coeffs = final_coeffs
        self.lof = final_lof

        return self


    def pruning_pass(
            self, X, y=None,
            sample_weight=None,
            output_weight=None,
            missing=None,
            skip_scrub=False):
        """
        Проход назад.


        Параметры
        ----------
        X : array-like, shape = [m, n], где m - кол-во объектов, n - кол-во признаков
            Обучающие данные.


        y : array-like, optional (default=None), shape = [m, p], где m - кол-во объектов, p - кол-во разных выходов
            Ответы на обучающих данных.


        sample_weight : array-like, optional (default=None), shape = [m], где m - кол-во объектов
            Пообъектное взвешивание. Веса >= 0. Полезно при несбалансированных дисперсиях распределений над объектами.

        ### как это понимать?
        output_weight : array-like, optional (default=None), shape = [p], где p - кол-во выходов
            Взвешивание всех ответов модели для каждого из выходовов после обучения.


        ### Пока игнорируем
        missing : array-like, shape = [m, n], где m - кол-во объектов, n - кол-во признаков.
            ...

        ### Что такое skip_scrub?
        """
        term_count = len(self.term_list)

        # названия переменных взяты из статьи
        best_K = list(self.term_list) # лучший локальный набор определённого кол-ва б.ф.
        for M in range(term_count, 1, -1): # M, M-1, ..., 2
            b = float('inf')
            L = list(best_K)

            # для очередного прореженного списка поочерёдно удаляем входящие в него б.ф.
            #   для определения кандидата на удаление
            for m in range(1, M):
                K = list(L)
                K.pop(m)

                # находим оптимальные коэфф-ты, считаем lof без очередной б.ф.
                B = self.b_calculation(X, K)
                coeffs = self.coeffs_calculation(B, y)
                lof = self.lof(B, y, coeffs)

                if lof < b:
                    # локальное улучшение
                    b = lof
                    best_K = K
                if lof <= self.best_lof:
                    # глобальное улучшение
                    self.best_lof = lof
                    self.term_list = K
                    self.coeffs = coeffs
        return self
    

    def linear_fit(
            self, X, y=None,
            sample_weight=None,
            output_weight=None,
            missing=None,
            skip_scrub=False):
        """
        Определение коэффициентов для линеной модели методом наименьших квадратов.

        Параметры
        ----------
        X : array-like, shape = [m, n], где m - кол-во объектов, n - кол-во признаков
            Обучающие данные.


        y : array-like, optional (default=None), shape = [m, p], где m - кол-во объектов, p - кол-во выходов
            Ответы на обучающих данных.


        sample_weight : array-like, optional (default=None), shape = [m], где m - кол-во объектов
            Пообъектное взвешивание. Веса >= 0. Полезно при несбалансированных дисперсиях распределений над объектами.


        output_weight : array-like, optional (default=None), shape = [p], где p - кол-во выходов
            Взвешивание всех ответов модели для каждого из выходовов после обучения.
        """
        pass


    def predict(self, X, missing=None, skip_scrub=False):
        """
        Предсказание модели на входных данных.

        Параметры
        ----------
        X : array-like, shape = [m, n], где m - кол-во объектов, n - кол-во признаков
            Входные данные, по которым требуется сделать прогноз.

        
        missing : array-like, shape = [m, n], где m - кол-во объектов, n - кол-во признаков
            ...


        Выход
        ----------
        y : array of shape = [m] или [m, p], где m - кол-во объектов, p - кол-во выходов
            ### TODO Множественная регрессия. Правильно понял?
            Прогнозы.
        """
        # подсчёт ф-ции g(x) на оптимальном наборе б.ф. и их коэффициентов
        B = self.b_calculation(X, self.term_list)
        res = self.g_calculation(B, self.coeffs)
        return res


    def predict_deriv(self, X, variables=None, missing=None):
        """
        Предсказание первых производных на основе входных данных.

        Параметры
        ----------
        X : array-like, shape = [m, n], где m - кол-во объектов, n - кол-во признаков
            Входные данные, по которым требуется сделать прогноз.


        missing : array-like, shape = [m, n], где m - кол-во объектов, n - кол-во признаков
            (пока забиваем)
            ...


        variables : list
            Перемнные, по которым будут вычисляться производные. None => по всем. Если такой


        Выход
        ----------
        X_deriv : array of shape = [m, n, p], где m - кол-во объектов, p - кол-во выходов,
        n - кол-во признаков, если variables не определён, иначе n = len(variables)
            Матрица первых производных всех выходов по каждой variables.
            
        """
        pass


    def score(
            self, X, y=None,
            sample_weight=None,
            output_weight=None,
            missing=None,
            skip_scrub=False):
        """
        Вычисляет обобщённый коэффициент детерминации (R-squared).
        (Чем больше, тем лучше)

        Параметры
        ----------
        ### Заполняется аналогично fit.


        Выход
        ----------
        score : float (максимум 1, мб отрицательным).
            Обобщённый коэффициент детерминации.
        """
        pass


    def score_samples(self, X, y=None, missing=None):
        """
        ### Не очень понял что это надо разобраться,.

        Параметры
        ----------
        ### Заполняется аналогично fit.


        Выход
        ----------
        scores : array of shape=[m, p] of floats (максимум 1, мб отрицательным).
            ### 
        """
        pass


    def transform(self, X, missing=None):
        """
        Переход в пространство базисных функций.

        Параметры
        ----------
        X : array-like, shape = [m, n], где m - кол-во объектов, n - кол-во признаков
            Входные непреобразованные данные.


        missing : array-like, shape = [m, n], где m - кол-во объектов, n - кол-во признаков
            ...


        Выход
        ----------
        B: array of shape [m, nb_terms], где m - кол-во объектов, nb_terms - кол-во
        получившихся базисных функций.
            Матрица, содержащая значения базисных функций, вычисленных на всех объектах.
        """
        B = self.b_calculation(X, self.term_list)
        return B


    ### ========================================================Вывод информации=======================================================================


    class InfoClass():
        """Класс, реализующий удобный отладочный интерфейс"""
        ### TODO
        ### Подумать над структурой.
        pass


    def forward_trace(self):
        '''
        Вывод информации о проходе вперёд.
        '''
        pass


    def pruning_trace(self):
        '''
        Вывод информации о проходе назад.
        '''
        pass


    def trace(self):
        '''
        Вывод информации о проходе вперёд и назад.
        '''
        #self.forward_trace()
        #self.pruning_trace()
        pass


    def summary(self):
        """
        Описание модели в виде строки.
        """
        return self.term_list


    def summary_feature_importances(self, sort_by=None):
        """
        Важность признаков в виде строки.


        Параметры
        ----------
        sory_by : string, optional
            Сортировка, если поддерживается, по feature_importance_type ('rrs', 'gcv', 'nb_subsets').
        """
        pass


    def get_penalty(self):
        """
        Возвращает параметр сглаживания d
            C_correct(M) = C(M) + d*M.
        """
        return self.penalty
    

    def get_minspan_endspan(self):
        """
        Возвращает L(alpha) и Le(alpha):
            L(alpha) задаёт шаг из порогов между соседними узлами
            Le(alpha) задаёт отступ из порогов для граничных узлов
        Смысл - сглаживание, скользящее окно.
        """
        return (self.minspan, self.endspan)


### ==========================================Для всякого====================================================== 
### TODO: Проверка на корректность атрибутов с исключениями.