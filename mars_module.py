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
        Ручное задание интервала данных между узлом и краем.


    minspan_alpha : float, optional, probability between 0 and 1 (default=0.05)
        Вероятности сделать больше minspan отклонений.


    minspan : int, optional (default=-1)
        Ручное задание минимального интервала данных между соседними узлами.


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

        ### TODO: число б.ф. должно быть меньше, чем размерность объектов d 
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
        self.lof = self.gcv # LOF ф-ция
        self.best_lof = float('inf') # лучшее полученное значение lof
        #self.method = 'nelder-mead' # метод оптимизации
        ### TODO: проверить и дописать
        # term_list = [B_1, ..., B_M] - список б.ф. (term)
        # B_m = [mult_1, ..., mult_{K_m}] - список множителей (mult) б.ф.
        self.term_list = self.TermListClass([[self.ConstantFunc(1.)], ])
        self.coeffs = np.array([1]) # коэффициенты при б.ф.
        self.B = None # матрица объекты-б.ф. чтобы не пересчитывать лишний раз


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
        Класс, реализующий представление мн-ва б.ф.
        """

        def __init__(self, term_list):
            super().__init__(term_list)
        #    self.term_list = term_list
        #    return self

        def __repr__(self):
            s = ''
            for term in self:
                for mult in term:
                    s += '[' + str(mult) + '], '
                s += '\n'
            return s

    
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
        '''
        g = B @ coeffs
        return g


    def term_calculation(self, X, term):
        '''
        Ф-ция, реализующая вычисление б.ф. B(x) с однотипными множителями:
            B(x) = mult_1(x) * ... * mult_K(x)
            или
            B(x) = 1

        Параметры
        ----------
        X: матрица объектов
        term: б.ф.

        
        Выход
        ----------
        '''
        term_values = 1
        for mult in term:
            term_values *= mult.calculate_func(X)
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
        '''
        B = np.empty((X.shape[0], len(term_list)))
        for ind in range(len(term_list)):
            B[:, ind] = self.term_calculation(X, term_list[ind])
        return B


    def c_correct_calculation(self, B):
        '''
        Ф-ция, вычисляющая поправочный коэффициент C(M).
            C(M) = trace(B @ (B^T @ B)^(-1) @ B^T) + 1
            C_correct(M) = C(M) + d*M
            B[i,m] = B_m(x_i)

        Параметры
        ----------
        B: матрица объекты-б.ф.


        Выход
        ----------
        '''
        V = B.T @ B

        # сдвигаем СЗ на половину машинной точности float32
        half_eps = np.finfo(np.float32).eps / 2
        V += half_eps * np.eye(V.shape[0])

        C = np.trace(B @ LA.inv(V) @ B.T) + 1
        C_correct = C + self.penalty * B.shape[1]
        return C_correct
        

    #def gcv(self, coeffs, *args):
    def gcv(self, B, y, coeffs):
        '''
        Generalized Cross-Validation criterion (GCV):
            GCV(M) = 1/N * sum([y_i - f(x_i)]^2) / [1 - C_correct(M)/N]^2
            C_correct(M) = C(M) + d*M
            C(M) = trace(B @ (B^T @ B)^(-1) @ B^T) + 1
            Смысл C - "число лин. нез. б.ф."
            Смысл GCV - скорректированный MSE, учитывающий возрастание дисперсии при увеличении кол-ва б.ф. 
        
        Параметры
        ----------
        B: матрица объекты-б.ф.
        y: вектор ответов на объектах
        coeffs: вектор коэффициентов


        Выход
        ----------
        '''
        y_pred = self.g_calculation(B, coeffs)
        mse = np.mean((y - y_pred) ** 2)
        correct_c = self.c_correct_calculation(B)
        correct_mse = mse / (1 - correct_c / y.size) ** 2
        return correct_mse


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


    def calculate_coeffs(self, B, y):
        """
        Ф-ция, вычисляющая коэффициенты с помощью псевдорешения СЛАУ (B @ a = y) методом Холецкого:
        (B^T @ B) @ a = B^T @ y => V @ a = c, V = L @ L^T
        Где:
            V - симметричная, положительно определённая матрица
            L - нижн. треуг. матрица

        Параметры
        ----------
        B: матрица объекты-б.ф.
        y: вектор ответов на объектах


        Выход
        ----------
        """
        # V @ a = (L @ L^T) @ a = L @ (L^T @ a) = L @ b = c
        # Где: b = L^T @ a

        V = B.T @ B
        c = B.T @ y

        # проверка на симметричность матрицы V
        if not np.allclose(V, V.T):
            raise LA.LinAlgError('Asymmetric matrix!')

        # сдвигаем СЗ на половину машинной точности float32
        ### хотя тип матрицы float64, но такой добавки не хватает
        half_eps = np.finfo(np.float32).eps / 2
        V += half_eps * np.eye(V.shape[0])

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
            ### TODO переопределить __class__
            #return f'{self.__class__}: value={self.value}'
            return f'<ConstantFunc> value={self.value}'

        def calculate_func(self, X):
            '''
            Ф-ция, вычисляющая константную ф-цию.

            Параметры
            ----------
            X: матрица объектов
            '''
            #const = self.const * np.ones(X.shape[0])
            return self.value

    
    class BaseFunc():
        """
        Класс-родитель для всех ф-ций, использующихся в качестве множителей в б.ф.

        Атрибуты
        ----------
        s: знак
        v: координата
        t: порог
        """

        ### TODO
        ### 1. Подумать, что общего, кроме (s, v, t), можно найти у этих и мб будущих функций.
        ###    Вспомнить статьи, которые читали. Мб там использовались какие-то ещё ф-ции?
        ###    Если да, то что у них общего с нашими?
        ###    
        ### 2. Подумать, какие ещё методы и атрибуты могли бы пригодиться?
        ### Ваши идеи:
        ### ...

        def __init__(self, s, v, t):
            self.s = s
            self.v = v
            self.t = t

        def __repr__(self):
            #return f'{self.__class__}: s={self.s}, v={self.v}, t={self.t:.3f}'
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

        def calculate_func(self, X):
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

        def calculate_func(self, X):
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
        Класс, реализующий положительную срезку (усечённая степенная сплайновая ф-ция).
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

        def calculate_func(self, X):
            '''
            Ф-ция, вычисляющая положительную срезку.

            Параметры
            ----------
            X: матрица объектов
            '''
            linear = self.s * (X[:, self.v] - self.t)
            hinge = np.where(linear > 0, linear, 0)
            return hinge


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
            #return (f'{self.__class__}: s={self.s}, v={self.v}, t={self.t:.3f},'
            #        f't_minus={self.t_minus:.3f}, t_plus={self.t_plus:.3f}')
            return (f'<CubicFunc> s={self.s}, v={self.v}, t={self.t:.3f},'
                    f't_minus={self.t_minus:.3f}, t_plus={self.t_plus:.3f}')

        def calculate_func(self, X):
            '''
            Ф-ция, вычисляющая кубический усечённый сплайн.

            Параметры
            ----------
            X: матрица объектов
            '''

            # точки, в которых X[v] удовлетворяет условиям:
            # X[v] <= t_minus:
            less_than_t_minus = X[:, self.v] <= self.t_minus
            # t_minus < X[v] < t_plus:
            between_t_plus_t_minus = self.t_minus < X[:, self.v] < self.t_plus
            # t_plus <= X[v]:
            greater_than_t_plus = (X[:, self.v] >= self.t_plus)

            # знак положительный
            if self.s > 0:
                p_plus = (2 * self.t_plus + self.t_minus - 3 * self.t) / ((self.t_plus - self.t_minus) ** 2)
                r_plus = (2 * self.t - self.t_plus - self.t_minus) / ((self.t_plus - self.t_minus) ** 3)

                return greater_than_t_plus * (X[:, self.v] - self.t) + \
                       between_t_plus_t_minus * (p_plus * ((X[:, self.v] - self.t_minus) ** 2) +
                                                 r_plus * ((X[:, self.v] - self.t_minus) ** 3))
                
            # знак отрицательный
            p_minus = (3 * self.t - 2 * self.t_minus - self.t_plus) / ((self.t_minus - self.t_plus) ** 2)
            r_minus = (self.t_minus + self.t_plus - 2 * self.t) / ((self.t_minus - self.t_plus) ** 3)

            return less_than_t_minus * (self.t - X[:, self.v]) + \
                   between_t_plus_t_minus * (p_minus * ((X[:, self.v] - self.t_plus) ** 2) +
                                             r_minus * ((X[:, self.v] - self.t_plus) ** 3))
                   

    ### Дополнительные ф-ции, которые использовались в py-earth. Следовать этому вообще не обязательно,
    ### просто для представления структуры отдельных блоков.
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

        ### Пока реализуем самый простой вариант MARS (алгоритм 2 из оригинальной статьи)


        # Обозначения из оригинальной статьи:
        # g(x) - модель: 
        #   g(x) = a_1 * B_1(x) + ... + a_M * B_M(x)
        #     M - итоговое кол-во базисных ф-ций (б.ф.)
        # B_m - m-ая б.ф. ---> term
        #   B_m(x) = [s_1 * (x[v_1] - t_1)]_+ * ... * [s_Km * (x[v_Km] - t_Km)]_+
        #   B_1 = 1 (константная б.ф.)
        #   множители в б.ф. ---> mult
        #     [...]_+ - положительная срезка
        #     Km - общее кол-во множителей в m-ой б.ф.
        #     s_j - знак j-ого множителя
        #     v_j - координата x j-ого множителя
        #     t_j - порог j-ого множителя
        # a_i - коэф-т при i-ой б.ф.
        # N - кол-во объектов ---> data_count
        # x_k = (x_k,1 , ... , x_k,d)
        #   d - размерность ---> data_dim

        data_count, data_dim = X.shape
        term_count = 2 # M <- 2

        # создаём б.ф. пока не достигнем макс. кол-ва
        while term_count <= self.max_terms:
            
            # Может произойти ситуация, когда добавление очередной пары б.ф. не приводит к улучшению.
            # В таком случае мы досрочно завершаем проход вперёд.
            flag_improve_lof = False

            # перебираем уже созданные б.ф.
            for term in self.term_list:
                # формируем мн-во уже использованных (невалидных) координат
                ### TODO: можно хранить для каждой б.ф. мн-во неиспользованных координат
                not_valid_coords = []
                # если это не константная б.ф. B_1
                ### TODO сделать соответствующую ф-цию добавления в классе б.ф.
                for mult in term:
                    if type(mult) != self.ConstantFunc:
                        not_valid_coords.append(mult.v)
                # формируем мн-во ещё не занятых (валидных) координат
                valid_coords = [coord for coord in range(0, data_dim)
                                if coord not in not_valid_coords]

                # перебираем все ещё не занятые координаты
                for v in valid_coords:
                    ### TODO:
                    ### t_plus и t_minus предлагается выбрать как среднее между 
                    ###  t и соседними узлами справа и слева

                    # перебираем обучающие данные (они же пороги)
                    for ind in range(data_count):
                        # учитываем только нетривиальные пороги
                        x = X[ind][np.newaxis, :]
                        if self.term_calculation(x, term) == 0:
                            continue
                        t = x[0, v]

                        # создаём новые множители
                        mult_with_plus = self.ReluFunc(-1, v, t)
                        mult_with_minus = self.ReluFunc(1, v, t)

                        # создаём потенциальные б.ф.
                        ### мб нужно deepcopy
                        term_with_plus = list(term)
                        term_with_minus = list(term)
                        term_with_plus.append(mult_with_plus)   # B_m * [+(x[v] - t)]_+
                        term_with_minus.append(mult_with_minus) # B_m * [-(x[v] - t)]_+

                        # создаём список с новыми б.ф.
                        term_list = list(self.term_list)
                        term_list.append(term_with_plus)
                        term_list.append(term_with_minus)

                        # находим оптимальные коэфф-ты, решая СЛАУ методом Холецкого,
                        #   считаем lof
                        B = self.b_calculation(X, term_list)
                        coeffs = self.calculate_coeffs(B, y)
                        lof = self.lof(B, y, coeffs)
                        if lof < self.best_lof:
                            flag_improve_lof = True
                            self.best_lof = lof
                            best_term = term
                            best_v = v
                            best_t = t
                            self.coeffs = coeffs


            if not flag_improve_lof:
                break

            # создаём оптимальные множители, если было улучшение lof
            mult_with_plus = self.ReluFunc(-1, best_v, best_t)
            mult_with_minus = self.ReluFunc(1, best_v, best_t)

            # создаём новые б.ф.
            ### мб нужно copy.deepcopy
            term_with_plus = list(best_term)
            term_with_minus = list(best_term)
            term_with_plus.append(mult_with_plus)   # B_M
            term_with_minus.append(mult_with_minus) # B_{M+1}
            self.term_list.append(term_with_plus)
            self.term_list.append(term_with_minus)
            term_count += 2 # M <- M + 2


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
                coeffs = self.calculate_coeffs(B, y)
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
            ### (пока забиваем)
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
            Матрица, содержащая значения базисных функций, вычисленных во всех объектах.
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
        pass


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
        Возвращает параметр сглаживания d из C_correct(M) = C(M) + d*M.
        """
        return self.penalty


    def _get_term_list(self):
        for term in self.term_list:
            for mult in term:
                print(mult)
            print('\n')



### ==========================================Для всякого====================================================== 
### TODO: Подумать, мб знаете способы проще масштабировать код?
### Ваши идеи:
### ...