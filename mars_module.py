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

        
    calc_error : float, optional
        Допустимая погрешность вычисления. Это значение будет использоваться для
        корректировки ошибок округления там, где это может быть важным
        (например, положительная определённость)

    
    zero_tol : float, optional (default=1e-12)
        Абсолютная точность. Используется для определения равно ли число с плавающей точкой нулю.
        Нужен только для сохранения оригинальных названий из py-earth.
        Рекомендуется использовать atol - обобщение zero_tol.

    
    atol : float, optional (default=1e-12)
        Абсолютная точность. Нужен, в частности, для встречающихся в коде сравнений, а также для
        корректировки положительно определённых матриц, которые иногда в силу ошибок округления могут таковыми не оказаться.
        При появлении подобной ошибки уменьшить значения абсоютной и относительной точности.


    rtol : float, optional (default=1e-05)
        Относительная точность. Ситуации применения аналогичны atol.
        

        
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
            thresh=None, zero_tol=None, atol=None, rtol=None, min_search_points=None,
            check_every=None, allow_linear=None, use_fast=None, fast_K=None,
            fast_h=None, smooth=None, enable_pruning=True,
            feature_importance_type=None, verbose=0):

        ### TODO число б.ф. должно быть меньше, чем размерность объектов d, проверка
        self.max_terms = max_terms          # +
        self.max_degree = max_degree
        self.penalty = penalty              # +
        self.endspan_alpha = endspan_alpha  # +
        self.endspan = endspan              # +
        self.minspan_alpha = minspan_alpha  # +
        self.minspan = minspan              # +
        self.thresh = thresh
        self.min_search_points = min_search_points
        self.allow_linear = allow_linear
        self.smooth = smooth
        self.enable_pruning = enable_pruning    # +
        self.feature_importance_type = feature_importance_type
        self.verbose = verbose
        self.zero_tol = zero_tol # +
        self.atol = atol         # +
        self.rtol = rtol         # +


        if self.minspan == None:
            self.minspan = -1

        if self.endspan == None:
            self.endspan = -1

        ### Пока не реализуем
        self.allow_missing = allow_missing
        self.use_fast = use_fast
        self.fast_K = fast_K
        self.fast_h = fast_h
        self.check_every = check_every

        ### Из py-earth вычисляемые аттрибуты
        self.coef_ = None
        self.basis_ = None
        self.mse_  = None
        self.rsq_  = None
        self.gcv_  = None
        self.grsq_ = None
        self.forward_pass_record_ = None
        self.pruning_pass_record_ = None
        self.xlabels_ = None
        self.allow_missing_ = None
        self.feature_importances_ = None


        # Введённые параметры
        ### TODO добавить поддержку не только GCV.
        ### В частности - поддержку произвольной ф-ции. Для оптимизации использовать численные методы. (хотя будет очень не эффективно)
        # term_list = [B_1, ..., B_M] - список б.ф. (term)
        # B = [mult_1 , ... , mult_K] - список множителей (mult) б.ф. B
        self.term_list = None
        self.coeffs = None
        self.method = 'nelder-mead'    # метод оптимизации
        self.lof_func = self.mse_func  # LOF ф-ция
        self.lof_value = float('inf')  # зн-ие LOF на выученных коэфф-ах
        self.B = None           # мат. объекты-б.ф. для обуч. выборки
        self.data_count = None  # кол-во данных в обуч. выборке
        self.data_dim   = None  # размерность данных обуч. выборки


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
    class TermClass(list):
        """
        Класс, реализующий представление б.ф.
        Обёртка над списком множителей, составляющих б.ф.
        TODO дописать
        """

        def __init__(self, term, valid_coords):
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
            Кроме простого добавления в списком, происходит удаление
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

        def __init__(self, earth_instance, term_list):
            super().__init__(term_list)
            self.earth_instance = earth_instance

        def __repr__(self):
            s = ''
            for term_num, term in enumerate(self):
                s += str(term) + f'coeff={self.earth_instance.coeffs[term_num]}\n'
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
        Ф-ция, реализующая вычисление матрицы B:
            B[i,m] = B_m(x_i)

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


    def c_calculation(self, V, B, b_mean=None):
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
        V: матрица, скорректированная симм. полож. опред. V = B.T @ B
        b_mean: вектор, среднее по объектам мат. B (по умолчанию = None)
            Выполняется ли нормализация мат. B (по умолчанию соотв-т нет)?
            Если не None, то считается, что матрица V уже учитывает нормирование мат. B.


        Выход
        ----------
        float, поправочный коэффициент C_correct(M)
        '''
        if self.penalty == None:
            self.penalty = 3

        # если b_mean != None => в V уже использована нормализованная мат. B (V = B^T @ (B - B_mean)) =>
        # => мат. B, которая используется в формуле для C(M), также должна быть нормализована и только она
        
        term_count = B.shape[1]

        if b_mean == None:
            B_mod = B
        else:
            B_mod = B - b_mean

        C = np.trace(B_mod @ LA.inv(V) @ B.T) + 1
        C_correct = C + self.penalty * term_count
        return C_correct
    

    def minspan_endspan_calculation(self, nonzero_count, data_dim):
        '''
        Ф-ция, вычисляющая L(alpha) и Le(alpha):
            L(alpha)  - задаёт шаг из порогов между соседними узлами
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
        if self.minspan_alpha == None:
            self.minspan_alpha = 0.05

        if self.endspan_alpha == None:
            self.endspan_alpha = 0.05


        minspan = self.minspan
        endspan = self.endspan

        if minspan == -1:
            minspan = int(-np.log2(-1 / (data_dim * nonzero_count) *
                          np.log(1 - self.minspan_alpha)) / 2.5)

        if endspan == -1:
            endspan = int(3 - np.log2(self.endspan_alpha / data_dim))


        return (minspan, endspan)


    def mse_func(self, B, y, coeffs, **kwargs):
        """
        Mean Squared Error (MSE):
            MSE = 1/N * sum([y_i - f(x_i)]^2)
            f(x) = g(x) = B @ a

        Параметры
        ----------
        B: матрица, объекты-б.ф.
        y: вектор, ответы на объектах
        coeffs: вектор, коэфф-ты при б.ф.


        Выход
        ----------
        float, значение MSE
        """
        y_pred = self.g_calculation(B, coeffs)
        mse = np.mean((y - y_pred) ** 2)
        return mse
        

    def gcv_func(self, B, y, coeffs, V, b_mean=None):
        '''
        TODO реализовать в виде класса
        Generalized Cross-Validation criterion (GCV):
            GCV(M) = 1/N * sum([y_i - f(x_i)]^2) / [1 - C_correct(M)/N]^2 <=>
            GCV(M) = MSE / [1 - C_correct(M)/N]^2
            C_correct(M) = C(M) + d*M
            C(M) = trace(B @ (B.T @ B)^(-1) @ B.T) + 1
            f(x) = g(x) = B @ a
            Смысл C: число лин. нез. б.ф.
            Смысл GCV: скорректированный MSE, учитывающий возрастание дисперсии, вызванное увеличением кол-ва б.ф. 
        
        Параметры
        ----------
        B: матрица, объекты-б.ф.
        y: вектор, ответы на объектах
        coeffs: вектор, коэфф-ты при б.ф.
        V: матрица, скорректированная симм. полож. опред. V = B.T @ B
            (обязательный параметр)
        b_mean: вектор, усреднённая по объектам мат. B (по умолчанию = None)
            Выполняется ли нормализация мат. B (по умолчанию соотв. нет)?
            Если не None, то считается, что матрица V уже учитывает нормирование мат. B.


        Выход
        ----------
        float, значение GCV
        '''
        data_count = y.size
        mse = self.mse_func(B, y, coeffs)
        C = self.c_calculation(V, B, b_mean=b_mean)
        gcv = mse / (1 - C / data_count)**2
        return gcv
    

    def rss_func(self, ):
        """
        
        """
        pass


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


    def analytically_pseudo_solves_slae(self, V, c):
        """
        Ф-ция, аналитически вычисляющая псевдорешение СЛАУ
          методом наименьших квадратов с применением метода Холецкого.
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
            lof = self.lof_func(B, y, coeffs, V=V)

        return (coeffs, lof)
    

    def symmetric_positive_matrix_correct(self, A):
        """
        Ф-ция, корректирующая потенциально симметричную положительно определённую матрицу,
          добавляя к ней диагональную матрицу, определяемую атрибутами точности: atol, rtol и zero_tol.
        Далее происходит проверка достижения желаемого рез-та.
        В противном случае выбрасывается исключение.
        
        Параметры
        ----------
        A: матрица, потенциально симм. полож. опред.


        Выход
        ----------
        матрица, симм. полож. опред.
        """
        if (self.atol == None) and (self.zero_tol != None):
            self.atol = self.zero_tol

        if self.atol == None:
            self.atol = 1e-04

        if self.rtol == None:
            self.rtol = 1e-05


        A += np.diag(self.atol + np.diag(A) * self.rtol)

        # проверка на симметричность
        if not np.allclose(A, A.T, rtol=self.rtol, atol=self.atol):
            raise LA.LinAlgError('Asymmetric matrix')

        # проверка на положительную определённость
        if not np.all(LA.eigvalsh(A) > 0):
            raise LA.LinAlgError('Matrix is not positive definite')

        return A
    

    def modified_cholesky(self, V, L):
        """
        Ф-ция, реализующая разложение Холецкого с частичным пересчётом элементов выходной матрицы L.

        Параметры
        ----------
        V: матрица, симм. полож. опред.
        L: матрица, треуг. с прошлой итерации


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
            indicator = self.s * ((X[:, self.v] - self.t) > 0.)
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
        X : матрица, обучающая выборка
        y : вектор, ответы на объектах
        B : матрица, объекты-б.ф.
        fixed_values : (m - int, v - int) - фикс. параметры: номер фикс. б.ф. и номер фикс. коор-ты v


        Выход
        ----------
        (best_t, best_lof) : (float, float), лучший порог t и лучшее значение lof при фикс. параметрах
        """
        pass


    def knot_optimization(self, X, y, B, fixed_values, sorted_features, splines_type='default', **kwargs):
        """
        Ф-ция находит лучший узел t при фиксированной б.ф. B_m и координате v.

        Параметры
        ----------
        X : матрица, обучающая выборка
        y : вектор, ответы на объектах
        B : матрица, объекты-б.ф.
        fixed_values : (m - int, v - int) - фикс. параметры: номер фикс. б.ф. и номер фикс. коор-ты v
        sorted_features : вектор, зн-я объектов по фикс. коорд. отсортированные по возрастанию
        splines_type :


        Выход
        ----------
        (best_t, best_lof) : (float, float), лучший порог t и лучшее значение lof при фикс. параметрах
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
        B_extnd[:, -1] = 0          # B_m(x) * [x[v] - t]_+ == 0 (t - самый крайний порог)
        

        # Прореживание мн-ва перебираемых порогов
        N_m = np.count_nonzero(B_m)
        minspan, endspan = self.minspan_endspan_calculation(N_m, d)
        thin_sorted_features = sorted_features#[endspan:-endspan:minspan]
        if thin_sorted_features.size == 0:
            return (None, float('inf'))
        t = thin_sorted_features[-1]


        # Нормализация:
        #   B @ a = y <=> {нормализация} <=> (B - B_mean) @ a = y - y_mean --->
        #   ---> {B.T @ (...)} ---> [B.T @ (B - B_mean)] @ a = B.T @ (y - y_mean) <=>
        #   <=> V @ a = c
        y_mean = np.mean(y)
        b_extnd_mean = np.mean(B_extnd, axis=0)

        V_extnd = self.symmetric_positive_matrix_correct(B_extnd.T @ (B_extnd - b_extnd_mean))
        c_extnd = B_extnd.T @ (y - y_mean)


        coeffs_extnd = self.analytically_pseudo_solves_slae(V_extnd, c_extnd)

        best_lof = self.lof_func(B_extnd, y, coeffs_extnd, V=V_extnd, b_mean=b_extnd_mean)
        best_t = t

    
        # Цикл по порогам от больших к меньшим, t <= u
        u = t
        s_u = 0
        for t in thin_sorted_features[-2::-1]:
            between_inds = np.nonzero((t <= x_v) & (x_v < u))[0]
            greater_inds = np.nonzero(x_v >= u)[0]

            # Обновляем последнее слагаемое в g', которое только одно зав-т от порога t
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

            # Находим коэфф-ты.
            # Они отвечают ф-ции g' и отличаются от оптимальных для набора б.ф.
            # При этом зн-я lof* моделей g и g' на своих оптимальных наборах коэфф-тов совпадают =>
            #   => совпадают оптимальные пороги t*
            coeffs_extnd = self.analytically_pseudo_solves_slae(V_extnd, c_extnd)

            # Находим lof*
            lof = self.lof_func(B_extnd, y, coeffs_extnd, V=V_extnd, b_mean=b_extnd_mean)

            if lof < best_lof:
                best_lof = lof
                best_t = t
            
            #print(f'v: {v}, m: {m}, t: {t}, lof: {lof}, best_lof: {best_lof}')

            u = t
            s_u = s_t

        return (best_t, best_lof)


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
        #     x = (x_1 , ... , x_d) - объект
        #       N - кол-во объектов ---> data_count
        #       d - размерность данных ---> data_dim
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

        self.data_count, self.data_dim = X.shape

        # Создание первой константной б.ф.
        const_term = self.TermClass([self.ConstantFunc(1.)], valid_coords=list(range(self.data_dim)))
        self.term_list = self.TermListClass(self, [const_term, ])

        # Создание матрицы объекты-б.ф., её заполнение для константной б.ф.
        B = np.ones((self.data_count, 1))

        best_lof = None
        term_count = 2  # M <- 2

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

                    # находим лучший порог t и lof при фикс. б.ф. term и координате v
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
            term_plus  = self.TermClass(best_term, valid_coords=copy.copy(best_term.valid_coords))  # B_M
            term_minus = self.TermClass(best_term, valid_coords=copy.copy(best_term.valid_coords))  # B_{M+1}
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


        # Нахождение коэфф-ов построенной модели
        self.coeffs, _ = self.coeffs_and_lof_calculation(B, y, self.term_list, for_X=False, need_lof=False)
        self.lof_value = best_lof
        self.B = B

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
                coeffs = self.analytically_pseudo_solves_slae(B, y)
                lof = self.lof_func(B, y, coeffs)

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
        B = self.b_calculation(X, self.term_list)
        y_pred = self.g_calculation(B, self.coeffs)
        return y_pred


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
            пропуски в данных ?


        Выход
        ----------
        B: array of shape [m, nb_terms], где m - кол-во объектов, nb_terms - кол-во
        получившихся базисных функций.
            Матрица объекты-б.ф.
        """
        B = self.b_calculation(X, self.term_list)
        return B


    ### ========================================================Вывод информации=======================================================================


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
        self.forward_trace()
        self.pruning_trace()


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
        Возвращает параметр сглаживания d.
            C_correct(M) = C(M) + d*M.
        """
        return self.penalty
    

    def get_minspan_endspan(self):
        """
        Возвращает L(alpha) и Le(alpha):
            L(alpha)  - задаёт шаг из порогов между соседними узлами
            Le(alpha) - задаёт отступ из порогов для граничных узлов
        Смысл: сглаживание скользящим окном.
        """
        return (self.minspan, self.endspan)


### ==========================================Для всякого====================================================== 
### TODO Проверка на корректность атрибутов с исключениями.