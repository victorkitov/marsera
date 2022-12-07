import numpy as np
from scipy import sparse
from scipy import optimize.linprog as linprog
from sklearn.base import RegressorMixin, BaseEstimator, TransformerMixin


class Earth(BaseEstimator, RegressorMixin, TransformerMixin):
    """
    Класс, реализующий MARS.


    Вся функциональность из py-earth должна быть сюда перенесена.
    По возможности надо сохранить оригинальные названия аргументов, атрибутов, методов и т.д.


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
        ### Это то, как определяется условие остановки в py-earth.
        ### Можно будет предложить ещё какие-то условия, но это пусть тоже остаётся.


    smooth : bool, optional (default=False)
        Использовать ли вместо срезки кубические усечённые сплайны с непрерывными 1ми производными?


    allow_linear : bool, optional (default=True)
         Допускать ли добавление линейных ф-ций?


    feature_importance_type: string or list of strings, optional (default=None)
        Критерии важности признаков ('gcv', 'rss', 'nb_subsets').
        По умолчанию не вычисляется.


    endspan_alpha : float, optional, probability between 0 and 1 (default=0.05)
        Вероятности сделать больше endspan отклонений (см. статью для пояснения).
        ### Это можно реализовывать не сразу. Сначала проще реализовать ручное задание в endspan.


    endspan : int, optional (default=-1)
        Ручное задание интервала данных между узлом и краем.


    minspan_alpha : float, optional, probability between 0 and 1 (default=0.05)
        Вероятности сделать больше minspan отклонений (см. статью для пояснения).
        ### Это можно реализовывать не сразу. Сначала проще реализовать ручное задание в minspan.


    minspan : int, optional (default=-1)
        Ручное задание минимального интервала данных между соседними узлами.


    enable_pruning : bool, optional (default=True)
        Делать ли обратный проход?


    verbose : bool, optional(default=False)
        Выводить ли дополнительную информацию?
        ### (пока не по канону и только для отладки)



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
        ### (тип выхода определяем сами, хотя можно глянуть как делали авторы библиотеки)


    `pruning_pass_record_` : ???
        Информация по обратному проходу.
        ### (тип выхода определяем сами, хотя можно глянуть как делали авторы библиотеки)
    """ 


    ### Множества нужны для verbose, trace и т.д.
    ### Из py-earth пока не добавлены сюда:
    ### allow_missing, zero_tol, use_fast, fast_K, fast_h, check_every
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


    def __init__(self, max_terms=None, max_degree=None, allow_missing=False,
                 penalty=None, endspan_alpha=None, endspan=None,
                 minspan_alpha=None, minspan=None,
                 thresh=None, zero_tol=None, min_search_points=None,
                 check_every=None, allow_linear=None, use_fast=None, fast_K=None,
                 fast_h=None, smooth=None, enable_pruning=True,
                 feature_importance_type=None, verbose=0):

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
        self.terms_list = [1, ]  # terms_list = [B_1,..., B_M]

        ### Пока не реализуем
        # self.allow_missing = allow_missing
        # self.use_fast = use_fast
        # self.fast_K = fast_K
        # self.fast_h = fast_h
        # self.zero_tol = zero_tol


    ### Дополнительные ф-ции, которые использовались в py-earth. Следовать этому вообще не обязательно,
    ### просто для представления структуры отдельных блоков.
    # def __eq__(self, other):
    # def __ne__(self, other):
    # def _pull_forward_args(self, **kwargs): "Pull named arguments relevant to the forward pass"
    # def _pull_pruning_args(self, **kwargs): "Pull named arguments relevant to the pruning pass"
    # def _scrape_labels(self, X): "Try to get labels from input data (for example, if X is a
                                  # pandas DataFrame).  Return None if no labels can be extracted"
    # def _scrub_x(self, X, missing, **kwargs): "Sanitize input predictors and extract column names if appropriate"
    # def _scrub(self, X, y, sample_weight, output_weight, missing, **kwargs): "Sanitize input data"


    ### Если какие-то параметры в последющих функциях не потребуется - ну значит не потребуются. Но для наглядности пусть будут все.
    def fit(self, X, y=None,
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
        pass


    @staticmethod
    def relu_func(x, s, v, t):
        '''
        Ф-ция, реализующая положительную срезку (усечённая степенная сплайновая ф-ция)
            [s * (x_v - t)]_+

        Параметры
        ----------
        x: 
        s: знак
        v: координата
        t: порог
        '''
        relu = max(s * (x[v] - t), 0)
        return relu


    @staticmethod
    def cubic_func(x, s, v, t, t_minus, t_plus):
        '''
        Ф-ция, реализующая кубический усечённый сплайн с непрерывной 1ой производной
            ...

        Параметры
        ----------
        x: 
        s: знак
        v: координата
        t: порог
        t_minus:
        t_plus:
        '''
        # знак положительный
        if s > 0:
            if x[v] >= t_plus:
                return x[v] - t
            # t_minus < x[v] < t_plus
            if x[v] > t_minus:
                p_plus = (2 * t_plus + t_minus - 3 * t) / ((t_plus - t_minus) ** 2)
                r_plus = (2 * t - t_plus - t_minus) / ((t_plus - t_minus) ** 3)
                return p_plus * ((x[v] - t_minus) ** 2) + r_plus * ((x[v] - t_minus) ** 3)
            # x[v] <= t_minus
            return 0
        # знак отрицательный
        else:
            if x[v] >= t_plus:
                return 0
            # t_minus < x[v] < t_plus
            if x[v] > t_minus:
                p_minus = (3 * t - 2 * t_minus - t_plus) / ((t_minus - t_plus) ** 2)
                r_minus = (t_minus + t_plus - 2 * t) / ((t_minus - t_plus) ** 3)
                return p_minus * ((x[v] - t_plus) ** 2) + r_minus * ((x[v] - t_plus) ** 3)
            # x[v] <= t_minus
            return t - x[v]


    ### Можно ещё какие-то ф-ции реализовать


    @staticmethod
    def term_calculation(x, term):
        # TODO:
        # нужно добавить аргумент под тип базисной функции (relu, cubic и т. д.)
        '''
        Ф-ция, реализующая вычисление б.ф. B(x):
            B(x) = [s_1 * (x_(v_1) - t_1)]_+ * ... * [s_Km * (x_(v_Km) - t_Km)]_+

        Параметры
        ----------
        x: 
        term:
        '''
        # если это не константная б.ф. B_1(x)
        if not isinstance(term, int):
            term_value = 1
            for prod in term:
                # TODO:
                # тут сложнее, потому что для cubic будут храниться (s, v, t, t_plus, t_minus)
                (s, v, t) = prod
                term_value *= Earth.relu_func(x, s, v, t)
                if term_value == 0:
                    return term_value
            return term_value
        else:
            return term
        
        

    @staticmethod
    def g_calculation(x, terms_list, coeffs_list):
        '''
        Ф-ция, реализующая ф-цию g(x):
            g(x) = a_1 * B_1(x) + ... + a_M * B_M(x)

        Параметры
        ----------
        x: 
        terms_list:
        coeffs_list:
        '''
        g_value = 0
        for ind in range(len(terms_list)):
            g_value += coeffs_list[ind] * Earth.term_calculation(x, terms_list[ind])
        return g_value


    @staticmethod
    def GCV(f, d=3):
        '''
        Generalized Cross-Validation criterion:
            GCV() = 1/N * sum[(y_i - f(x_i))^2]/[1 - C_correct(M)/N]^2
            C(M) = tr(B @ (B^T @ B)^(-1) @ B^T) + 1
            C_correct(M) = C(M) + d * M
            C(M) - "число лин. нез. б.ф."
        
        f: 
        M: (хотя мб и не надо передавать)
        d: параметр сглаживания (обычно 2-4) чем больше, тем меньше узлов
        '''
        pass
        

    def forward_pass(self, X, y=None,
                     sample_weight=None,
                     output_weight=None,
                     missing=None,
                     xlabels=None, linvars=[],
                     skip_scrub=False):
        """
        Отдельно проход вперёд.


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

        # Пока реализуем самый простой вариант MARS (алгоритм 2 из оригинальной статьи)
        # В комментариях приведены обозначения из оригинальной статьи


        # Обозначения:
        # g(x) - модель: 
        #   g(x) = a_1 * B_1(x) + ... + a_M * B_M(x)
        #     M - итоговое кол-во базисных ф-ций (б.ф.)
        # B_i - i-ая б.ф. ---> term
        #   B_i(x) = [s_1 * (x_(v_1) - t_1)]_+ * ... * [s_Km * (x_(v_Km) - t_Km)]_+
        #   B_1 = 1 (константная б.ф.)
        #   составляющие множители б.ф. ---> hinge
        #     [...]_+ - положительная срезка
        #     Km - общее кол-во множителей в m-ой б.ф.
        #     s_j - знак j-ого множителя
        #     v_j - координата x j-ого множителя
        #     t_j - порог j-ого множителя
        # a_i - коэф-т при i-ой б.ф.
        # x = (x_1,...,x_d), d - размерность ---> data_dim

        
        data_count, data_dim = X.shape
        terms_count = 2     # M = 2

        # создаём б.ф. пока не достигнем макс. кол-ва
        while terms_count <= self.max_terms:
            best_lof = float('inf')  # lof* = inf

            # перебираем уже созданные б.ф.
            for term in self.terms_list:
                # формируем мн-во невалидных координат (уже использованных)
                not_valid_coords = []
                # если это не константная б.ф. B_1
                if term != 1:
                    for prod in term:
                        (s, v, t) = prod
                        not_valid_coords.append(v)
                # формируем мн-во валидных координат
                valid_coords = [coord for coord in range(0, data_dim)
                                if coord not in not_valid_coords]

                # перебираем все ещё не занятые координаты
                for v in valid_coords:
                    # TODO:
                    # здесь в переборе нужно дописать версию для cubic
                    # t_plus и t_minus предлагается выбрать как среднее между 
                    # t и соседними узлами справа и слева

                    # перебираем обучающие данные
                    for ind in range(data_count):
                        # учитываем только нетривиальные пороги
                        if Earth.term_calculation(X[ind], term) == 0:
                            continue
                        g = ...
                        lof = ...
                        if lof < best_lof:
                            best_lof = lof
                            best_term = term
                            best_v = v
                            best_t = t
            self.terms_list.append()
                    

    def pruning_pass(self, X, y=None,
                      sample_weight=None,
                      output_weight=None,
                      missing=None,
                      skip_scrub=False):
        """
        Отдельно проход назад.


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
        data_count, data_dim = X.shape

        # названия переменных взяты из статьи
        best_K = list(self.terms_list)
        best_lof = ...
        for M in range(len(self.terms_list), 1, -1):
            b = float('inf')
            L = list(best_K)
            for m in range(M):
                K = list(L)
                K.remove(L[m])
                # считаем lof для K
                lof = ...
                if lof < b:
                    b = lof
                    best_K = K
                if lof < best_lof:
                    best_lof = lof
                    self.terms_list = K


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


    def linear_fit(self, X, y=None,
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
            (пока забиваем)
            ...


        Выход
        ----------
        y : array of shape = [m] или [m, p], где m - кол-во объектов, p - кол-во выходов
            Прогнозы.
        """
        pass


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


    def score(self, X, y=None,
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
        ### Не очень понял что это надо разобраться, чтобы потом можно было сравнивать свою реализацию и их.

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
        pass


    def get_penalty(self):
        """
        Возвращает параметр сглаживания d из C_new(M) = C(M) + d*M.
        """
        pass