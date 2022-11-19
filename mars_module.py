import numpy as np
from scipy import sparse
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
        ### Заполняется аналогично fit.

        ### Что такое skip_scrub?
        """
        pass


    def pruning_pass(self, X, y=None,
                      sample_weight=None,
                      output_weight=None,
                      missing=None,
                      skip_scrub=False):
        """
        Отдельно проход назад.

        Параметры
        ----------
        ### Заполняется аналогично fit.
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