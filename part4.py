import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
from sklearn.utils.validation import _deprecate_positional_args

folds = 5
seed = 1
kf = KFold(n_splits=5, shuffle=True, random_state=2021)
lgb_params = {'lambda_l1': 0.0005898262404628944, 'lambda_l2': 0.0026596119232511423, 'num_leaves': 233, 'learning_rate': 0.0010456924170680736, 'n_estimators': 2168, 'feature_fraction': 0.4974578727308233, 'bagging_fraction': 0.7353514883774996, 'bagging_freq': 7, 'min_child_samples': 9}
xgb_params = {'lambda_l1': 8.1774182136912e-08, 'lambda_l2': 3.8077681237375737e-06, 'num_leaves': 114, 'learning_rate': 0.35898484150217996, 'n_estimators': 1677, 'feature_fraction': 0.4802817585097421, 'bagging_fraction': 0.714465669890466, 'bagging_freq': 4, 'min_child_samples': 88}




# modified code for group gaps; source
# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(self,
                 n_splits=5,
                 *,
                 max_train_group_size=np.inf,
                 max_test_group_size=np.inf,
                 group_gap=None,
                 verbose=False
                 ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError(
                "The 'groups' parameter should not be None")
        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)
        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size
        n_folds = n_splits + 1
        group_dict = {}
        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)
        for idx in np.arange(n_samples):
            if (groups[idx] in group_dict):
                group_dict[groups[idx]].append(idx)
            else:
                group_dict[groups[idx]] = [idx]
        if n_folds > n_groups:
            raise ValueError(
                ("Cannot have number of folds={0} greater than"
                 " the number of groups={1}").format(n_folds,
                                                     n_groups))

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(n_groups - n_splits * group_test_size,
                                  n_groups, group_test_size)
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)
            for train_group_idx in unique_groups[group_st:(group_test_start - group_gap)]:
                train_array_tmp = group_dict[train_group_idx]

                train_array = np.sort(np.unique(
                                      np.concatenate((train_array,
                                                      train_array_tmp)),
                                      axis=None), axis=None)

            train_end = train_array.size

            for test_group_idx in unique_groups[group_test_start:
                                                group_test_start +
                                                group_test_size]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(np.unique(
                                              np.concatenate((test_array,
                                                              test_array_tmp)),
                                     axis=None), axis=None)

            test_array  = test_array[group_gap:]


            yield [int(i) for i in train_array], [int(i) for i in test_array]

# 1.基础代码
def stacking_reg(
        clf, train_x, train_y, test_x, clf_name, kf, label_split=None
    ):
    train = np.zeros((train_x.shape[0], 1))
    test = np.zeros((test_x.shape[0], 1))
    test_pre = np.empty((folds, test_x.shape[0], 1))
    cv_scores = []
    try:
        temp = kf.split(train_x, groups=label_split)
    except:
        temp = kf.split(train_x, label_split=label_split)
    for i, (train_index, test_index) in enumerate(temp):
        tr_x = train_x[train_index]
        tr_y = train_y[train_index]
        te_x = train_x[test_index]
        te_y = train_y[test_index]
        if clf_name in ['rf', 'ada', 'gb', 'et', 'lr', 'lsvc', 'knn']:
            clf.fit(tr_x, tr_y)
            pre = clf.predict(te_x).reshape(-1, 1)
            train[test_index] = pre
            test_pre[i, :] = clf.predict(test_x).reshape(-1, 1)
            cv_scores.append(mean_squared_error(te_y, pre))
            
        elif clf_name in ['xgb']:
            train_mat = clf.DMatrix(tr_x, label=tr_y, missing=-1)
            test_mat = clf.DMatrix(te_x, label=te_y, missing=-1)
            z = clf.DMatrix(test_x, label=np.ones(test_x.shape[0]), missing=-1)
            num_round = 10000
            early_stopping_rounds = 100
            watch_list = [(train_mat, 'train'), (test_mat, 'eval')]
            if test_mat:
                model = clf.train(
                    xgb_params, 
                    train_mat,
                    num_boost_round=num_round,
                    evals=watch_list,
                    early_stopping_rounds=early_stopping_rounds
                )
                pre = model.predict(
                    test_mat,
                    ntree_limit=model.best_ntree_limit
                ).reshape(-1, 1)
                train[test_index] = pre
                test_pre[i, :] = model.predict(
                    z, 
                    ntree_limit=model.best_ntree_limit
                ).reshape(-1, 1)
                
                cv_scores.append(mean_squared_error(te_y, pre))

        elif clf_name in ['lgb']:
            train_mat = clf.Dataset(tr_x, label=tr_y)
            test_mat = clf.Dataset(te_x, label=te_y)
            num_round = 7000
            early_stopping_rounds = 100
            if test_mat:
                print(train_mat)
                model = clf.train(
                    lgb_params, 
                    train_mat,
                    num_round,
                    valid_sets=test_mat,
                    # early_stopping_rounds=early_stopping_rounds
                )
                pre = model.predict(
                    te_x,
                    num_iteration=model.best_iteration
                ).reshape(-1, 1)
                train[test_index] = pre
                test_pre[i, :] = model.predict(
                    test_x, num_iteration=model.best_iteration
                ).reshape(-1, 1)
                cv_scores.append(mean_squared_error(te_y, pre))
        else:
            raise IOError('please add new clf.')
        print(f"{clf_name} now score is {cv_scores}")
    test[:] = test_pre.mean(axis=0)
    print(f"{clf_name}_score_list:{cv_scores}")
    print(f"{clf_name}_Score_mean:{np.mean(cv_scores)}")
    return train.reshape(-1, 1), test.reshape(-1, 1)

# 2 模型融合stacking基学习器
def rf_reg(x_train, y_train, x_valid, kf, label_split=None):
    rf = RandomForestRegressor(
        n_estimators=600,
        max_depth=20,
        n_jobs=-1,
        random_state=2021,
        max_features='auto',
        verbose=1
    )
    train, test = stacking_reg(
        rf, x_train, y_train, x_valid, 'rf', kf, label_split=label_split
    )
    return train, test, 'rf_reg'

def ada_reg(x_train, y_train, x_valid, kf, label_split=None):
    ad = AdaBoostRegressor(
        n_estimators=30,
        random_state=2021,
        learning_rate=.01
    )
    train, test = stacking_reg(
        ad, x_train, y_train, x_valid, 'ada', kf, label_split=label_split
    )
    return train, test, 'ada_reg'

def gb_reg(x_train, y_train, x_valid, kf, label_split=None):
    gbdt = GradientBoostingRegressor(
        learning_rate=.04,
        n_estimators=100,
        subsample=.8,
        random_state=2021,
        max_depth=5,
        verbose=1
    )
    train, test = stacking_reg(
        gbdt, x_train, y_train, x_valid, 'gb', kf, label_split=label_split
    )
    return train, test, 'gb'


def xgb_reg(x_train, y_train, x_valid, kf, label_split=None):
    train, test = stacking_reg(
        xgb, x_train, y_train, x_valid, 'xgb', kf, label_split=label_split
    )
    return train, test, 'xgb_reg'

def lgb_reg(x_train, y_train, x_valid, kf, label_split=None):
    train, test = stacking_reg(
        lgb, x_train, y_train, x_valid, 'lgb', kf, label_split=label_split
    )
    return train, test, 'lgb_reg'

# 3 模型融合stacking预测函数
def stacking_pred(
    x_train, y_train, x_valid, kf, clf_list, label_split=None, clf_fin='lgb', if_concat_origin=True
):
    for clf_list in clf_list:
        clf_list = [clf_list]
        col_list, train_data_list, test_data_list = [], [], []
        for clf in clf_list:
            train_data, test_data, clf_name = clf(
                x_train,
                y_train,
                x_valid,
                kf,
                label_split=label_split
            )
            train_data_list.append(train_data)
            test_data_list.append(test_data)
            col_list.append(f"clf_{clf_name}")
        train, test = np.concatenate(train_data_list, axis=1), np.concatenate(test_data_list, axis=1)


        if if_concat_origin:
            train = np.concatenate([x_train, train], axis=1)
            test = np.concatenate([x_valid, test], axis=1)
        print('train:', train.shape)
        print('test:', test.shape)
        print(x_train.shape)
        print(train.shape)
        print(clf_name)
        print(clf_name in ['lgb'])
        if clf_fin in ['rf', 'ada', 'gb', 'et', 'lr', 'lsvc', 'knn']:
            if clf_fin in ['rf']:
                clf = RandomForestRegressor(
                    n_estimators=600,
                    max_depth=20,
                    n_jobs=-1,
                    random_state=2021,
                    max_features='auto',
                    verbose=1
                )
            elif clf_fin in ['ada']:
                clf = AdaBoostRegressor(
                    n_estimators=30,
                    random_state=2021,
                    learning_rate=.01,
                )
            elif clf_fin in ['gb']:
                clf = GradientBoostingRegressor(
                    learning_rate=.04,
                    n_estimators=100,
                    subsample=.8,
                    random_state=201,
                    max_depth=5,
                    verbose=1
                )
        elif clf_fin in ['xgb']:
            return _extracted_from_stacking_pred_56(train, y_train, test)
        elif clf_fin in ['lgb']:
            return _extracted_from_stacking_pred_72(clf_name, train, y_train, test)    


# TODO Rename this here and in `stacking_pred`
def _extracted_from_stacking_pred_72(clf_name, train, y_train, test):
    print(clf_name)
    clf = lgb
    train_mat = clf.Dataset(train, label=y_train)
    test_mat = clf.Dataset(train, label=y_train)
    num_round = 10000
    early_stopping_rounds = 1000
    model = clf.train(
        lgb_params, 
        train_mat,
        num_round,
        valid_sets=test_mat,
        # early_stopping_rounds=early_stopping_rounds
    )
    print('pred')
    pre = model.predict(
        test,
        num_iteration=model.best_iteration
    ).reshape(-1, 1)
    print(pre)
    return pre    


# TODO Rename this here and in `stacking_pred`
def _extracted_from_stacking_pred_56(train, y_train, test):
    clf = xgb
    train_mat = clf.DMatrix(train, label=y_train, missing=-1)
    test_mat = clf.DMatrix(train, label=y_train, missing=-1)
    num_round = 10000
    early_stopping_rounds = 100
    watch_list = [(train_mat, 'train'), (test_mat, 'eval')]
    model = clf.train(
        xgb_params, 
        train_mat,
        num_boost_round=num_round,
        evals=watch_list,
        early_stopping_rounds=early_stopping_rounds
    )
    return model.predict(clf.DMatrix(test), ntree_limit=model.best_ntree_limit).reshape(-1, 1)    