__author__ = 'aclapes'

import numpy as np
from sklearn import svm
from sklearn.metrics import average_precision_score, label_ranking_average_precision_score, make_scorer
import utils
from copy import deepcopy
from sklearn import preprocessing, cross_validation, grid_search
from sklearn.ensemble import RandomForestClassifier

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

INTERNAL_PARAMETERS = dict(
    weights = None
)

svm_parameters = {'C': [1e-4,1e-3,1e-2,1e-1,1,10,100,1e3,1e4], \
                  'gamma' : [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3, 1e4], \
                  'kernel':['rbf'], 'class_weight':['balanced']}

rbf_parameters = {'n_estimators': [10,20,30,40],
              'max_features': ['auto','sqrt','log2',0.5,0.3],
                  'class_weight':['balanced']}

# def merge(input_kernels):
#     kernels_train = input_kernels[0]['train']
#     for i,k in enumerate(input_kernels):
#         for j,feat
#         kernels_train[i]['train']


def classify(input_kernels, class_labels, traintest_parts, a, feat_types, strategy='kernel_fusion', C=[1], opt_criterion='acc'):
    '''
    TODO Fill this.
    :param feats_path:
    :param class_labels:
    :param traintest_parts:
    :param a:
    :param feat_types:
    :param c:
    :return:
    '''

    results = [None] * len(traintest_parts)
    for k, part in enumerate(traintest_parts):
        train_inds, test_inds = np.where(part <= 0)[0], np.where(part > 0)[0]

        kernels_train = input_kernels[k]['train']
        kernels_test  = input_kernels[k]['test']
        if strategy == 'kernel_fusion':
            results[k] = kernel_fusion_classification(kernels_train, kernels_test, a, feat_types, class_labels, (train_inds, test_inds), \
                                                      C=C, opt_criterion=opt_criterion)
        elif strategy == 'simple_voting':
            results[k] = simple_voting_classification(kernels_train, kernels_test, a, feat_types, class_labels, (train_inds, test_inds), \
                                                      C=C, opt_criterion=opt_criterion)
        elif strategy == 'learning_based_fusion':
            results[k] = learning_based_fusion_classification(kernels_train, kernels_test, a, feat_types, class_labels, (train_inds, test_inds), \
                                                              C=C, opt_criterion=opt_criterion)

    return results



# ==============================================================================
# Helper functions
# ==============================================================================

# def print_progressbar(value, size=20, percent=True):
#     """
#     Print progress bar with value as an ASCII bar in the console.
#     :param value: progress value ranging within [0-1]
#     :param size: width of the bar
#     :param percent: print the progress as a % value, if not print in the range
#     :return:
#     """
#     bar_fill = '#'*int(np.floor(size*value))+'-'*int(np.ceil(size*(1-value)))
#     bar_expr = '\r[{:}]\t{:.3}' if not percent else '\r[{:}]\t{:.1%}'
#     print(bar_expr.format(bar_fill, value)),


# def train_and_classify(input_kernels_tr, input_kernels_te, a, feat_types, class_labels, train_test_idx, c=[1], nl=1):
#     '''
#
#     :param kernels_tr:
#     :param kernels_te:
#     :param a: trade-off parameter controlling importance of root representation vs edges representation
#     :param feat_types:
#     :param class_labels:
#     :param train_test_idx:
#     :param c:
#     :return:
#     '''
#
#     # Assign weights to channels
#     feat_weights = INTERNAL_PARAMETERS['weights']
#     if feat_weights is None: # if not specified a priori (when channels' specification)
#         feat_weights = {feat_t : 1.0/len(input_kernels_tr) for feat_t in input_kernels_tr.keys()}
#
#     tr_inds, te_inds = train_test_idx[0], train_test_idx[1]
#     # lb = LabelBinarizer(neg_label=-1, pos_label=1)
#
#     class_ints = np.dot(class_labels, np.logspace(0, class_labels.shape[1]-1, class_labels.shape[1]))
#     skf = StratifiedKFold(class_ints[tr_inds], n_folds=4, shuffle=False, random_state=42)
#
#     S = [None] * class_labels.shape[1]  # selected (best) params
#     p = [None] * class_labels.shape[1]  # performances
#     C = [(a,c) for k in xrange(class_labels.shape[1])]  # candidate values for params
#
#     Rval_ap = np.zeros((class_labels.shape[1], len(a), len(c)), dtype=np.float32)
#     for k in xrange(class_labels.shape[1]):
#         for l in xrange(nl):
#             for i, a_i in enumerate(C[k][0]):
#                 kernels_tr = deepcopy(input_kernels_tr)
#                 kernels_te = deepcopy(input_kernels_te)
#                 for feat_t in kernels_tr.keys():
#                     kernels_tr[feat_t]['root'] = sum_of_arrays(kernels_tr[feat_t]['root'], [1,0], norm=None)
#                     kernels_tr[feat_t]['nodes'] = sum_of_arrays(kernels_tr[feat_t]['nodes'], [a_i[0], 1-a_i[0]], norm=None)
#                 for feat_t in kernels_te.keys():
#                     kernels_te[feat_t]['root'] = sum_of_arrays(kernels_te[feat_t]['root'], [1, 0], norm=None)
#                     kernels_te[feat_t]['nodes'] = sum_of_arrays(kernels_te[feat_t]['nodes'], [a_i[0], 1-a_i[0]], norm=None)
#
#                 K_tr = None
#                 # Weight each channel accordingly
#                 for feat_t in kernels_tr.keys():
#                     Kr_tr, _ = normalize_by_median(kernels_tr[feat_t]['root'])
#                     Kn_tr, _ = normalize_by_median(kernels_tr[feat_t]['nodes'])
#                     if K_tr is None:
#                         K_tr = np.zeros(Kr_tr.shape, dtype=np.float32)
#                     K_tr += feat_weights[feat_t] * (a_i[1]*Kr_tr + (1-a_i[1])*Kn_tr)
#
#                 for j, c_j in enumerate(C[k][1]):
#                     # print l, str(i+1) + '/' + str(len(C[k][0])), str(j+1) + '/' + str(len(C[k][1]))
#                     Rval_ap[k,i,j] = 0
#                     for (val_tr_inds, val_te_inds) in skf:
#                         # test instances not indexed directly, but a mask is created excluding negative instances
#                         val_te_msk = np.ones(tr_inds.shape, dtype=np.bool)
#                         val_te_msk[val_tr_inds] = False
#                         negatives_msk = np.negative(np.any(class_labels[tr_inds] > 0, axis=1))
#                         val_te_msk[negatives_msk] = False
#
#                         acc_tmp, ap_tmp = _train_and_classify_binary(
#                             K_tr[val_tr_inds,:][:,val_tr_inds], K_tr[val_te_msk,:][:,val_tr_inds], \
#                             class_labels[tr_inds,k][val_tr_inds], class_labels[tr_inds,k][val_te_msk], \
#                             c_j)
#                         # TODO: decide what it is
#                         Rval_ap[k,i,j] += acc_tmp/skf.n_folds
#                         # Rval_ap[k,i,j] += (ap_tmp/skf.n_folds if acc_tmp > 0.5 else 0)
#
#             a_bidx, c_bidx = np.unravel_index(Rval_ap[k].argmax(), Rval_ap[k].shape)  # a and c bests' indices
#             S[k] = (C[k][0][a_bidx], C[k][1][c_bidx])
#             p[k] = Rval_ap.max()
#
#             # a0_new = np.linspace(C[k][0][a_bidx-1 if a_bidx > 0 else a_bidx][0], \
#             #                      C[k][0][a_bidx+1 if a_bidx < len(a)-1 else a_bidx][0], np.sqrt(len(a)))
#             # a1_new = np.linspace(C[k][0][a_bidx-1 if a[a_bidx] > 0 else a_bidx][1], \
#             #                      C[k][0][a_bidx+1 if a_bidx < len(a)-1 else a_bidx][1], np.sqrt(len(a)))
#             # a_new = [c for c in itertools.product(*[a0_new,a1_new])]
#             c_new = np.linspace(C[k][1][c_bidx-1 if c_bidx > 0 else c_bidx], C[k][1][c_bidx+1 if c_bidx < len(c)-1 else c_bidx], len(c))
#
#             C[k] = (a, c_new)
#
#     # X, Y = np.meshgrid(np.linspace(0,len(c)-1,len(c)),np.linspace(0,len(a)-1,len(a)))
#     # fig = plt.figure(figsize=plt.figaspect(0.5))
#     # for k in xrange(class_labels.shape[1]):
#     #     ax = fig.add_subplot(2,5,k+1, projection='3d')
#     #     ax.plot_surface(X, Y, Rval_acc[k,:,:])
#     #     ax.set_zlim([0.5, 1])
#     #     ax.set_xlabel('c value')
#     #     ax.set_ylabel('a value')
#     #     ax.set_zlabel('acc [0-1]')
#     # plt.show()
#
#     te_msk = np.ones((len(te_inds),), dtype=np.bool)
#     negatives_msk = np.negative(np.any(class_labels[te_inds] > 0, axis=1))
#     te_msk[negatives_msk] = False
#
#     acc_classes = []
#     ap_classes = []
#     for k in xrange(class_labels.shape[1]):
#         a_best = S[k][0]
#         print a_best
#
#         kernels_tr = deepcopy(input_kernels_tr)
#         kernels_te = deepcopy(input_kernels_te)
#         for feat_t in kernels_tr.keys():
#             kernels_tr[feat_t]['root'] = sum_of_arrays(kernels_tr[feat_t]['root'], [1, 0], norm=None)
#             kernels_tr[feat_t]['nodes'] = sum_of_arrays(kernels_tr[feat_t]['nodes'], [a_best[0], 1-a_best[0]], norm=None)
#         for feat_t in kernels_te.keys():
#             kernels_te[feat_t]['root'] = sum_of_arrays(kernels_te[feat_t]['root'], [1, 0], norm=None)
#             kernels_te[feat_t]['nodes'] = sum_of_arrays(kernels_te[feat_t]['nodes'], [a_best[0], 1-a_best[0]], norm=None)
#
#         # normalize kernel (dividing by the median value of training's kernel)
#         K_tr = K_te = None
#         for feat_t in kernels_tr.keys():
#             Kr_tr, mr_tr = normalize_by_median(kernels_tr[feat_t]['root'])
#             Kn_tr, me_tr = normalize_by_median(kernels_tr[feat_t]['nodes'])
#
#             Kr_te, _ = normalize_by_median(kernels_te[feat_t]['root'], p=mr_tr)
#             Kn_te, _ = normalize_by_median(kernels_te[feat_t]['nodes'], p=me_tr)
#
#             if K_tr is None:
#                 K_tr = np.zeros(Kr_tr.shape, dtype=np.float32)
#             K_tr += feat_weights[feat_t] * (a_best[1]*Kr_tr + (1-a_best[1])*Kn_tr)
#
#             if K_te is None:
#                 K_te = np.zeros(Kr_te.shape, dtype=np.float32)
#             K_te += feat_weights[feat_t] * (a_best[1]*Kr_te + (1-a_best[1])*Kn_te)
#
#         c_best = S[k][1]
#         acc, ap = _train_and_classify_binary(K_tr, K_te[te_msk], class_labels[tr_inds,k], class_labels[te_inds,k][te_msk], c=c_best)
#
#         acc_classes.append(acc)
#         ap_classes.append(ap)
#
#     return dict(acc_classes=acc_classes, ap_classes=ap_classes)


def kernel_fusion_classification(input_kernels_tr, input_kernels_te, a, feat_types, class_labels, train_test_idx, \
                                 C=[1], square_kernels=False, opt_criterion='acc'):
    '''

    :param kernels_tr:
    :param kernels_te:
    :param a: trade-off parameter controlling importance of root representation vs edges representation
    :param feat_types:
    :param class_labels:
    :param train_test_idx:
    :param c:
    :return:
    '''

    # Assign weights to channels
    feat_weights = INTERNAL_PARAMETERS['weights']
    if feat_weights is None: # if not specified a priori (when channels' specification)
        feat_weights = {feat_t : 1.0/len(input_kernels_tr) for feat_t in input_kernels_tr.keys()}

    tr_inds, te_inds = train_test_idx[0], train_test_idx[1]
    # lb = LabelBinarizer(neg_label=-1, pos_label=1)

    class_ints = np.dot(class_labels, np.logspace(0, class_labels.shape[1]-1, class_labels.shape[1]))
    skf = cross_validation.StratifiedKFold(class_ints[tr_inds], n_folds=4, shuffle=False, random_state=74)

    Rval = np.zeros((class_labels.shape[1], len(a), len(C)), dtype=np.float32)
    for k in xrange(class_labels.shape[1]):
        print "[Validation] Optimizing weights and svm-C for class %d/%d" % (k+1, class_labels.shape[1])
        for i, a_i in enumerate(a):
            kernels_tr = deepcopy(input_kernels_tr)
            # kernels_te = deepcopy(input_kernels_te)

            for feat_t in kernels_tr.keys():
                if isinstance(kernels_tr[feat_t]['root'], tuple):
                    kernels_tr[feat_t]['root'] = utils.normalize(kernels_tr[feat_t]['root'][0])
                    x = kernels_tr[feat_t]['nodes']
                    kernels_tr[feat_t]['nodes'] = utils.normalize(a_i[1]*x[0]+(1-a_i[1])*x[1] if len(x)==2 else x[0])

                    kernels_tr[feat_t] = a_i[2]*np.array(kernels_tr[feat_t]['root']) + (1-a_i[2])*np.array(kernels_tr[feat_t]['nodes'])

                else:
                    kernels_tr[feat_t]['root']  = [utils.normalize(x[0] if np.sum(x[0])>0 else kernels_tr[feat_t]['nodes'][j][0])
                                                   for j,x in enumerate(kernels_tr[feat_t]['root'])]
                    kernels_tr[feat_t]['nodes'] = [utils.normalize(a_i[1]*x[0]+(1-a_i[1])*x[1] if len(x)==2 else x[0])
                                                   for x in kernels_tr[feat_t]['nodes']]

                    # for j,Kr in enumerate(kernels_tr[feat_t]['root']):
                    #     K = None
                    #     if np.sum(Kr) > 0:
                    #         K = a_i[2]*Kr + (1-a_i[2])*kernels_tr[feat_t]['nodes'][j]
                    #     else:
                    #         K = kernels_tr[feat_t]['nodes'][j]
                    #     kernels_tmp.append(K)
                    # kernels_tr[feat_t] = kernels_tmp
                    kernels_tr[feat_t] = list(a_i[2]*np.array(kernels_tr[feat_t]['root']) + (1-a_i[2])*np.array(kernels_tr[feat_t]['nodes']))

            K_tr = None
            # Weight each channel accordingly
            for j, feat_t in enumerate(kernels_tr.keys()):
                if K_tr is None:
                    K_tr = np.zeros(kernels_tr[feat_t].shape if isinstance(kernels_tr[feat_t],np.ndarray) else kernels_tr[feat_t][0].shape, dtype=np.float32)
                K_tr += a_i[3][j] * utils.sum_of_arrays(kernels_tr[feat_t], a_i[0])

            if square_kernels:
                K_tr = np.sqrt(K_tr)

            for j, c_j in enumerate(C):
                # print l, str(i+1) + '/' + str(len(C[k][0])), str(j+1) + '/' + str(len(C[k][1]))
                Rval[k,i,j] = 0
                for (val_tr_inds, val_te_inds) in skf:
                    # test instances not indexed directly, but a mask is created excluding negative instances
                    val_te_msk = np.ones(tr_inds.shape, dtype=np.bool)
                    val_te_msk[val_tr_inds] = False
                    negatives_msk = np.negative(np.any(class_labels[tr_inds] > 0, axis=1))
                    val_te_msk[negatives_msk] = False

                    acc_tmp, ap_tmp = _train_and_classify_binary(
                        K_tr[val_tr_inds,:][:,val_tr_inds], K_tr[val_te_msk,:][:,val_tr_inds], \
                        class_labels[tr_inds,k][val_tr_inds], class_labels[tr_inds,k][val_te_msk], \
                        c_j)

                    if str.lower(opt_criterion) == 'map':
                        Rval[k,i,j] += ap_tmp/skf.n_folds if acc_tmp > 0.50 else 0
                    else: # 'acc' or other criterion
                        Rval[k,i,j] += acc_tmp/skf.n_folds

    # print p, np.mean(p)

    # X, Y = np.meshgrid(np.linspace(0,len(c)-1,len(c)),np.linspace(0,len(a)-1,len(a)))
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # for k in xrange(class_labels.shape[1]):
    #     ax = fig.add_subplot(2,5,k+1, projection='3d')
    #     ax.plot_surface(X, Y, Rval_acc[k,:,:])
    #     ax.set_zlim([0.5, 1])
    #     ax.set_xlabel('c value')
    #     ax.set_ylabel('a value')
    #     ax.set_zlabel('acc [0-1]')
    # plt.show()



    te_msk = np.ones((len(te_inds),), dtype=np.bool)
    negatives_msk = np.negative(np.any(class_labels[te_inds] > 0, axis=1))
    te_msk[negatives_msk] = False

    results_val = []
    for k in xrange(class_labels.shape[1]):
        best_res = np.max(Rval[k,:,:])
        print best_res, "\t",
        results_val.append(best_res)
    print "Validation ACC:", np.mean(results_val)


    acc_classes = []
    ap_classes = []
    for k in xrange(class_labels.shape[1]):
        i,j = np.unravel_index(np.argmax(Rval[k,:,:]), Rval[k,:,:].shape)
        a_best, c_best = a[i], C[j]
        print a_best, c_best

        kernels_tr = deepcopy(input_kernels_tr)
        kernels_te = deepcopy(input_kernels_te)

        for feat_t in kernels_tr.keys():
            if isinstance(kernels_tr[feat_t]['root'], tuple):
                kernels_tr[feat_t]['root'], pr = utils.normalization(kernels_tr[feat_t]['root'][0])
                kernels_te[feat_t]['root']     = pr * kernels_te[feat_t]['root'][0]

                xn_tr, xn_te = kernels_tr[feat_t]['nodes'], kernels_te[feat_t]['nodes']
                kernels_tr[feat_t]['nodes'], pn = utils.normalization(a_best[1]*xn_tr[0]+(1-a_best[1])*xn_tr[1] if len(xn_tr)==2 else xn_tr[0])
                kernels_te[feat_t]['nodes']     = pn * (a_best[1]*xn_te[0]+(1-a_best[1])*xn_te[1] if len(xn_te)==2 else xn_te[0])

                kernels_tr[feat_t] = a_best[2]*kernels_tr[feat_t]['root'] + (1-a_best[2])*kernels_tr[feat_t]['nodes']
                kernels_te[feat_t] = a_best[2]*kernels_te[feat_t]['root'] + (1-a_best[2])*kernels_te[feat_t]['nodes']
            else:
                for i in xrange(len(kernels_tr[feat_t]['root'])):
                    kernels_tr[feat_t]['root'][i], pr  = utils.normalization(kernels_tr[feat_t]['root'][i][0] if np.sum(kernels_tr[feat_t]['root'][i][0]) > 0
                                                                             else kernels_tr[feat_t]['nodes'][i][0])
                    kernels_te[feat_t]['root'][i]      = pr * (kernels_te[feat_t]['root'][i][0] if np.sum(kernels_tr[feat_t]['root'][i][0]) > 0
                                                               else kernels_te[feat_t]['nodes'][i][0])

                    xn_tr, xn_te = kernels_tr[feat_t]['nodes'][i], kernels_te[feat_t]['nodes'][i]
                    kernels_tr[feat_t]['nodes'][i], pn = utils.normalization(a_best[1]*xn_tr[0]+(1-a_best[1])*xn_tr[1] if len(xn_tr)==2 else xn_tr[0])
                    kernels_te[feat_t]['nodes'][i]     = pn * (a_best[1]*xn_te[0]+(1-a_best[1])*xn_te[1] if len(xn_te)==2 else xn_te[0])

                kernels_tr[feat_t] = list(a_best[2]*np.array(kernels_tr[feat_t]['root']) + (1-a_best[2])*np.array(kernels_tr[feat_t]['nodes']))
                kernels_te[feat_t] = list(a_best[2]*np.array(kernels_te[feat_t]['root']) + (1-a_best[2])*np.array(kernels_te[feat_t]['nodes']))

        K_tr = K_te = None
        # Weight each channel accordingly
        for j,feat_t in enumerate(kernels_tr.keys()):
            if K_tr is None:
                K_tr = np.zeros(kernels_tr[feat_t].shape if isinstance(kernels_tr[feat_t],np.ndarray) else kernels_tr[feat_t][0].shape, dtype=np.float32)
                K_te = np.zeros(kernels_te[feat_t].shape if isinstance(kernels_te[feat_t],np.ndarray) else kernels_te[feat_t][0].shape, dtype=np.float32)
            K_tr += a_best[3][j] * utils.sum_of_arrays(kernels_tr[feat_t], a_best[0])
            K_te += a_best[3][j] * utils.sum_of_arrays(kernels_te[feat_t], a_best[0])

        if square_kernels:
            K_tr, K_te = np.sqrt(K_tr), np.sqrt(K_te)
        acc, ap = _train_and_classify_binary(K_tr, K_te[te_msk], class_labels[tr_inds,k], class_labels[te_inds,k][te_msk], c=c_best)

        acc_classes.append(acc)
        ap_classes.append(ap)

    return dict(acc_classes=acc_classes, ap_classes=ap_classes)


def simple_voting_classification(input_kernels_tr, input_kernels_te, a, feat_types, class_labels, train_test_idx, C=[1], nl=1):
    '''

    :param kernels_tr:
    :param kernels_te:
    :param a: trade-off parameter controlling importance of root representation vs edges representation
    :param feat_types:
    :param class_labels:
    :param train_test_idx:
    :param C:
    :return:
    '''

    tr_inds, te_inds = train_test_idx[0], train_test_idx[1]
    # lb = LabelBinarizer(neg_label=-1, pos_label=1)

    class_ints = np.dot(class_labels, np.logspace(0, class_labels.shape[1]-1, class_labels.shape[1]))
    skf = cross_validation.StratifiedKFold(class_ints[tr_inds], n_folds=4, shuffle=False, random_state=74)

    # S = [None] * class_labels.shape[1]  # selected (best) params
    # p = [None] * class_labels.shape[1]  # performances
    # C = [(a, C) for k in xrange(class_labels.shape[1])]  # candidate values for params

    kernels_tr = []
    for feat_t in input_kernels_tr.keys():
        for k,v in input_kernels_tr[feat_t].iteritems():
            for x in v:
                if np.any(x != 0):
                    kernels_tr.append(x)

    Rp = [None] * class_labels.shape[1]
    for cl in xrange(class_labels.shape[1]):
        print "[Validation] Optimizing weights and svm-C for class %d/%d" % (cl + 1, class_labels.shape[1])

        if Rp[cl] is None:
            Rp[cl] = np.zeros((len(kernels_tr),len(a),len(C)), dtype=np.float32)

        for i, a_i in enumerate(a):
            for k, x in enumerate(kernels_tr):
                for j, c_j in enumerate(C):
                    for (val_tr_inds, val_te_inds) in skf:
                        # test instances not indexed directly, but a mask is created excluding negative instances
                        val_te_msk = np.ones(tr_inds.shape, dtype=np.bool)
                        val_te_msk[val_tr_inds] = False
                        negatives_msk = np.all(class_labels[tr_inds] <= 0, axis=1)
                        val_te_msk[negatives_msk] = False

                        K_tr = utils.normalize(a_i*x[0]+(1-a_i)*x[1]) if isinstance(x,tuple) else utils.normalize(x)
                        acc, ap = _train_and_classify_binary(
                            K_tr[val_tr_inds,:][:,val_tr_inds], K_tr[val_te_msk,:][:,val_tr_inds], \
                            class_labels[tr_inds,cl][val_tr_inds], class_labels[tr_inds,cl][val_te_msk], \
                            c=c_j)

                        # TODO: decide what it is
                        Rp[cl][k,i,j] += acc / skf.n_folds
                        # Rp[cl][k,i,j] += ap/skf.n_folds

    params = [ np.zeros((Rp[cl].shape[0],2),dtype=np.float32) ] * class_labels.shape[1]
    perfs = [ np.zeros((Rp[cl].shape[0],),dtype=np.float32) ] * class_labels.shape[1]
    for cl in xrange(class_labels.shape[1]):
        print cl
        for k in xrange(Rp[cl].shape[0]):
            P = Rp[cl][k,:,:]  # #{a}x#{C} performance matrix
            coords = np.unravel_index(np.argmax(P), P.shape)
            params[cl][k,0], params[cl][k,1] = a[coords[0]], C[coords[1]]
            print cl,k,params[cl][k]
            perfs[cl][k] = np.max(P)

    Rval_acc = np.zeros((class_labels.shape[1],))
    for cl in xrange(class_labels.shape[1]):
        print "[Validation] Optimizing weights and svm-C for class %d/%d" % (cl + 1, class_labels.shape[1])

        for (val_tr_inds, val_te_inds) in skf:
            # Remove negatives from test data
            val_te_msk = np.ones(tr_inds.shape, dtype=np.bool)
            val_te_msk[val_tr_inds] = False
            negatives_msk = np.all(class_labels[tr_inds] <= 0, axis=1)
            val_te_msk[negatives_msk] = False

            # Get the predictions of each and every kernel
            P = np.zeros((len(val_te_inds), params[cl].shape[0]))  # matrix of predictions
            for k,x in enumerate(kernels_tr):
                # Value of best parameters
                a_val, c_val = params[cl][k,0], params[cl][k,1]

                # Merge a kernel according to a_val
                K_tr = utils.normalize(a_val*x[0]+(1-a_val)*x[1]) if isinstance(x,tuple) else utils.normalize(x)

                # Train using c_val as SVM-C parameter
                clf = _train_binary(K_tr[val_tr_inds,:][:,val_tr_inds], class_labels[tr_inds,cl][val_tr_inds], c=c_val)
                P[:,k] = clf.predict(K_tr[val_te_inds,:][:,val_tr_inds])

            y_preds = 2*(np.sum(P,axis=1) > 0).astype('int')-1
            #y_preds[y_preds == 0] = -1
            y_true = class_labels[tr_inds,cl][val_te_msk]
            acc = average_binary_accuracy(y_true, y_preds)
            Rval_acc[cl] += acc / skf.n_folds

    print Rval_acc, np.mean(Rval_acc)
    print 'done'

    # print p, np.mean(p)

    # X, Y = np.meshgrid(np.linspace(0,len(c)-1,len(c)),np.linspace(0,len(a)-1,len(a)))
    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # for k in xrange(class_labels.shape[1]):
    #     ax = fig.add_subplot(2,5,k+1, projection='3d')
    #     ax.plot_surface(X, Y, Rval_acc[k,:,:])
    #     ax.set_zlim([0.5, 1])
    #     ax.set_xlabel('c value')
    #     ax.set_ylabel('a value')
    #     ax.set_zlabel('acc [0-1]')
    # plt.show()

    te_msk = np.ones((len(te_inds),), dtype=np.bool)
    negatives_msk = np.negative(np.any(class_labels[te_inds] > 0, axis=1))
    te_msk[negatives_msk] = False

    acc_classes = []
    ap_classes = []

    return dict(acc_classes=acc_classes, ap_classes=ap_classes)

def learning_based_fusion_classification(input_kernels_tr, input_kernels_te, a, feat_types, class_labels, train_test_idx, C=[1], nl=1):
    '''

    :param kernels_tr:
    :param kernels_te:
    :param a: trade-off parameter controlling importance of root representation vs edges representation
    :param feat_types:
    :param class_labels:
    :param train_test_idx:
    :param C:
    :return:
    '''

    tr_inds, te_inds = train_test_idx[0], train_test_idx[1]
    # lb = LabelBinarizer(neg_label=-1, pos_label=1)

    class_ints = np.dot(class_labels, np.logspace(0, class_labels.shape[1]-1, class_labels.shape[1]))
    skf = cross_validation.StratifiedKFold(class_ints[tr_inds], n_folds=4, shuffle=False, random_state=74)

    # S = [None] * class_labels.shape[1]  # selected (best) params
    # p = [None] * class_labels.shape[1]  # performances
    # C = [(a, C) for k in xrange(class_labels.shape[1])]  # candidate values for params

    kernels_tr = []
    for feat_t in input_kernels_tr.keys():
        for k,v in input_kernels_tr[feat_t].iteritems():
            for x in v:
                if np.any(x != 0):
                    kernels_tr.append(x)
    kernels_te = []
    for feat_t in input_kernels_te.keys():
        for k,v in input_kernels_te[feat_t].iteritems():
            for x in v:
                if np.any(x != 0):
                    kernels_te.append(x)

    Rp = [None] * class_labels.shape[1]
    for cl in xrange(class_labels.shape[1]):
        print "[Validation] Optimizing weights and svm-C for class %d/%d" % (cl + 1, class_labels.shape[1])
        Rp[cl] = np.zeros((len(kernels_tr),len(a),len(C)), dtype=np.float32)

        for i, a_i in enumerate(a):
            for k, x in enumerate(kernels_tr):
                K_tr = utils.normalize(a_i*x[0]+(1-a_i)*x[1]) if isinstance(x,tuple) else utils.normalize(x)

                for j, c_j in enumerate(C):
                    for (val_tr_inds, _) in skf:
                        # test instances not indexed directly, but a mask is created excluding negative instances
                        val_te_msk = np.ones(tr_inds.shape, dtype=np.bool)
                        val_te_msk[val_tr_inds] = False
                        negatives_msk = np.all(class_labels[tr_inds] <= 0, axis=1)
                        val_te_msk[negatives_msk] = False

                        acc, ap = _train_and_classify_binary(
                            K_tr[val_tr_inds,:][:,val_tr_inds], K_tr[val_te_msk,:][:,val_tr_inds], \
                            class_labels[tr_inds,cl][val_tr_inds], class_labels[tr_inds,cl][val_te_msk], \
                            c=c_j)

                        # TODO: decide what it is
                        # Rp[cl][k,i,j] += acc / skf.n_folds
                        Rp[cl][k,i,j] += ap/skf.n_folds if acc > 0.5 else 0

    params = [None] * class_labels.shape[1]
    perfs = [None] * class_labels.shape[1]
    for cl in xrange(class_labels.shape[1]):
        # if params[cl] is None:
        params[cl] = np.zeros((Rp[cl].shape[0],2),dtype=np.float32)
        perfs[cl]  = np.zeros((Rp[cl].shape[0],),dtype=np.float32)
        for k in xrange(Rp[cl].shape[0]):
            P = Rp[cl][k,:,:]  # #{a}x#{C} performance matrix
            coords = np.unravel_index(np.argmax(P), P.shape)
            params[cl][k,0], params[cl][k,1] = a[coords[0]], C[coords[1]]

    clfs = [None] * class_labels.shape[1]
    for cl in xrange(class_labels.shape[1]):
        print "[Validation] Optimizing stacket classifiers %d/%d" % (cl + 1, class_labels.shape[1])

        D_tr = []  # training data for stacked learning (predicted outputs from single clfs)
        y_tr = []
        for (val_tr_inds, val_te_inds) in skf:
            # Remove negatives from test data
            # val_te_msk = np.ones(tr_inds.shape, dtype=np.bool)
            # val_te_msk[val_tr_inds] = False
            # negatives_msk = np.all(class_labels[tr_inds] <= 0, axis=1)
            # val_te_msk[negatives_msk] = False

            # Get the predictions of each and every kernel
            # X = np.zeros((len(val_te_inds), params[cl].shape[0]))
            X = np.zeros((len(val_te_inds), 2*params[cl].shape[0]))  # matrix of predictions
            for k,x in enumerate(kernels_tr):
                # Value of best parameters
                a_val, c_val = params[cl][k,0], params[cl][k,1]

                # Merge a kernel according to a_val
                K_tr_aux = (a_val*x[0]+(1-a_val)*x[1]) if isinstance(x,tuple) else x
                K_tr = utils.normalize(K_tr_aux)

                # Train using c_val as SVM-C parameter
                clf = _train_binary(K_tr[val_tr_inds,:][:,val_tr_inds], class_labels[tr_inds,cl][val_tr_inds], c=c_val)
                # X[:,k] = clf.decision_function(K_tr[val_te_inds,:][:,val_tr_inds])
                X[:,2*k] = clf.predict_proba(K_tr[val_te_inds,:][:,val_tr_inds])[:,0]
                X[:,2*k+1] = clf.decision_function(K_tr[val_te_inds,:][:,val_tr_inds])
            D_tr.append(X)
            y_tr.append(class_labels[tr_inds,cl][val_te_inds])

        D_tr = (np.vstack(D_tr))
        # D_tr = np.sign(D_tr)*np.sqrt(np.abs(D_tr))
        # D_tr = np.hstack([preprocessing.normalize(D_tr[:,::2], norm='l2'), preprocessing.normalize(D_tr[:,1::2], norm='l2')])
        y_tr = np.concatenate(y_tr)
        n = len(class_labels[tr_inds,cl])
        # LOOCV = cross_validation.KFold(n, n_folds=n)
        grid_scorer = make_scorer(average_precision_score, greater_is_better=True)
        LOOCV = cross_validation.StratifiedKFold(class_labels[tr_inds,cl], n_folds=3, shuffle=False, random_state=74)
        clfs[cl] = grid_search.GridSearchCV(svm.SVC(), svm_parameters, \
                                       n_jobs=20, cv=LOOCV, scoring=grid_scorer, verbose=False)
        clfs[cl].fit(D_tr,y_tr)
        clfs[cl].best_params_

    val_scores = [clf.best_score_ for clf in clfs]
    print np.mean(val_scores), np.std(val_scores)

    # quit()

    #
    # Test
    #

    # Train individual classifiers to use in test partition
    ind_clfs = [None] * class_labels.shape[1]
    F = np.zeros((class_labels.shape[1], len(kernels_tr)))
    for cl in xrange(class_labels.shape[1]):
        ind_clfs[cl] = [None] * len(kernels_tr)

        for k,x in enumerate(kernels_tr):
            a_val, c_val = params[cl][k,0],params[cl][k,1]
            K_tr_aux = (a_val*x[0]+(1-a_val)*x[1]) if isinstance(x,tuple) else x
            F[cl,k] = utils.argnormalize(K_tr_aux)
            K_tr = F[cl,k]*K_tr_aux
            ind_clfs[cl][k] = _train_binary(K_tr, class_labels[tr_inds,cl], c=c_val)

    # Use a mask to exclude negatives from test
    te_msk = np.ones((len(te_inds),), dtype=np.bool)
    negatives_msk = np.negative(np.any(class_labels[te_inds] > 0, axis=1))
    te_msk[negatives_msk] = False

    # Construct the stacked test data and predict
    acc_classes = []
    ap_classes = []
    for cl in xrange(class_labels.shape[1]):
        # X_te = np.zeros((len(te_inds), len(kernels_te)))
        X_te = np.zeros((len(te_inds), 2*len(kernels_te)))
        for k,x in enumerate(kernels_te):
            a_val, c_val = params[cl][k,0], params[cl][k,1]

            K_te = F[cl,k] * ((a_val*x[0]+(1-a_val)*x[1]) if isinstance(x,tuple) else x)
            # K_te = (a_val*x[0]+(1-a_val)*x[1]) if isinstance(x,tuple) else x
            # X_te[:,k] = ind_clfs[cl][k].decision_function(K_te)
            X_te[:,2*k] = ind_clfs[cl][k].predict_proba(K_te)[:,0]
            X_te[:,2*k+1] = ind_clfs[cl][k].decision_function(K_te)

        X_te = X_te[te_msk,:]
        # X_te = np.sign(X_te)*np.sqrt(np.abs(X_te))
        # X_te = np.hstack([preprocessing.normalize(X_te[te_msk,::2],norm='l2'), preprocessing.normalize(X_te[te_msk,1::2],norm='l2')])

        y_preds = clfs[cl].predict(X_te)
        acc = average_binary_accuracy(class_labels[te_inds,cl][te_msk], y_preds)
        ap = average_precision_score(class_labels[te_inds,cl][te_msk], y_preds)

        acc_classes.append(acc)
        ap_classes.append(ap)

    print acc_classes, np.mean(acc_classes)
    print ap_classes, np.mean(ap_classes)
    return dict(acc_classes=acc_classes, ap_classes=ap_classes)

def learning_based_fusion_classification2(input_kernels_tr, input_kernels_te, a, feat_types, class_labels, train_test_idx, C=[1], nl=1):
    '''

    :param kernels_tr:
    :param kernels_te:
    :param a: trade-off parameter controlling importance of root representation vs edges representation
    :param feat_types:
    :param class_labels:
    :param train_test_idx:
    :param C:
    :return:
    '''

    tr_inds, te_inds = train_test_idx[0], train_test_idx[1]
    # lb = LabelBinarizer(neg_label=-1, pos_label=1)

    class_ints = np.dot(class_labels, np.logspace(0, class_labels.shape[1]-1, class_labels.shape[1]))
    skf = cross_validation.StratifiedKFold(class_ints[tr_inds], n_folds=4, shuffle=False, random_state=74)

    # S = [None] * class_labels.shape[1]  # selected (best) params
    # p = [None] * class_labels.shape[1]  # performances
    # C = [(a, C) for k in xrange(class_labels.shape[1])]  # candidate values for params

    kernels_tr = []
    for feat_t in input_kernels_tr.keys():
        for k,v in input_kernels_tr[feat_t].iteritems():
            for x in v:
                if np.any(x != 0):
                    kernels_tr.append(x)
    kernels_te = []
    for feat_t in input_kernels_te.keys():
        for k,v in input_kernels_te[feat_t].iteritems():
            for x in v:
                if np.any(x != 0):
                    kernels_te.append(x)

    Rp = [None] * class_labels.shape[1]
    for cl in xrange(class_labels.shape[1]):
        print "[Validation] Optimizing weights and svm-C for class %d/%d" % (cl + 1, class_labels.shape[1])
        Rp[cl] = np.zeros((len(kernels_tr),len(a),len(C)), dtype=np.float32)

        for i, a_i in enumerate(a):
            for k, x in enumerate(kernels_tr):
                K_tr = utils.normalize(a_i*x[0]+(1-a_i)*x[1]) if isinstance(x,tuple) else utils.normalize(x)

                for j, c_j in enumerate(C):
                    for (val_tr_inds, _) in skf:
                        # test instances not indexed directly, but a mask is created excluding negative instances
                        val_te_msk = np.ones(tr_inds.shape, dtype=np.bool)
                        val_te_msk[val_tr_inds] = False
                        negatives_msk = np.all(class_labels[tr_inds] <= 0, axis=1)
                        val_te_msk[negatives_msk] = False

                        acc, ap = _train_and_classify_binary(
                            K_tr[val_tr_inds,:][:,val_tr_inds], K_tr[val_te_msk,:][:,val_tr_inds], \
                            class_labels[tr_inds,cl][val_tr_inds], class_labels[tr_inds,cl][val_te_msk], \
                            c=c_j)

                        # TODO: decide what it is
                        Rp[cl][k,i,j] += acc / skf.n_folds
                        # Rp[cl][k,i,j] += ap/skf.n_folds

    params = [None] * class_labels.shape[1]
    perfs = [None] * class_labels.shape[1]
    for cl in xrange(class_labels.shape[1]):
        # if params[cl] is None:
        params[cl] = np.zeros((Rp[cl].shape[0],2),dtype=np.float32)
        perfs[cl]  = np.zeros((Rp[cl].shape[0],),dtype=np.float32)
        for k in xrange(Rp[cl].shape[0]):
            P = Rp[cl][k,:,:]  # #{a}x#{C} performance matrix
            coords = np.unravel_index(np.argmax(P), P.shape)
            params[cl][k,0], params[cl][k,1] = a[coords[0]], C[coords[1]]

    XX = []
    yy_tr = []
    for cl in xrange(class_labels.shape[1]):
        print "[Validation] Optimizing stacket classifiers %d/%d" % (cl + 1, class_labels.shape[1])

        D_tr = []  # training data for stacked learning (predicted outputs from single clfs)
        y_tr = []
        for (val_tr_inds, val_te_inds) in skf:
            # Remove negatives from test data
            # val_te_msk = np.ones(tr_inds.shape, dtype=np.bool)
            # val_te_msk[val_tr_inds] = False
            # negatives_msk = np.all(class_labels[tr_inds] <= 0, axis=1)
            # val_te_msk[negatives_msk] = False

            # Get the predictions of each and every kernel
            X = np.zeros((len(val_te_inds), params[cl].shape[0]))  # matrix of predictions
            # X = np.zeros((len(val_te_inds), 2*params[cl].shape[0]))  # matrix of predictions
            for k,x in enumerate(kernels_tr):
                # Value of best parameters
                a_val, c_val = params[cl][k,0], params[cl][k,1]

                # Merge a kernel according to a_val
                K_tr_aux = (a_val*x[0]+(1-a_val)*x[1]) if isinstance(x,tuple) else x
                K_tr = utils.normalize(K_tr_aux)

                # Train using c_val as SVM-C parameter
                clf = _train_binary(K_tr[val_tr_inds,:][:,val_tr_inds], class_labels[tr_inds,cl][val_tr_inds], c=c_val)
                X[:,k] = clf.decision_function(K_tr[val_te_inds,:][:,val_tr_inds])
                # X[:,2*k] = clf.predict_proba(K_tr[val_te_inds,:][:,val_tr_inds])[:,0]
                # X[:,2*k+1] = clf.decision_function(K_tr[val_te_inds,:][:,val_tr_inds])
            D_tr.append(X)
            y_tr.append(class_labels[tr_inds,cl][val_te_inds])

        XX.append(np.vstack(D_tr))
        yy_tr.append(np.concatenate(y_tr))

    XX = np.hstack(XX)

    clfs = [None] * class_labels.shape[1]
    for cl in xrange(class_labels.shape[1]):
        n = len(class_labels[tr_inds,cl])
        # LOOCV = cross_validation.KFold(n, n_folds=n)
        grid_scorer = make_scorer(average_binary_accuracy, greater_is_better=True)
        LOOCV = cross_validation.StratifiedKFold(class_labels[tr_inds,cl], n_folds=3, shuffle=False, random_state=74)
        clfs[cl] = grid_search.GridSearchCV(svm.SVC(), svm_parameters, \
                                       n_jobs=20, cv=LOOCV, scoring=grid_scorer, verbose=False)
        clfs[cl].fit(XX,yy_tr[cl])
        clfs[cl].best_params_

    val_scores = [clf.best_score_ for clf in clfs]
    print np.mean(val_scores), np.std(val_scores)

    # quit()

    #
    # Test
    #

    # Train individual classifiers to use in test partition
    ind_clfs = [None] * class_labels.shape[1]
    F = np.zeros((class_labels.shape[1], len(kernels_tr)))
    for cl in xrange(class_labels.shape[1]):
        ind_clfs[cl] = [None] * len(kernels_tr)

        for k,x in enumerate(kernels_tr):
            a_val, c_val = params[cl][k,0],params[cl][k,1]
            K_tr_aux = (a_val*x[0]+(1-a_val)*x[1]) if isinstance(x,tuple) else x
            F[cl,k] = utils.argnormalize(K_tr_aux)
            K_tr = F[cl,k]*K_tr_aux
            ind_clfs[cl][k] = _train_binary(K_tr, class_labels[tr_inds,cl], c=c_val)

    # Use a mask to exclude negatives from test
    te_msk = np.ones((len(te_inds),), dtype=np.bool)
    negatives_msk = np.negative(np.any(class_labels[te_inds] > 0, axis=1))
    te_msk[negatives_msk] = False

    # Construct the stacked test data and predict
    acc_classes = []
    ap_classes = []
    for cl in xrange(class_labels.shape[1]):
        X_te = np.zeros((len(te_inds), len(kernels_te)))
        # X_te = np.zeros((len(te_inds), 2*len(kernels_te)))
        for k,x in enumerate(kernels_te):
            a_val, c_val = params[cl][k,0], params[cl][k,1]

            K_te = F[cl,k] * ((a_val*x[0]+(1-a_val)*x[1]) if isinstance(x,tuple) else x)
            # K_te = (a_val*x[0]+(1-a_val)*x[1]) if isinstance(x,tuple) else x
            X_te[:,k] = ind_clfs[cl][k].decision_function(K_te)
            # X_te[:,2*k] = ind_clfs[cl][k].predict_proba(K_te)[:,0]
            # X_te[:,2*k+1] = ind_clfs[cl][k].decision_function(K_te)

        X_te = X_te[te_msk,:]
        # X_te = np.sign(X_te)*np.sqrt(np.abs(X_te))
        # X_te = np.hstack([preprocessing.normalize(X_te[te_msk,::2],norm='l2'), preprocessing.normalize(X_te[te_msk,1::2],norm='l2')])

        y_preds = clfs[cl].predict(X_te)
        acc = average_binary_accuracy(class_labels[te_inds,cl][te_msk], y_preds)

        acc_classes.append(acc)

    print acc_classes, np.mean(acc_classes)

    return dict(acc_classes=acc_classes, ap_classes=ap_classes)


def _train_binary(K_tr, train_labels, c=1.0):
    # Train
    clf = svm.SVC(kernel='precomputed', class_weight='balanced', C=c, max_iter=-1, tol=1e-3, probability=False, verbose=False)
    clf.fit(K_tr, train_labels)

    return clf


def _train_and_classify_binary(K_tr, K_te, train_labels, test_labels, c=1.0):
    clf = _train_binary(K_tr, train_labels, c=c)

    # Predict
    test_scores = clf.decision_function(K_te)
    test_preds = clf.predict(K_te)

    # Compute accuracy and average precision
    acc = average_binary_accuracy(test_labels, test_preds)

    # TODO: decide what is it
    # ap = average_precision_score(test_labels, test_preds)
    ap = average_precision_score(test_labels, test_scores, average='weighted')
    # precision_recall_fscore_support

    return acc, ap

def average_binary_accuracy(test_labels, test_preds):
    # test_preds = test_scores > 0
    cmp = test_labels == test_preds
    neg_acc = float(np.sum(cmp[test_labels <= 0]))/len(test_labels[test_labels <= 0])
    pos_acc = float(np.sum(cmp[test_labels > 0]))/len(test_labels[test_labels > 0])
    acc = (pos_acc + neg_acc) / 2.0

    return acc


def print_results(results):
    '''
    Print in a given format.
    :param results: array of results which is a structure {no folds x #{acc,ap} x classes}.
    :return:
    '''
    accs = np.zeros((len(results),), dtype=np.float32)
    maps = accs.copy()
    for k in xrange(len(results)):
        accs[k] = np.mean(results[k]['acc_classes'])
        maps[k] = np.mean(results[k]['ap_classes'])

    # Print the results

    for k in xrange(len(results)):
        print("#Fold, Class_name, ACC, mAP")
        print("---------------------------")
        for i in xrange(len(results[k]['acc_classes'])):
            print("%d, %s, %.1f%%, %.1f%%" % (k+1, i, results[k]['acc_classes'][i]*100, results[k]['ap_classes'][i]*100))
        print("%d, ALL classes, %.1f%%, %.1f%%" % (k+1, accs[k]*100, maps[k]*100))
        print

    print("TOTAL, All folds, ACC: %.1f%%, mAP: %.1f%%" % (np.mean(accs)*100, np.mean(maps)*100))