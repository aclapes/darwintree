__author__ = 'aclapes'

import numpy as np
import sys
from sklearn import svm
from sklearn.metrics import average_precision_score
import utils
from copy import deepcopy
from sklearn import preprocessing, cross_validation, grid_search
from sklearn.ensemble import RandomForestClassifier
import itertools

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

INTERNAL_PARAMETERS = dict(
    weights = None
)

def classify(input_kernels, class_labels, traintest_parts, params, feat_types, strategy='kernel_fusion',
             C=[1], opt_criterion='acc', random_state=42, verbose=False):
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

    combs = [c for c in itertools.product(*params)]

    results = [None] * len(traintest_parts)
    for k, part in enumerate(traintest_parts):
        train_inds, test_inds = np.where(np.array(part) <= 0)[0], np.where(np.array(part) > 0)[0]

        kernels_train = input_kernels[k]['train']
        kernels_test  = input_kernels[k]['test']
        if strategy == 'kernel_fusion':
            results[k] = kernel_fusion_classification(kernels_train, kernels_test, combs, feat_types, class_labels, (train_inds, test_inds), \
                                                      C=C, opt_criterion=opt_criterion, random_state=random_state, verbose=verbose)
        elif strategy == 'learning_based_fusion':
            results[k] = learning_based_fusion_classification(kernels_train, kernels_test, feat_types, class_labels, (train_inds, test_inds), \
                                                              C=C, opt_criterion=opt_criterion, random_state=random_state, verbose=verbose)
        else:
            sys.stderr.write('Not a valid classification method.')
            sys.stderr.flush()

    return results



# ==============================================================================
# Helper functions
# ==============================================================================

def kernel_fusion_classification(input_kernels_tr, input_kernels_te, a, feat_types, class_labels, train_test_idx, \
                                 C=[1], opt_criterion='acc', random_state=42, verbose=False):
    '''

    :param input_kernels_tr:
    :param input_kernels_te:
    :param a:
    :param feat_types:
    :param class_labels:
    :param train_test_idx:
    :param C:
    :param opt_criterion:
    :return:
    '''

    # Assign weights to channels
    feat_weights = INTERNAL_PARAMETERS['weights']
    if feat_weights is None: # if not specified a priori (when channels' specification)
        feat_weights = {feat_t : 1.0/len(input_kernels_tr) for feat_t in input_kernels_tr.keys()}

    tr_inds, te_inds = train_test_idx[0], train_test_idx[1]
    # lb = LabelBinarizer(neg_label=-1, pos_label=1)

    class_ints = np.dot(class_labels, np.logspace(0, class_labels.shape[1]-1, class_labels.shape[1]))
    skf = cross_validation.StratifiedKFold(class_ints[tr_inds], n_folds=4)

    Rval = np.zeros((class_labels.shape[1], len(a), len(C)), dtype=np.float32)
    for k in xrange(class_labels.shape[1]):
        if verbose:
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
                    kernels_tr[feat_t]['nodes'] = [utils.normalize(a_i[1][j]*x[0]+(1-a_i[1][j])*x[1] if len(x)==2 else x[0])
                                                   for j,x in enumerate(kernels_tr[feat_t]['nodes'])]

                    kernels_tr[feat_t] = list(a_i[2]*np.array(kernels_tr[feat_t]['root']) + (1-a_i[2])*np.array(kernels_tr[feat_t]['nodes']))

            K_tr = None
            # Weight each channel accordingly
            for j, feat_t in enumerate(kernels_tr.keys()):
                if K_tr is None:
                    K_tr = np.zeros(kernels_tr[feat_t].shape if isinstance(kernels_tr[feat_t],np.ndarray) else kernels_tr[feat_t][0].shape, dtype=np.float32)
                K_tr += a_i[3][j] * utils.sum_of_arrays(kernels_tr[feat_t], a_i[0])

            for j, c_j in enumerate(C):
                # print l, str(i+1) + '/' + str(len(C[k][0])), str(j+1) + '/' + str(len(C[k][1]))
                Rval[k,i,j] = 0
                for (val_tr_inds, val_te_inds) in skf:
                    # test instances not indexed directly, but a mask is created excluding negative instances
                    val_te_msk = np.ones(tr_inds.shape, dtype=np.bool)
                    val_te_msk[val_tr_inds] = False
                    negatives_msk = np.negative(np.any(class_labels[tr_inds] > 0, axis=1))
                    val_te_msk[negatives_msk] = False

                    acc_tmp, ap_tmp, _ = _train_and_classify_binary(
                        K_tr[val_tr_inds,:][:,val_tr_inds], K_tr[val_te_msk,:][:,val_tr_inds], \
                        class_labels[tr_inds,k][val_tr_inds], class_labels[tr_inds,k][val_te_msk], \
                        probability=True, c=c_j, random_state=random_state)

                    if str.lower(opt_criterion) == 'map':
                        Rval[k,i,j] += ap_tmp/skf.n_folds # if acc_tmp > 0.50 else 0
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
        if verbose:
            print best_res, "\t",
        results_val.append(best_res)
    if verbose:
        print("Validation best %s : %2.2f" %(opt_criterion, np.mean(results_val)*100.0))


    acc_classes = []
    ap_classes = []
    for k in xrange(class_labels.shape[1]):
        i,j = np.unravel_index(np.argmax(Rval[k,:,:]), Rval[k,:,:].shape)
        a_best, c_best = a[i], C[j]
        if verbose:
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
                    kernels_te[feat_t]['root'][i]      = pr * (kernels_te[feat_t]['root'][i][0] if np.sum(kernels_te[feat_t]['root'][i][0]) > 0
                                                               else kernels_te[feat_t]['nodes'][i][0])

                    xn_tr, xn_te = kernels_tr[feat_t]['nodes'][i], kernels_te[feat_t]['nodes'][i]
                    kernels_tr[feat_t]['nodes'][i], pn = utils.normalization(a_best[1][i]*xn_tr[0]+(1-a_best[1][i])*xn_tr[1] if len(xn_tr)==2 else xn_tr[0])
                    kernels_te[feat_t]['nodes'][i]     = pn * (a_best[1][i]*xn_te[0]+(1-a_best[1][i])*xn_te[1] if len(xn_te)==2 else xn_te[0])

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

        acc, ap, _ = _train_and_classify_binary(K_tr, K_te[te_msk], class_labels[tr_inds,k], class_labels[te_inds,k][te_msk],
                                                probability=True, c=c_best, random_state=random_state)

        acc_classes.append(acc)
        ap_classes.append(ap)

    return dict(acc_classes=acc_classes, ap_classes=ap_classes)


def learning_based_fusion_classification(input_kernels_tr, input_kernels_te, feat_types, class_labels, train_test_idx, \
                                         C=[1], opt_criterion='acc', random_state=42, verbose=False):
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

    # prepare training/test partition
    tr_inds, te_inds = train_test_idx[0], train_test_idx[1]

    class_ints = np.dot(class_labels, np.logspace(0, class_labels.shape[1]-1, class_labels.shape[1]))
    skf = cross_validation.StratifiedKFold(class_ints[tr_inds], n_folds=4, shuffle=False, random_state=random_state)

    # prepare data structures
    kernels_tr = []
    for feat_t in input_kernels_tr.keys():
        for tree_t, mids in input_kernels_tr[feat_t].iteritems():  # X is a list of mid level representations (ex: ATEP-FV, ATNBEP)
            for X in mids:  # X is a kernel tuple. ATEPs are of size 1, and ATNBEP of size 2
                for x in X:
                    if np.sum(np.abs(x)) > 1e-7:
                        kernels_tr.append(x)
    kernels_te = []
    for feat_t in input_kernels_te.keys():
        for tree_t, mids in input_kernels_te[feat_t].iteritems():
            for X in mids:
                for x in X:
                    if np.sum(np.abs(x)) > 1e-7:
                        kernels_te.append(x)

    Rp = np.zeros((class_labels.shape[1],len(kernels_tr),len(C)), dtype=np.float32)
    for cl in xrange(class_labels.shape[1]):
        if verbose:
            print "[learning_based_fusion_classification] Optimizing svm-C for kernel(s) of class %d/%d" % (cl + 1, class_labels.shape[1])

        for k, x in enumerate(kernels_tr):
            K_tr = utils.normalize(x)

            for i, c_i in enumerate(C):
                for (val_tr_inds, val_te_inds) in skf:
                    positives_msk = np.any(class_labels[tr_inds][val_te_inds] > 0, axis=1)

                    K_val_tr,p = utils.normalization(K_tr[val_tr_inds,:][:,val_tr_inds])
                    K_val_te = p * K_tr[val_te_inds[positives_msk],:][:,val_tr_inds]
                    # test instances not indexed directly, but a mask is created excluding negative instances

                    acc, ap, _ = _train_and_classify_binary(
                        K_val_tr, K_val_te,
                        class_labels[tr_inds,cl][val_tr_inds], class_labels[tr_inds,cl][val_te_inds[positives_msk]],
                        probability=True, c=c_i)

                    Rp[cl,k,i] += (ap / skf.n_folds) if str.lower(opt_criterion) == 'map' else (acc / skf.n_folds)

    perfs = np.max(Rp, axis=2)  # axis=2 is C
    params = np.array(C)[np.argmax(Rp, axis=2)]

    clfs = [None] * class_labels.shape[1]
    best_perfs = [None] * class_labels.shape[1]
    for cl in xrange(class_labels.shape[1]):
        if verbose:
            print "[learning_based_fusion_classification] Optimizing stacket classifiers %d/%d" % (cl + 1, class_labels.shape[1])

        D_tr = []  # training data for stacked learning (predicted outputs from single clfs)
        y_tr = []
        for (val_tr_inds, val_te_inds) in skf:
            # Get the predictions of each and every kernel
            X = np.zeros((len(val_te_inds), params[cl].shape[0]))
            # X = np.zeros((len(val_te_inds), 2*params[cl].shape[0]))  # matrix of predictions

            for k,x in enumerate(kernels_tr):
                # Merge a kernel according to a_val
                K_tr = utils.normalize(x)

                # Train using c_val as SVM-C parameter
                clf = _train_binary(K_tr[val_tr_inds,:][:,val_tr_inds], class_labels[tr_inds,cl][val_tr_inds], probability=True, c=params[cl,k])
                X[:,k] = clf.predict_proba(K_tr[val_te_inds,:][:,val_tr_inds])[:,1]
            D_tr.append(X)
            y_tr.append(class_labels[tr_inds,cl][val_te_inds])

        D_tr = (np.vstack(D_tr))
        y_tr = np.concatenate(y_tr)

        skf_in = cross_validation.StratifiedKFold(y_tr > 0, n_folds=2, shuffle=False, random_state=random_state)
        C_in = [3**(-5), 3**(-4), 3**(-3), 3**(-2), 3**(-1), 3**0, 3**1, 3**2, 3**3, 3**4, 3**5, 3**6, 3**7]
        perfs = np.zeros((len(C_in),), dtype=np.float32)
        for i, c_i in enumerate(C_in):
            for j, (val_tr_inds, val_te_inds) in enumerate(skf_in):
                clf = svm.SVC(kernel='linear', class_weight='balanced', C=c_i,
                              max_iter=-1, tol=1e-3, probability=True, random_state=random_state, verbose=False)
                clf.fit(D_tr[val_tr_inds,:], y_tr[val_tr_inds])

                positives_msk = np.any(class_labels[tr_inds[val_te_inds],:], axis=1)
                if opt_criterion == 'map':
                    y_scores = clf.predict_proba(D_tr[val_te_inds[positives_msk],:])
                    perfs[i] += average_precision_score(y_tr[val_te_inds[positives_msk]], y_scores[:,1], average='weighted') / skf_in.n_folds
                else:
                    y_preds = clf.predict(D_tr[val_te_inds[positives_msk],:])
                    perfs[i] += average_binary_accuracy(y_tr[val_te_inds[positives_msk]], y_preds) / skf_in.n_folds

        clfs[cl] = svm.SVC(kernel='linear', class_weight='balanced', C=C_in[np.argmax(perfs)],
                           max_iter=-1, tol=1e-3, probability=True, random_state=random_state, verbose=False)
        clfs[cl].fit(D_tr, y_tr)
        best_perfs[cl] = np.max(perfs)

    if verbose:
        print np.mean(best_perfs), np.std(best_perfs)
        print 'all trained'


    #
    # Test
    #

    # Use a mask to exclude negatives from test
    positives_msk = np.any(class_labels[te_inds,:] > 0, axis=1)

    # Construct the stacked test data and predict
    acc_classes = []
    ap_classes = []
    for cl in xrange(class_labels.shape[1]):
        X_te = np.zeros((len(te_inds), len(kernels_te)))
        # X_te = np.zeros((len(te_inds), 2*len(kernels_te)))
        for k,x in enumerate(kernels_te):
            K_tr, p = utils.normalization(kernels_tr[k])

            clf = _train_binary(K_tr, class_labels[tr_inds,cl], probability=True, c=params[cl,k])

            K_te = p * x
            X_te[:,k] = clf.predict_proba(K_te)[:,1]

            # X_te[:,2*k] = ind_clfs[cl][k].predict_proba(K_te)[:,0]
            # X_te[:,2*k+1] = ind_clfs[cl][k].decision_function(K_te)

        X_te = X_te[positives_msk,:]
        y_preds = clfs[cl].predict(X_te)
        acc = average_binary_accuracy(class_labels[te_inds,cl][positives_msk], y_preds)

        y_scores = clfs[cl].predict_proba(X_te)
        ap = average_precision_score(class_labels[te_inds,cl][positives_msk], y_scores[:,1], average='weighted') #clfs[cl].decision_function(X_te))

        acc_classes.append(acc)
        ap_classes.append(ap)

    if verbose:
        print acc_classes, np.mean(acc_classes)
        print ap_classes, np.mean(ap_classes)
    return dict(acc_classes=acc_classes, ap_classes=ap_classes)


def _train_binary(K_tr, train_labels, probability=False, c=1.0, random_state=42):
    # Train
    clf = svm.SVC(kernel='precomputed', class_weight='balanced', C=c, max_iter=-1, tol=1e-3,
                  probability=probability, random_state=random_state, verbose=False)
    clf.fit(K_tr, train_labels)

    return clf


def _train_and_classify_binary(K_tr, K_te, train_labels, test_labels, probability=False, c=1.0, random_state=42):
    clf = _train_binary(K_tr, train_labels, probability=probability, random_state=random_state, c=c)

    # Compute accuracy and average precision
    test_preds = clf.predict(K_te)
    acc = average_binary_accuracy(test_labels, test_preds)

    test_scores = clf.predict_proba(K_te)
    ap = average_precision_score(test_labels, test_scores[:,1], average='weighted')

    return acc, ap, test_preds

def average_binary_accuracy(test_labels, test_preds):
    # test_preds = test_scores > 0
    cmp = test_labels == test_preds
    neg_acc = float(np.sum(cmp[test_labels <= 0]))/len(test_labels[test_labels <= 0])
    pos_acc = float(np.sum(cmp[test_labels > 0]))/len(test_labels[test_labels > 0])
    acc = (pos_acc + neg_acc) / 2.0

    return acc

def print_results(results, summarize=False, opt_criterion='acc'):
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

    if not summarize:
        for k in xrange(len(results)):
            print("#Fold, Class_name, ACC, mAP")
            print("---------------------------")
            for i in xrange(len(results[k]['acc_classes'])):
                print("%d, %s, " % (k+1, i)),
                if str.lower(opt_criterion) == 'acc':
                    print("%.1f%%" % (results[k]['acc_classes'][i]*100))
                elif str.lower(opt_criterion) == 'map':
                    print("%.1f%%" % (results[k]['ap_classes'][i]*100))

            print("%d, ALL classes, " % (k+1)),
            if str.lower(opt_criterion) == 'acc':
                print('%.1f%%' % (accs[k]*100))
            elif str.lower(opt_criterion) == 'map':
                print('%.1f%%' % (maps[k]*100))

    print("TOTAL, All folds, "),
    if str.lower(opt_criterion) == 'acc':
        print("ACC: %.1f%%" % (np.mean(accs)*100))
    elif str.lower(opt_criterion) == 'map':
        print("mAP: %.1f%%" % (np.mean(maps)*100))


