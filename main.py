'''Framework for action/activity recognition on videos (computer vision research work)

LICENSE: BSD

Copyrights: Albert Clap\'{e}s, 2015

'''

import numpy as np
from scipy.io import loadmat, savemat
from os.path import isfile, exists, join, splitext
from os import listdir
from os import makedirs
from sklearn import svm
from sklearn.metrics import accuracy_score, average_precision_score

import tracklet_extraction, tracklet_clustering, tracklet_representation
from videodarwin import darwin, rootSIFT, normalizeL2

# change depending on the computer
INTERNAL_PARAMETERS = dict(
    home_path = '/Volumes/MacintoshHD/Users/aclapes/',
    datasets_path = 'Datasets/',
    dataset_name = 'HighFive',  #'HighFive',
    data_path = 'Data/darwintree/'
)

def get_global_config():
    '''

    :return:
    '''
    parent_path = INTERNAL_PARAMETERS['home_path'] + INTERNAL_PARAMETERS['data_path'] + INTERNAL_PARAMETERS['dataset_name'] + '/'
    tracklets_path = parent_path + 'tracklets/'
    clusters_path = parent_path + 'clusters/'
    intermediates_path = parent_path + 'intermediates/'
    feats_path = parent_path + 'feats/'
    darwins_path = parent_path + 'darwins/'

    return tracklets_path, clusters_path, intermediates_path, feats_path, darwins_path


def get_hollywood2_config():
    '''
    Hard-codes some paths and configuration values.
    :return:
    '''
    parent_path = INTERNAL_PARAMETERS['home_path'] + INTERNAL_PARAMETERS['datasets_path'] + INTERNAL_PARAMETERS['dataset_name'] + '/'
    videos_dir = parent_path + 'AVIClips/'
    split_file_path = parent_path + 'train_test_split.mat'

    tmp_dict = loadmat(split_file_path)
    fullvideonames = np.array([videos_dir + str(element[0][0]) for element in tmp_dict['fnames']])
    videonames = np.array([str(element[0][0]) for element in tmp_dict['fnames']])

    class_labels = tmp_dict['labels2']

    cur_train_indx = np.squeeze(tmp_dict['cur_train_indx'][0][0])
    cur_test_indx = np.squeeze(tmp_dict['cur_test_indx'][0][0])
    train_test_indx = (cur_train_indx-1, cur_test_indx-1)  # was MATLAB's indexing

    action_names = ['AnswerPhone','DriveCar','Eat','FightPerson','GetOutCar', \
                    'HandShake','HugPerson','Kiss','Run','SitDown','SitUp','StandUp']

    return fullvideonames, videonames, class_labels, action_names, train_test_indx

def get_highfive_config():
    '''
    Hard-codes some paths and configuration values.
    :return:
    '''
    parent_path = INTERNAL_PARAMETERS['home_path'] + INTERNAL_PARAMETERS['datasets_path'] + INTERNAL_PARAMETERS['dataset_name'] + '/'
    videos_dir = parent_path + 'tv_human_interactions_videos/'
    # splits_dir = parent_path + 'tv_human_interaction_annotations/'  # TODO: think if i'll need to use for test

    # Half train/test partition got from:
    # http://www.robots.ox.ac.uk/~alonso/tv_human_interactions.html
    train_test = dict(
        train_inds = dict(
            handShake = [2,14,15,16,18,19,20,21,24,25,26,27,28,32,40,41,42,43,44,45,46,47,48,49,50],
            highFive = [1,6,7,8,9,10,11,12,13,23,24,25,27,28,29,30,31,32,33,34,35,44,45,47,48],
            hug = [2,3,4,11,12,15,16,17,18,20,21,27,29,30,31,32,33,34,35,36,42,44,46,49,50],
            kiss = [1,7,8,9,10,11,12,13,14,16,17,18,22,23,24,26,29,31,35,36,38,39,40,41,42],
            negative = np.linspace(1,50,50,dtype=np.int32)
        ),
        test_inds = dict(
            handShake = [1,3,4,5,6,7,8,9,10,11,12,13,17,22,23,29,30,31,33,34,35,36,37,38,39],
            highFive = [2,3,4,5,14,15,16,17,18,19,20,21,22,26,36,37,38,39,40,41,42,43,46,49,50],
            hug = [1,5,6,7,8,9,10,13,14,19,22,23,24,25,26,28,37,38,39,40,41,43,45,47,48],
            kiss = [2,3,4,5,6,15,19,20,21,25,27,28,30,32,33,34,37,43,44,45,46,47,48,49,50],
            negative = np.linspace(51,100,50,dtype=np.int32)
        )
    )

    action_names = sorted(train_test['train_inds'].keys(), key=lambda x: x)

    videonames = []
    int_inds = []
    train_test_indx = []
    for partition_name in train_test:
        n = len(videonames)
        for i, name in enumerate(action_names):
            videonames += [name + '_' + str(j).zfill(4)
                           for j in train_test[partition_name][name]]
            int_inds += [i]*len(train_test[partition_name][name])

        train_test_indx.append(np.linspace(n, len(videonames)-1, len(videonames)-n, dtype=np.int32))

    fullvideonames = [parent_path + 'tv_human_interactions_videos/' + videoname for videoname in videonames]

    # create a matrix #{instances}x#{classes}, where entries are all "-1" except for 1s in corresponding class columns
    class_labels = (-1) * np.ones((len(int_inds),len(action_names)),dtype=np.int32)
    for i in xrange(len(action_names)):
        class_labels[np.array(int_inds)==i,i] = 1

    return fullvideonames, videonames, class_labels, action_names, tuple(train_test_indx)


def get_videodarwin_representation(feat_t, feat_channel, darwin_t, feats_path, videonames, st, num_videos, darwins_path):
    """
    Computes the videodarwin representation for a (sub)set of videos,
    given a feature type.
    :param feat_t:
    :param feat_channel:
    :param darwin_t:
    :param feats_dir:
    :param videonames:
    :param st:
    :param num_videos:
    :param darwins_dir:
    :return:
    """

    c_param = 1
    repr_folder_name = 'representation_' + feat_t + '_' + feat_channel + '_' + darwin_t

    # create output directory structure if needed
    if not exists(darwins_path + repr_folder_name):
        if not exists(darwins_path):
            makedirs(darwins_path)
        makedirs(darwins_path + repr_folder_name)

    U = []
    total = len(videonames)
    for i in range(st,min(st+num_videos,total)):
        darwin_file_path = darwins_path + repr_folder_name + '/' + str(i) + '-Cval' + str(c_param) + '.mat'
        if isfile(darwin_file_path):
            u = loadmat(darwin_file_path)['u'][0]
        else:
            feats_file_path = feats_path + feat_channel + '/' + videonames[i] + '-' + feat_t + '.mat'
            histv = loadmat(feats_file_path)['histv']
            u = darwin(histv, c_svm_param=c_param)
            savemat(darwin_file_path, {'u':u})

        print("%s -> OK" % darwin_file_path)
        U.append(u)

    return np.matrix(U)

def train_and_classify(K_tr, K_te, train_labels, test_labels):
    # one_to_n = np.linspace(1,K_tr.shape[0],K_tr.shape[0])
    # K_tr = np.hstack((one_to_n[:,np.newaxis], K_tr))
    # one_to_n = np.linspace(1,K_te.shape[0],K_te.shape[0])
    # K_te = np.hstack((one_to_n[:,np.newaxis], K_te))

    # Train
    c_param = 100
    clf = svm.SVC(kernel='precomputed', C=c_param, max_iter=-1, tol=1e-3)
    clf.fit(K_tr, train_labels)

    # Predict
    test_preds = clf.predict(K_te)

    # Compute accuracy and average precision
    acc = accuracy_score(test_labels, test_preds)
    ap = average_precision_score(test_labels, test_preds)

    return acc, ap

if __name__ == "__main__":
    # load configuration (computation node-dependent)
    tracklets_path, clusters_path, intermediates_path, feats_path, darwins_path = get_global_config()
    # load dataset configuration (if prepared)
    if INTERNAL_PARAMETERS['dataset_name'] == 'Hollywood2':
        fullvideonames, videonames, class_labels, action_names, train_test_indx = get_hollywood2_config()
    elif INTERNAL_PARAMETERS['dataset_name'] == 'HighFive':
        fullvideonames, videonames, class_labels, action_names, train_test_indx = get_highfive_config()

    # Change some values if wanna compute a subset of data (instances or classes)
    INSTANCE_ST = 0
    INSTANCE_TOTAL = 5  #len(videonames)

    tracklet_extraction.extract(fullvideonames, videonames, INSTANCE_ST, INSTANCE_TOTAL, tracklets_path)
    # tracklet_clustering.cluster(tracklets_path, videonames, INSTANCE_ST, INSTANCE_TOTAL, clusters_path)

    train_indx, test_indx = train_test_indx

    tracklet_representation.train_reduction_maps_and_gmms(tracklets_path, \
                                                          videonames[train_indx], INSTANCE_ST, INSTANCE_TOTAL, \
                                                          intermediates_path)
    tracklet_representation.compute_fv_representations(tracklets_path, clusters_path, intermediates_path, \
                                                       videonames, INSTANCE_ST, INSTANCE_TOTAL, \
                                                       feats_path)
    quit()

    # Get the videodarwin representation
    channels = ['trj']  # channels = ['trj', 'hog', 'hof', 'mbh']

    U = dict()  # darwin of different channels
    for ch in channels:
        U_ch = get_videodarwin_representation('fv', ch, 'lin', feats_path, videonames, INSTANCE_ST, INSTANCE_TOTAL, darwins_path)
        U[ch] = U_ch

    # Classification
    CLASS_ST = 0
    CLASS_TOTAL = len(INTERNAL_PARAMETERS['action_names'])  # want to use a subset of classes?

    class_labels = class_labels[:,CLASS_ST:CLASS_ST+CLASS_TOTAL]
    indices = np.any(class_labels == 1, axis=1)
    n_train = len(train_indx)
    n_test = len(test_indx)
    train_indx = train_indx[indices[0:n_train]]
    test_indx = test_indx[indices[n_train:(n_train+n_test)]]

    # Normalization of different channels
    for ch in channels:
        U[ch] = normalizeL2(U[ch])

    # Kernel computation
    kernels_tr = []
    kernels_te = []
    for i, ch in enumerate(channels):
        X_tr_ch = U[ch][train_indx,:]
        X_te_ch = U[ch][test_indx,:]
        kernels_tr.append( np.dot(X_tr_ch, X_tr_ch.T) )
        kernels_te.append( np.dot(X_te_ch, X_tr_ch.T) )

    # Assign weights to channels
    if not 'weights' in locals(): # if not specified a priori (when channels' specification)
        weights = [1.0/len(channels) for i in channels]

    # Perform the classification
    acc_classes = []
    ap_classes = []
    for cl in range(CLASS_ST,CLASS_ST+CLASS_TOTAL):
        train_labels = class_labels[train_indx, cl]
        test_labels = class_labels[test_indx, cl]
        # Weight each channel accordingly
        K_tr = weights[0] * kernels_tr[0]
        K_te = weights[0] * kernels_te[0]
        for i in range(1,len(channels)):
            K_tr += weights[i] * kernels_tr[i]
            K_te += weights[i] * kernels_te[i]
        # Get class results
        acc, ap = train_and_classify(K_tr, K_te, train_labels, test_labels)
        acc_classes.append(acc)
        ap_classes.append(ap)

    # Get global results
    print("ACTION_NAME AC mAP")
    for i in xrange(class_labels.shape[1]):
        print("%s %.2f %.2f" % (INTERNAL_PARAMETERS['action_names'][i], acc_classes[i], ap_classes[i]))
    print("%.2f %.2f" % (np.mean(acc_classes), np.mean(ap_classes)))