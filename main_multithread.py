#!/Users/Shared/anaconda/bin/python

'''Framework for action/activity recognition on videos (computer vision research work)

LICENSE: BSD

Copyrights: Albert Clap\'{e}s, 2015

'''

import numpy as np
from os.path import isdir, splitext, basename, join
from os import listdir, makedirs
import time
import copy
import utils
import itertools

from scipy.io import loadmat

import tracklet_extraction, tracklet_clustering, tracklet_representation, kernels, classification

import xmltodict


# ==============================================================================
# Configuration functions
# ==============================================================================

def load_XML_config(filepath):
    """
    Read the configuration from a .xml file in disk.
    This sets some paths and the features to use.
    :param filepath:
    :return:
    """
    config_dict = dict()

    with open(filepath) as fd:
        xml = xmltodict.parse(fd.read())

    for path in xml['configuration']['path']:
        config_dict[path['@key']] = path['#text'].encode('utf-8')

    if not isinstance(xml['configuration']['option'], list):
        option = xml['configuration']['option']
        config_dict[option['@key']] = option['#text'].encode('utf-8')
    else:
        for option in xml['configuration']['option']:
            config_dict[option['@key']] = option['#text'].encode('utf-8')

    if 'num_threads' in config_dict:
        config_dict['num_threads'] = int(config_dict['num_threads'])

    features_list = xml['configuration']['features_list']

    if type(features_list['item']) is unicode:
        config_dict.setdefault('features_list',[]).append(features_list['item'].encode('utf-8'))
    else:
        for item in features_list['item']:
            feat = item.encode('utf-8')
            config_dict.setdefault('features_list',[]).append(feat)

    methods_list = xml['configuration']['methods_list']

    if type(methods_list['item']) is unicode:
        config_dict.setdefault('methods_list',[]).append(methods_list['item'].encode('utf-8'))
    else:
        for item in methods_list['item']:
            method = item.encode('utf-8')
            config_dict.setdefault('methods_list',[]).append(method)

    return config_dict


def get_global_config(xml_config):
    """
    Construct some additional (output) paths from the xml configuration.
    :param xml_config:
    :return:
    """
    parent_path = xml_config['data_path'] + xml_config['dataset_name'] + '/'
    if not isdir(parent_path):
        makedirs(parent_path)

    tracklets_path = parent_path + 'tracklets/'
    clusters_path = parent_path + 'clusters/'
    intermediates_path = parent_path + 'intermediates/'
    feats_path = parent_path + 'feats/'
    kernels_path = parent_path + 'kernels/'

    return tracklets_path, clusters_path, intermediates_path, feats_path, kernels_path


def get_dataset_info(xml_config):
    """
    Get dataset-related information. Configuration from xml is as input needed.
    :param xml_config:
    :return:
    """
    dataset_name = xml_config['dataset_name']
    dataset_path = xml_config['datasets_path'] + dataset_name + '/'

    if dataset_name == 'hollywood2':
        dataset_info = get_hollywood2_config(dataset_path)
    elif dataset_name == 'highfive':
        dataset_info = get_highfive_config(dataset_path)
    elif dataset_name == 'ucf_sports_actions':
        dataset_info = get_ucfsportsaction_dataset(dataset_path)
    elif dataset_name == 'olympic_sports':
        dataset_info = get_olympicsports_dataset(dataset_path)

    return dataset_info


def get_hollywood2_config(parent_path):
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


def get_highfive_config(parent_path):
    '''
    Hard-codes some paths and configuration values.
    :return:
    '''

    videos_dir = parent_path + 'tv_human_interactions_videos/'
    # splits_dir = parent_path + 'tv_human_interaction_annotations/'  # TODO: think if i'll need to use for test

    # Half train/test partition got from:
    # http://www.robots.ox.ac.uk/~alonso/tv_human_interactions.html
    # ----------------------------------------------------
    train_test = dict(
        train_inds = dict(
            handShake = [2,14,15,16,18,19,20,21,24,25,26,27,28,32,40,41,42,43,44,45,46,47,48,49,50],
            highFive = [1,6,7,8,9,10,11,12,13,23,24,25,27,28,29,30,31,32,33,34,35,44,45,47,48],
            hug = [2,3,4,11,12,15,16,17,18,20,21,27,29,30,31,32,33,34,35,36,42,44,46,49,50],
            kiss = [1,7,8,9,10,11,12,13,14,16,17,18,22,23,24,26,29,31,35,36,38,39,40,41,42],
            negative = np.linspace(1,50,50).astype('int32')
        ),
        test_inds = dict(
            handShake = [1,3,4,5,6,7,8,9,10,11,12,13,17,22,23,29,30,31,33,34,35,36,37,38,39],
            highFive = [2,3,4,5,14,15,16,17,18,19,20,21,22,26,36,37,38,39,40,41,42,43,46,49,50],
            hug = [1,5,6,7,8,9,10,13,14,19,22,23,24,25,26,28,37,38,39,40,41,43,45,47,48],
            kiss = [2,3,4,5,6,15,19,20,21,25,27,28,30,32,33,34,37,43,44,45,46,47,48,49,50],
            negative = np.linspace(51,100,50).astype('int32')
        )
    )
    # ----------------------------------------------------

    action_names = sorted(train_test['train_inds'].keys(), key=lambda x: x)

    videonames = []
    int_inds = []
    train_test_inds = []
    for partition_name in train_test:
        n = len(videonames)
        for i, name in enumerate(action_names):
            videonames += [name + '_' + str(j).zfill(4)
                           for j in train_test[partition_name][name]]
            int_inds += [i]*len(train_test[partition_name][name])

        train_test_inds.append(np.linspace(n, len(videonames)-1, len(videonames)-n).astype('int32'))

    traintest_parts = []
    for part_inds in train_test_inds:
        part = np.zeros((len(videonames),)).astype('int32')
        part[part_inds] = 1
        traintest_parts.append(part)

    fullvideonames = [join(parent_path,'tv_human_interactions_videos/', videoname+'.avi')  for videoname in videonames]

    # create a matrix #{instances}x#{classes}, where entries are all "-1" except for 1s in corresponding class columns
    class_labels = (-1) * np.ones((len(int_inds),len(action_names))).astype('int32')
    for i in xrange(len(action_names)):
        class_labels[np.array(int_inds)==i,i] = 1

    class_labels = class_labels[:,np.where(np.array(action_names) != 'negative')[0]]

    return fullvideonames, videonames, class_labels, action_names, traintest_parts


def get_ucfsportsaction_dataset(parent_path):
    videos_dir = parent_path

    # From the publication webpage, some hard-coded metainfo.
    # http://crcv.ucf.edu/data/UCF_Sports_Action.php
    # ----------------------------------------------------
    action_classes = {
        'Diving' : 14,
        'Golf Swing' : 18,
        'Kicking' : 20,
        'Lifting' : 6,
        'Riding Horse' : 12,
        'Running' : 13,
        'SkateBoarding' : 12,
        'Swing-Bench' : 20,
        'Swing-Side' : 13,
        'Walking' : 22
    }

    train_inds = [5,6,7,8,9,10,11,12,13,14,21,22,23,24,25,26,27,28,29,30,31,32,39,40,41,42,43,44,45,46,47,48,49,50, \
                  51,52,55,56,57,58,63,64,65,66,67,68,69,70,75,76,77,78,79,80,81,82,83,88,89,90,91,92,93,94,95,102, \
                  103,104,105,106,107,108,109,110,111,112,113,114,115,120,121,122,123,124,125,126,127,128,136,137,  \
                  138,139,140,141,142,143,144,145,146,147,148,149,150]
    test_inds = [1,2,3,4,15,16,17,18,19,20,33,34,35,36,37,38,53,54,59,60,61,62,71,72,73,74,84,85,86,87,96,97,98,99, \
                 100,101,116,117,118,119,129,130,131,132,133,134,135]

    # ----------------------------------------------------

    # This dataset was quite a mess. So I used a re-organized version of it. READ carefully.
    #
    # These dataset provides both .avi videos and .jpg image files most of times. However, I FOUND PROBLEMS:
    # (1) Video were missing (Diving action, instances in folders: 008-014),
    # (2) Video was shorter than the sequence of JPGs (ex: Diving action, instance 006).
    #
    # Therefore, I generated new videos from JPGs when available, or copying the existing video otherwise. The
    # generation of videos was done with ffmpeg. For more info please refer to the provided "fix_ucf_sports_dataset.py".
    # It prepares the dataset to be parsed with the following code:

    action_names = sorted(action_classes.keys(), key = lambda x : x)

    videonames = []
    for i, element in enumerate(listdir(videos_dir)):  # important listdir lists in alphabetical order
        stem = splitext(element)
        if stem[1] == '.avi':
            videonames.append(stem[0])

    fullvideonames = [videos_dir + videoname for videoname in videonames]

    traintest_parts = []
    part = np.zeros((len(videonames),)).astype('int32')
    part[np.array(test_inds)-1] = 1
    traintest_parts.append(part)

    int_inds = []
    for i, name in enumerate(action_names):
        int_inds += [i] * action_classes[name]

    # create a matrix #{instances}x#{classes}, where entries are all "-1" except for 1s in corresponding class columns
    class_labels = (-1) * np.ones((len(int_inds),len(action_names))).astype('int32')
    for i in xrange(len(action_names)):
        class_labels[np.array(int_inds)==i,i] = 1

    # train_test_indx = (np.array(train_inds) - 1, np.array(test_inds) - 1)

    return fullvideonames, videonames, class_labels, action_names, traintest_parts


def get_olympicsports_dataset(parent_path):
    # recall in train_test_split/ diving_platform_10m and diving_spring_3m has the "diving" part missing. Correct it.
    action_names = ['basketball_layup', 'bowling', 'clean_and_jerk', 'discus_throw', 'diving_platform_10m', \
                    'diving_springboard_3m', 'hammer_throw', 'high_jump', 'javelin_throw', 'long_jump', 'pole_vault', \
                    'shot_put', 'snatch', 'tennis_serve', 'triple_jump', 'vault']


    fullvideonames = []
    videonames = []
    int_inds = []
    traintest_parts = []
    for i, part in enumerate(['train', 'test']):
        c = 0
        for action in action_names:
            with open(join(parent_path, 'train_test_split', part, action.replace('_',' ')+'.txt'), 'r') as f:
                k = 1
                line = f.readline().rstrip()
                fullvideonames.append(join(parent_path, action, line+'.mp4'))
                videonames.append(line)
                traintest_parts.append(i)
                while 1:
                    line = f.readline().rstrip()
                    if line != '':
                        fullvideonames.append(join(parent_path, action, line+'.mp4'))
                        videonames.append(line)
                        traintest_parts.append(i)
                        k += 1
                    else:
                        break
                int_inds += [c] * k
            c += 1


    class_labels = (-1) * np.ones((len(int_inds),len(action_names))).astype('int32')
    for i in xrange(len(action_names)):
        class_labels[np.array(int_inds)==i,i] = 1

    return fullvideonames, videonames, class_labels, action_names, [traintest_parts]



# ==============================================================================
# Main
# ==============================================================================


if __name__ == "__main__":
    # prepare configuration-related variables
    xml_config = load_XML_config('config.xml')  # get it from XML file
    tracklets_path, clusters_path, intermediates_path, feats_path, kernels_path = get_global_config(xml_config)  # build output dirs
    fullvideonames, videonames, class_labels, action_names, traintest_parts = get_dataset_info(xml_config)  # build dataset-related variable

    # Extract improved dense trajectories
    tracklet_extraction.extract_multithread(fullvideonames, videonames, xml_config['features_list'], tracklets_path, nt=xml_config['num_threads'])
    tracklet_clustering.cluster_multithread(tracklets_path, videonames, clusters_path)

    # BOVW-tree descriptor computation and classification
    if 'atep_bovwtree' in xml_config['methods_list']:
        # tracklet_representation.train_bovw_codebooks(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path, pca_reduction=False)
        # tracklet_representation.compute_bovw_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
        #                                                              feats_path + 'bovwtree/', \
        #                                                              pca_reduction=False, treelike=True, clusters_path=clusters_path)
        st_time = time.time()
        kernels = kernels.compute_kernels(feats_path + 'bovwtree/', videonames, traintest_parts, xml_config['features_list'])
        results = classification.classify(kernels, class_labels, traintest_parts, np.linspace(0, 1, 11), xml_config['features_list'], c=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 1e4, 1e5, 1e6])
        print('ATEP classification (bovwtree) took %.2f secs.' % (time.time() - st_time))
        # print_results(results)

    if 'atep_fvtree' in xml_config['methods_list']:
        # tracklet_representation.train_bovw_codebooks(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path, pca_reduction=False)
        # tracklet_representation.compute_bovw_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
        #                                                              feats_path + 'bovwtree/', \
        #                                                              pca_reduction=False, treelike=True, clusters_path=clusters_path)
        st_time = time.time()
        atep = kernels.compute_ATEP_kernels(feats_path + 'fvtree/', videonames, traintest_parts, xml_config['features_list'])
        results = classification.classify(atep, class_labels, traintest_parts, np.linspace(0, 1, 11), xml_config['features_list'], c=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 1e4, 1e5, 1e6])
        print('ATEP classification (fvtree) took %.2f secs.' % (time.time() - st_time))
        # print_results(results)

    if 'atnbep_fvtree' in xml_config['methods_list']:
        # tracklet_representation.train_bovw_codebooks(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path, pca_reduction=False)
        # tracklet_representation.compute_bovw_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
        #                                                              feats_path + 'bovwtree/', \
        #                                                              pca_reduction=False, treelike=True, clusters_path=clusters_path)
        st_time = time.time()
        atnbep = kernels.compute_ATNBEP_kernels(feats_path + 'fvtree/', videonames, traintest_parts, xml_config['features_list'], from_disk=False)
        results = classification.classify(atnbep, class_labels, traintest_parts, np.linspace(0, 1, 11), xml_config['features_list'], c=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 1e4, 1e5, 1e6])
        print('ATBEP classification (fvtree) took %.2f secs.' % (time.time() - st_time))
        # classification.print_results(results)

    # WORKING HERE NOW
    if 'atnbep_vd-fv' in xml_config['methods_list']:
        # tracklet_representation.train_bovw_codebooks(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path, pca_reduction=False)
        tracklet_representation.compute_fv_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + 'fvtree/', \
                                                                   treelike=True, clusters_path=clusters_path, nt=xml_config['num_threads'])
        # tracklet_representation.compute_vd_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
        #                                                            feats_path + 'vdtree/', \
        #                                                            treelike=True, clusters_path=clusters_path, nt=xml_config['num_threads'])

        st_time = time.time()
        # atep = kernels.compute_ATEP_kernels(feats_path + 'fvtree/', videonames, traintest_parts, xml_config['features_list'], \
        #                                     kernels_path + 'atep-fv/', use_disk=False)
        atnbep = kernels.compute_ATNBEP_kernels(feats_path + 'fvtree/', videonames, traintest_parts, xml_config['features_list'], \
                                                kernels_path + 'atnbep-fv/', use_disk=False, nt=xml_config['num_threads'])
        atep = kernels.compute_ATEP_kernels(feats_path + 'vdtree/', videonames, traintest_parts, xml_config['features_list'], \
                                            kernels_path + 'atep-vd/', use_disk=False, nt=xml_config['num_threads'])

        merged = [utils.merge_dictionaries([atep[i], atnbep[i]]) for i in xrange(len(atep))]
        combs = [c for c in itertools.product(*[np.linspace(0, 1, 11),np.linspace(0, 1, 11),[1.]])]
        results = classification.classify(merged, \
                                          class_labels, traintest_parts, combs, \
                                          xml_config['features_list'], \
                                          c=[0.001, 0.01, 0.1, 1, 10, 100, 1000, 1e4, 1e5, 1e6])
        print('ATBEP classification (vdtree) took %.2f secs.' % (time.time() - st_time))
        classification.print_results(results)

    # # FV-tree descriptor computation and classification
    # if 'atep_fvtree' in xml_config['methods_list']:
    #     # tracklet_representation.train_fv_gmms(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path)
    #     tracklet_representation.compute_fv_descriptors(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
    #                                                                feats_path + 'fvtree/', \
    #                                                                treelike=True, clusters_path=clusters_path)
    #
    #     st_time = time.time()
    #     results = atep_classification.classify(feats_path + 'fvtree/', videonames, class_labels, traintest_parts, \
    #                                            np.linspace(0, 1, 11), xml_config['features_list'], \
    #                                            c=[0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 1e4, 1e5, 1e6])
    #     print('ATEP classification (fvtree) took %.2f secs.' % (time.time() - st_time))
    #     print_results(results)
    #
    # # Darwin-tree descriptor computation and classification
    if 'atep_vdtree' in xml_config['methods_list']:
        tracklet_representation.train_fv_gmms(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path)
        tracklet_representation.compute_vd_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + 'vdtree/', \
                                                                   treelike=True, clusters_path=clusters_path)
        st_time = time.time()
        atep = kernels.compute_ATEP_kernels(feats_path + 'vdtree/', videonames, traintest_parts, xml_config['features_list'], from_disk=False)
        results = classification.classify(atep, class_labels, traintest_parts, np.linspace(0, 1, 11), xml_config['features_list'], c=[1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000, 1e4, 1e5])
        print('ATEP classification (vdtree) took %.2f secs.' % (time.time() - st_time))
        print_results(results)

    # if 'atep_fvtree+vdtree' in xml_config['methods_list']:
    #     # tracklet_representation.train_fv_gmms(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path)
    #     # tracklet_representation.compute_vd_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
    #     #                                                            feats_path + 'vdtree/', \
    #     #                                                            treelike=True, clusters_path=clusters_path)
    #     # tracklet_representation.compute_vd_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
    #     #                                                            feats_path + 'vdtree/', \
    #     #                                                            treelike=True, clusters_path=clusters_path)
    #     st_time = time.time()
    #     results = atep_classification.classify([feats_path + 'fvtree', feats_path + 'vdtree/'], videonames, class_labels, traintest_parts, \
    #                                            np.linspace(0, 1, 11), xml_config['features_list'], c=[0.01, 0.1, 1, 10, 100, 1000, 1e4, 1e5, 1e6])
    #     print('ATEP classification (vdtree) took %.2f secs.' % (time.time() - st_time))
    #     print_results(results)
    #
    # if 'atep_fvdpathtree' in xml_config['methods_list']:
    #     # tracklet_representation.train_fv_gmms(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path)
    #     # tracklet_representation.compute_fv_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
    #     #                                                            feats_path + 'fvtree/', \
    #     #                                                            treelike=True, clusters_path=clusters_path)
    #     # TODO: use multithreading
    #     tracklet_representation.compute_node_evolution(feats_path + 'fvtree/', videonames, traintest_parts, xml_config['features_list'], \
    #                                                    feats_path + 'fvevo/')

    quit()  # TODO: remove this for further processing