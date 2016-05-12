#!/Users/Shared/anaconda/bin/python

'''Framework for action/activity recognition on videos (computer vision research work)

LICENSE: BSD

Copyrights: Albert Clap\'{e}s, 2015

'''

import numpy as np
from os.path import join
import utils
import itertools
import argparse

from configuration import *
import tracklet_extraction, tracklet_clustering, tracklet_representation, kernels, classification

# ==============================================================================
# Main
# ==============================================================================

desc_weights_gbl = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1], \
                    [0.5,0.5,0,0], [0.5,0,0.5,0], [0.5,0,0,0.5], [0,0.5,0.5,0], [0,0.5,0,0.5], [0,0,0.5,0.5], \
                    [0,0.333,0.333,0.333], [0.333,0,0.333,0.333], [0.333,0.333,0,0.333], [0.333,0.333,0.333,0], \
                    [0.25,0.25,0.25,0.25]]

desc_weights_gbl = [[1,0,0], [0,1,0], [0,0,1], \
                    [0.5,0.5,0], [0.5,0,0.5], [0,0.5,0.5], \
                    [0,0.333,0.333,0.333]]

# C_gbl = [1e-6, 1e-5, 5e-5, 1e-4, 2.5e-4, 5e-4, 7.5e-4, 1e-3, 2.5e-3, 5e-3, 7.5e-3, 1e-2, 2.5e-2, 5e-2, 7.5e-2, 0.1, 0.25, 0.5, 0.75, 1, 2.5, 5, 7.5, 10, 25, 50, 75, 100]
C_gbl = [3**(-5), 3**(-4), 3**(-3), 3**(-2), 3**(-1), 3**0, 3**1, 3**2, 3**3, 3**4, 3**5, 3**6, 3**7]

# DEBUG
# ---
# C_gbl = [1e7 ]
desc_weights_gbl = [[1]] #[[0.25,0.25,0.25,0.25]]
# ---

# ucf_sports_actions --methods atep-fv acp-fv atep-vd acp-vd atnbep atlp atep-fv+atep-vd atep-fv+atnbep atep-vd+atnbep acp-fv+atnbep acp-vd+atnbep atep-fv+atep-vd+atnbep acp-fv+acp-vd+atnbep -nt 1 -s 42
# highfive --methods atep-fv acp-fv atep-vd acp-vd atnbep atlp atep-fv+atep-vd atep-fv+atnbep atep-vd+atnbep acp-fv+atnbep acp-vd+atnbep atep-fv+atep-vd+atnbep acp-fv+acp-vd+atnbep -nt 1 -s 42
# olympic_sports --methods atep-fv acp-fv atep-vd acp-vd atnbep atlp atep-fv+atep-vd atep-fv+atnbep atep-vd+atnbep acp-fv+atnbep acp-vd+atnbep atep-fv+atep-vd+atnbep acp-fv+acp-vd+atnbep -nt 1 -s 42


if __name__ == "__main__":

    ################################################################
    ## Program arguments, configuration, and dataset-related info ##
    ################################################################

    # read program arguments
    # example: "python main_multithread.py dataset_name -m method1 [method2 ... ] -nt 20 -v"
    parser = argparse.ArgumentParser(description='Process the videos to see whether they contain speaking-while-facing-a-camera scenes.')
    parser.add_argument('dataset_name', nargs=1, help='Choose among: ucf_sports_actions, highfive, olympic_sports, hollywood2.')
    parser.add_argument('-m', '--methods', nargs='+', default=[], help='List methods to use: atep-bovw, atep-fv, atep-vd, atnbep, and combinations using + sign.')
    parser.add_argument('-nt', '--num-threads', dest='nt', type=int, default=1, help='Set the number of threads for parallelization.')
    parser.add_argument('-s', '--random-seed', dest='random_seed', type=int, default=None, help='Set the number of threads for parallelization.')
    parser.add_argument('-v', '--verbose', action='store_true', help='Whether or not print debugging information.')
    args = parser.parse_args()

    # read additional configuration parameters from an xml file
    xml_config = load_XML_config('config.xml')  # datasets path, derived (output) data path, and trajectory descriptors to use

    # from a dataset we need some metadata. any function to process a dataset must return those
    fullvideonames, videonames, class_labels, action_names, traintest_parts, opt_criterion, traj_length = \
        get_dataset_info(xml_config['datasets_path'], args.dataset_name[0])  # build dataset-related variable

    tracklets_path, clusters_path, intermediates_path, feats_path, kernels_path = \
        create_main_directories(join(xml_config['data_path'], args.dataset_name[0]))

    print args


    ##########################
    ## Execution of methods ##
    ##########################

    # whatever is the method, these two are common and mandatory
    tracklet_extraction.extract_multithread(fullvideonames, videonames, xml_config['features_list'], traj_length, tracklets_path, nt=args.nt, verbose=args.verbose)
    tracklet_clustering.cluster_multithread(tracklets_path, videonames, clusters_path, nt=args.nt, verbose=args.verbose)

    if 'atep-bovw' in args.methods:
        tracklet_representation.train_bovw_codebooks(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path, pca_reduction=False, nt=args.nt, verbose=args.verbose)
        tracklet_representation.compute_bovw_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                     feats_path + '/bovwtree/', \
                                                                     treelike=True, pca_reduction=False, clusters_path=clusters_path, nt=args.nt, verbose=args.verbose)

        atep_bovw = kernels.compute_ATEP_kernels(feats_path + '/bovwtree/', videonames, traintest_parts, xml_config['features_list'], \
                                                 kernels_path + '/atep-bovw/', kernel_type='intersection', norm='l1', power_norm=False, \
                                                 use_disk=False, nt=args.nt, verbose=args.verbose)

        params = [[[1]], [1], np.linspace(0,1,11), desc_weights_gbl]
        results = classification.classify(atep_bovw, \
                                          class_labels, traintest_parts, params, \
                                          xml_config['features_list'], \
                                          C=C_gbl,
                                          strategy='kernel_fusion',
                                          opt_criterion=opt_criterion,
                                          verbose=args.verbose)
        print 'atep-bovw'
        classification.print_results(results, opt_criterion=opt_criterion)

    if 'atep-fv' in args.methods:
        tracklet_representation.train_fv_gmms(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path, pca_reduction=True, nt=args.nt, verbose=args.verbose)
        tracklet_representation.compute_fv_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + '/fvtree/', \
                                                                   treelike=True, pca_reduction=True, clusters_path=clusters_path, nt=args.nt, verbose=args.verbose)

        atep_fv = kernels.compute_ATEP_kernels(feats_path + '/fvtree/', videonames, traintest_parts, xml_config['features_list'], \
                                               kernels_path + '/atep-fv/', kernel_type='hellinger', use_disk=False, nt=args.nt, verbose=args.verbose)

        params = [[[1]], [1], np.linspace(0,1,11), desc_weights_gbl]
        results = classification.classify(atep_fv, \
                                          class_labels, traintest_parts, params, \
                                          xml_config['features_list'], \
                                          C=C_gbl,
                                          strategy='kernel_fusion',
                                          opt_criterion=opt_criterion,
                                          verbose=args.verbose)
        print 'atep-fv'
        classification.print_results(results, opt_criterion=opt_criterion)

    if 'acp-fv' in args.methods:
        tracklet_representation.train_fv_gmms(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path, pca_reduction=True, nt=args.nt, verbose=args.verbose)
        tracklet_representation.compute_fv_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + '/fvtree/', \
                                                                   treelike=True, pca_reduction=True, clusters_path=clusters_path, nt=args.nt, verbose=args.verbose)

        acp_fv = kernels.compute_ACP_kernels(feats_path + '/fvtree/', videonames, traintest_parts, xml_config['features_list'], \
                                               kernels_path + '/acp-fv/', kernel_type='hellinger', use_disk=False, nt=args.nt, verbose=args.verbose)

        params = [[[1]], [1], np.linspace(0,1,11), desc_weights_gbl]
        results = classification.classify(acp_fv, \
                                          class_labels, traintest_parts, params, \
                                          xml_config['features_list'], \
                                          C=C_gbl,
                                          strategy='kernel_fusion',
                                          opt_criterion=opt_criterion,
                                          verbose=args.verbose)
        print 'acp-fv'
        classification.print_results(results, opt_criterion=opt_criterion)

    # # Darwin-tree descriptor computation and classification
    if 'atep-vd' in args.methods:
        tracklet_representation.train_fv_gmms(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path, pca_reduction=True, nt=args.nt)
        tracklet_representation.compute_vd_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + '/vdtree/', \
                                                                   treelike=True, pca_reduction=True, clusters_path=clusters_path, nt=args.nt, verbose=args.verbose)

        atep_vd = kernels.compute_ATEP_kernels(feats_path + '/vdtree/', videonames, traintest_parts, xml_config['features_list'], \
                                               kernels_path + '/atep-vd/', kernel_type='hellinger', use_disk=False, nt=args.nt, verbose=args.verbose)

        params = [[[1]], [1], np.linspace(0,1,11), desc_weights_gbl]
        results = classification.classify(atep_vd, \
                                          class_labels, traintest_parts, params, \
                                          xml_config['features_list'], \
                                          C=C_gbl,
                                          strategy='kernel_fusion',
                                          opt_criterion=opt_criterion,
                                          verbose=args.verbose)
        print 'atep-vd'
        classification.print_results(results, opt_criterion=opt_criterion)

    # # Darwin-tree descriptor computation and classification
    if 'acp-vd' in args.methods:
        tracklet_representation.train_fv_gmms(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path, pca_reduction=True, nt=args.nt)
        tracklet_representation.compute_vd_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + '/vdtree/', \
                                                                   treelike=True, pca_reduction=True, clusters_path=clusters_path, nt=args.nt, verbose=args.verbose)

        acp_vd = kernels.compute_ACP_kernels(feats_path + '/vdtree/', videonames, traintest_parts, xml_config['features_list'], \
                                               kernels_path + '/acp-vd/', kernel_type='hellinger', use_disk=False, nt=args.nt, verbose=args.verbose)

        params = [[[1]], [1], np.linspace(0,1,11), desc_weights_gbl]
        results = classification.classify(acp_vd, \
                                          class_labels, traintest_parts, params, \
                                          xml_config['features_list'], \
                                          C=C_gbl,
                                          strategy='kernel_fusion',
                                          opt_criterion=opt_criterion,
                                          verbose=args.verbose)
        print 'acp-vd'
        classification.print_results(results, opt_criterion=opt_criterion)

    if 'atnbep' in args.methods:
        tracklet_representation.train_fv_gmms(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path, pca_reduction=True, nt=args.nt, verbose=args.verbose)
        tracklet_representation.compute_fv_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + '/fvtree/', \
                                                                   treelike=True, pca_reduction=True, clusters_path=clusters_path, nt=args.nt, verbose=args.verbose)

        atnbep = kernels.compute_ATNBEP_kernels(feats_path + '/fvtree/', videonames, traintest_parts, xml_config['features_list'], \
                                                   kernels_path + '/atnbep/', use_disk=False, nt=args.nt, verbose=args.verbose)

        # params = [[[1]], [0], [0], desc_weights_gbl]
        params = [[[1]], np.linspace(0,1,21), [0], desc_weights_gbl]  # new fw+rv atnbep kernel
        results = classification.classify(atnbep, \
                                          class_labels, traintest_parts, params, \
                                          xml_config['features_list'], \
                                          C=C_gbl,
                                          strategy='kernel_fusion',
                                          opt_criterion=opt_criterion,
                                          random_state=args.random_seed,
                                          verbose=args.verbose)
        print 'atnbep'
        classification.print_results(results, opt_criterion=opt_criterion)

    if 'atlp' in args.methods:
        tracklet_representation.train_fv_gmms(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path, pca_reduction=True, nt=args.nt, verbose=args.verbose)
        tracklet_representation.compute_fv_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + '/fvtree/', \
                                                                   treelike=True, pca_reduction=True, clusters_path=clusters_path, nt=args.nt, verbose=args.verbose)

        atlp = kernels.compute_ATLP_kernels(feats_path + '/fvtree/', videonames, traintest_parts, xml_config['features_list'], \
                                            kernels_path + '/atlp/', use_disk=False, nt=args.nt, verbose=args.verbose)

        # params = [[[1]], [0], [0], desc_weights_gbl]
        params = [[[1]], np.linspace(0,1,11), [0], desc_weights_gbl]  # new fw+rv atnbep kernel
        results = classification.classify(atlp, \
                                          class_labels, traintest_parts, params, \
                                          xml_config['features_list'], \
                                          C=C_gbl,
                                          strategy='kernel_fusion',
                                          opt_criterion=opt_criterion,
                                          random_state=args.random_seed,
                                          verbose=args.verbose)
        print 'atlp'
        classification.print_results(results, opt_criterion=opt_criterion)

    if 'atep-fv+atep-vd' in args.methods:
        tracklet_representation.train_fv_gmms(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path, pca_reduction=True, nt=args.nt)
        tracklet_representation.compute_fv_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + '/fvtree/', \
                                                                   treelike=True, pca_reduction=True, clusters_path=clusters_path, nt=args.nt, verbose=args.verbose)
        tracklet_representation.compute_vd_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + '/vdtree/', \
                                                                   treelike=True, pca_reduction=True, clusters_path=clusters_path, nt=args.nt, verbose=args.verbose)

        atep_fv = kernels.compute_ATEP_kernels(feats_path + '/fvtree/', videonames, traintest_parts, xml_config['features_list'], \
                                            kernels_path + '/atep-fv/', use_disk=False, nt=args.nt, verbose=args.verbose)
        atep_vd = kernels.compute_ATEP_kernels(feats_path + '/vdtree/', videonames, traintest_parts, xml_config['features_list'], \
                                               kernels_path + '/atep-vd/', use_disk=False, nt=args.nt, verbose=args.verbose)

        merged = [utils.merge_dictionaries([atep_fv[i], atep_vd[i]]) for i in xrange(len(atep_fv))]

        # w = [(a,1-a) for a in np.linspace(0,1,11)]
        w = [[0,1],[0.5,0.5],[1,0]]
        params = [w, [[1],[1]], np.linspace(0,1,11), desc_weights_gbl]
        results = classification.classify(merged, \
                                          class_labels, traintest_parts, params, \
                                          xml_config['features_list'], \
                                          C=C_gbl,
                                          strategy='learning_based_fusion',
                                          opt_criterion=opt_criterion,
                                          random_state=args.random_seed,
                                          verbose=args.verbose)
        print 'atep-fv+atep-vd'
        classification.print_results(results, opt_criterion=opt_criterion)

    if 'atep-fv+atnbep' in args.methods:
        tracklet_representation.train_fv_gmms(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path, pca_reduction=True, nt=args.nt)
        tracklet_representation.compute_fv_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + '/fvtree/', \
                                                                   treelike=True, pca_reduction=True, clusters_path=clusters_path, nt=args.nt, verbose=args.verbose)

        atep_fv = kernels.compute_ATEP_kernels(feats_path + '/fvtree/', videonames, traintest_parts, xml_config['features_list'], \
                                            kernels_path + '/atep-fv/', use_disk=False, nt=args.nt, verbose=args.verbose)
        atnbep = kernels.compute_ATNBEP_kernels(feats_path + '/fvtree/', videonames, traintest_parts, xml_config['features_list'], \
                                               kernels_path + '/atnbep/', use_disk=False, nt=args.nt, verbose=args.verbose)

        merged = [utils.merge_dictionaries([atep_fv[i], atnbep[i]]) for i in xrange(len(atep_fv))]

        # w = [(a,1-a) for a in np.linspace(0,1,11)]
        w = [[0,1],[0.5,0.5],[1,0]]
        params = [w, [c for c in itertools.product(*[[1],[0,.5,1]])], np.linspace(0,1,11), desc_weights_gbl]
        results = classification.classify(merged, \
                                          class_labels, traintest_parts, params, \
                                          xml_config['features_list'], \
                                          C=C_gbl,
                                          strategy='learning_based_fusion',
                                          opt_criterion=opt_criterion,
                                          random_state=args.random_seed,
                                          verbose=args.verbose)
        print 'atep-fv+atnbep'
        classification.print_results(results, opt_criterion=opt_criterion)

    if 'acp-fv+atnbep' in args.methods:
        tracklet_representation.train_fv_gmms(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path, pca_reduction=True, nt=args.nt)
        tracklet_representation.compute_fv_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + '/fvtree/', \
                                                                   treelike=True, pca_reduction=True, clusters_path=clusters_path, nt=args.nt, verbose=args.verbose)

        acp_fv = kernels.compute_ACP_kernels(feats_path + '/fvtree/', videonames, traintest_parts, xml_config['features_list'], \
                                            kernels_path + '/acp-fv/', use_disk=False, nt=args.nt, verbose=args.verbose)
        atnbep = kernels.compute_ATNBEP_kernels(feats_path + '/fvtree/', videonames, traintest_parts, xml_config['features_list'], \
                                               kernels_path + '/atnbep/', use_disk=False, nt=args.nt, verbose=args.verbose)

        merged = [utils.merge_dictionaries([acp_fv[i], atnbep[i]]) for i in xrange(len(acp_fv))]

        # w = [(a,1-a) for a in np.linspace(0,1,11)]
        w = [[0,1],[0.5,0.5],[1,0]]
        params = [w, [c for c in itertools.product(*[[1],np.linspace(0,1,11)])], np.linspace(0,1,11), desc_weights_gbl]
        results = classification.classify(merged, \
                                          class_labels, traintest_parts, params, \
                                          xml_config['features_list'], \
                                          C=C_gbl,
                                          strategy='learning_based_fusion',
                                          opt_criterion=opt_criterion,
                                          random_state=args.random_seed,
                                          verbose=args.verbose)
        print 'acp-fv+atnbep'
        classification.print_results(results, opt_criterion=opt_criterion)


    if 'atep-vd+atnbep' in args.methods:
        tracklet_representation.train_fv_gmms(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path, pca_reduction=True, nt=args.nt)
        tracklet_representation.compute_vd_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + '/vdtree/', \
                                                                   treelike=True, pca_reduction=True, clusters_path=clusters_path, nt=args.nt, verbose=args.verbose)
        tracklet_representation.compute_fv_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + '/fvtree/', \
                                                                   treelike=True, pca_reduction=True, clusters_path=clusters_path, nt=args.nt, verbose=args.verbose)

        atep_vd = kernels.compute_ATEP_kernels(feats_path + '/vdtree/', videonames, traintest_parts, xml_config['features_list'], \
                                               kernels_path + '/atep-vd/', use_disk=False, nt=args.nt, verbose=args.verbose)
        atnbep = kernels.compute_ATNBEP_kernels(feats_path + '/fvtree/', videonames, traintest_parts, xml_config['features_list'], \
                                               kernels_path + '/atnbep/', use_disk=False, nt=args.nt, verbose=args.verbose)

        merged = [utils.merge_dictionaries([atep_vd[i], atnbep[i]]) for i in xrange(len(atep_vd))]

        w = [(a,1-a) for a in np.linspace(0,1,11)] # [[0,1],[0.5,0.5],[1,0]]
        params = [w, [c for c in itertools.product(*[[1],np.linspace(0,1,3)])], np.linspace(0,1,3), desc_weights_gbl]
        results = classification.classify(merged, \
                                          class_labels, traintest_parts, params, \
                                          xml_config['features_list'], \
                                          C=C_gbl,
                                          strategy='kernel_fusion',
                                          opt_criterion=opt_criterion,
                                          random_state=args.random_seed,
                                          verbose=args.verbose)
        print 'atep-vd+atnbep'
        classification.print_results(results, opt_criterion=opt_criterion)

    if 'acp-vd+atnbep' in args.methods:
        tracklet_representation.train_fv_gmms(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path, pca_reduction=True, nt=args.nt)
        tracklet_representation.compute_vd_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + '/vdtree/', \
                                                                   treelike=True, pca_reduction=True, clusters_path=clusters_path, nt=args.nt, verbose=args.verbose)
        tracklet_representation.compute_fv_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + '/fvtree/', \
                                                                   treelike=True, pca_reduction=True, clusters_path=clusters_path, nt=args.nt, verbose=args.verbose)

        acp_vd = kernels.compute_ACP_kernels(feats_path + '/vdtree/', videonames, traintest_parts, xml_config['features_list'], \
                                               kernels_path + '/acp-vd/', use_disk=False, nt=args.nt, verbose=args.verbose)
        atnbep = kernels.compute_ATNBEP_kernels(feats_path + '/fvtree/', videonames, traintest_parts, xml_config['features_list'], \
                                               kernels_path + '/atnbep/', use_disk=False, nt=args.nt, verbose=args.verbose)

        merged = [utils.merge_dictionaries([acp_vd[i], atnbep[i]]) for i in xrange(len(acp_vd))]

        w = [(a,1-a) for a in np.linspace(0,1,11)] # [[0,1],[0.5,0.5],[1,0]]
        params = [w, [c for c in itertools.product(*[[1],np.linspace(0,1,11)])], np.linspace(0,1,11), desc_weights_gbl]
        results = classification.classify(merged, \
                                          class_labels, traintest_parts, params, \
                                          xml_config['features_list'], \
                                          C=C_gbl,
                                          strategy='learning_based_fusion',
                                          opt_criterion=opt_criterion,
                                          random_state=args.random_seed,
                                          verbose=args.verbose)
        print 'acp-vd+atnbep'
        classification.print_results(results, opt_criterion=opt_criterion)


    if 'atep-fv+atep-vd+atnbep' in args.methods:
        tracklet_representation.train_fv_gmms(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path, pca_reduction=True, nt=args.nt)
        tracklet_representation.compute_fv_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + '/fvtree/', \
                                                                   treelike=True, pca_reduction=True, clusters_path=clusters_path, nt=args.nt, verbose=args.verbose)
        tracklet_representation.compute_vd_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + '/vdtree/', \
                                                                   treelike=True, pca_reduction=True, clusters_path=clusters_path, nt=args.nt, verbose=args.verbose)

        atep_fv = kernels.compute_ATEP_kernels(feats_path + '/fvtree/', videonames, traintest_parts, xml_config['features_list'], \
                                            kernels_path + '/atep-fv/', use_disk=False, nt=args.nt, verbose=args.verbose)
        atep_vd = kernels.compute_ATEP_kernels(feats_path + '/vdtree/', videonames, traintest_parts, xml_config['features_list'], \
                                               kernels_path + '/atep-vd/', use_disk=False, nt=args.nt, verbose=args.verbose)
        atnbep = kernels.compute_ATNBEP_kernels(feats_path + '/fvtree/', videonames, traintest_parts, xml_config['features_list'], \
                                               kernels_path + '/atnbep/', use_disk=False, nt=args.nt, verbose=args.verbose)

        merged = [utils.merge_dictionaries([atep_fv[i], atep_vd[i]]) for i in xrange(len(atep_fv))]
        merged = [utils.merge_dictionaries([merged[i], atnbep[i]]) for i in xrange(len(merged))]
        w = utils.uniform_weights_dist(3)
        # w = [[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[0,.5,.5],[.5,0,.5],[.333,.333,.333]]
        params = [w, [c for c in itertools.product(*[[1],[1],np.linspace(0,1,11)])], np.linspace(0,1,11), desc_weights_gbl]
        results = classification.classify(merged, \
                                          class_labels, traintest_parts, params, \
                                          xml_config['features_list'], \
                                          C=C_gbl,
                                          strategy='learning_based_fusion',
                                          opt_criterion=opt_criterion,
                                          random_state=args.random_seed,
                                          verbose=args.verbose)
        print 'atep-fv+atep-vd+atnbep'
        classification.print_results(results, opt_criterion=opt_criterion)

    if 'acp-fv+acp-vd+atnbep' in args.methods:
        tracklet_representation.train_fv_gmms(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path, pca_reduction=True, nt=args.nt)
        tracklet_representation.compute_fv_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + '/fvtree/', \
                                                                   treelike=True, pca_reduction=True, clusters_path=clusters_path, nt=args.nt, verbose=args.verbose)
        tracklet_representation.compute_vd_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + '/vdtree/', \
                                                                   treelike=True, pca_reduction=True, clusters_path=clusters_path, nt=args.nt, verbose=args.verbose)

        acp_fv = kernels.compute_ACP_kernels(feats_path + '/fvtree/', videonames, traintest_parts, xml_config['features_list'], \
                                            kernels_path + '/acp-fv/', use_disk=False, nt=args.nt, verbose=args.verbose)
        acp_vd = kernels.compute_ACP_kernels(feats_path + '/vdtree/', videonames, traintest_parts, xml_config['features_list'], \
                                               kernels_path + '/acp-vd/', use_disk=False, nt=args.nt, verbose=args.verbose)
        atnbep = kernels.compute_ATNBEP_kernels(feats_path + '/fvtree/', videonames, traintest_parts, xml_config['features_list'], \
                                               kernels_path + '/atnbep/', use_disk=False, nt=args.nt, verbose=args.verbose)

        merged = [utils.merge_dictionaries([acp_fv[i], acp_vd[i]]) for i in xrange(len(acp_fv))]
        merged = [utils.merge_dictionaries([merged[i], atnbep[i]]) for i in xrange(len(merged))]
        w = utils.uniform_weights_dist(3)
        # w = [[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[0,.5,.5],[.5,0,.5],[.333,.333,.333]]
        params = [w, [c for c in itertools.product(*[[1],[1],np.linspace(0,1,11)])], np.linspace(0,1,11), desc_weights_gbl]
        results = classification.classify(merged, \
                                          class_labels, traintest_parts, params, \
                                          xml_config['features_list'], \
                                          C=C_gbl,
                                          strategy='learning_based_fusion',
                                          opt_criterion=opt_criterion,
                                          random_state=args.random_seed,
                                          verbose=args.verbose)
        print 'atep-fv+atep-vd+atnbep'
        classification.print_results(results, opt_criterion=opt_criterion)

    if 'atep-vd+atnbep+atlp' in args.methods:
        tracklet_representation.train_fv_gmms(tracklets_path, videonames, traintest_parts, xml_config['features_list'], intermediates_path, pca_reduction=True, nt=args.nt)
        tracklet_representation.compute_fv_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + '/fvtree/', \
                                                                   treelike=True, pca_reduction=True, clusters_path=clusters_path, nt=args.nt, verbose=args.verbose)
        tracklet_representation.compute_vd_descriptors_multithread(tracklets_path, intermediates_path, videonames, traintest_parts, xml_config['features_list'], \
                                                                   feats_path + '/vdtree/', \
                                                                   treelike=True, pca_reduction=True, clusters_path=clusters_path, nt=args.nt, verbose=args.verbose)

        atep_fv = kernels.compute_ATEP_kernels(feats_path + '/fvtree/', videonames, traintest_parts, xml_config['features_list'], \
                                            kernels_path + '/atep-fv/', use_disk=False, nt=args.nt, verbose=args.verbose)
        atnbep = kernels.compute_ATNBEP_kernels(feats_path + '/fvtree/', videonames, traintest_parts, xml_config['features_list'], \
                                               kernels_path + '/atnbep/', use_disk=False, nt=args.nt, verbose=args.verbose)
        atlp = kernels.compute_ATLP_kernels(feats_path + '/fvtree/', videonames, traintest_parts, xml_config['features_list'], \
                                            kernels_path + '/atlp/', use_disk=False, nt=args.nt, verbose=args.verbose)
        merged = [utils.merge_dictionaries([atep_fv[i], atnbep[i]]) for i in xrange(len(atep_fv))]
        merged = [utils.merge_dictionaries([merged[i], atlp[i]]) for i in xrange(len(merged))]
        w = utils.uniform_weights_dist(3)
        # w = [[1,0,0],[0,1,0],[0,0,1],[.5,.5,0],[0,.5,.5],[.5,0,.5],[.333,.333,.333]]
        params = [w, [c for c in itertools.product(*[[1],[1],np.linspace(0,1,11)])], np.linspace(0,1,11), desc_weights_gbl]
        results = classification.classify(merged, \
                                          class_labels, traintest_parts, params, \
                                          xml_config['features_list'], \
                                          C=C_gbl,
                                          strategy='learning_based_fusion',
                                          opt_criterion=opt_criterion,
                                          random_state=args.random_seed,
                                          verbose=args.verbose)
        print 'atep-fv+atnbep+atlp'
        classification.print_results(results, opt_criterion=opt_criterion)

    quit()  # TODO: remove this for further processing
