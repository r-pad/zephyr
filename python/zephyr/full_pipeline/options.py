import argparse

def getOptions():
    parser = argparse.ArgumentParser()
    '''About dataset'''
    parser.add_argument("--bop_root", type=str, default="/datasets/bop/")
    parser.add_argument("--ppf_resuls_root", type=str, default="/datasets/bop/ppf_data/results")
    parser.add_argument("--dataset_name", type=str, default="lmo")
    parser.add_argument("--model_type", type=str, default=None)
    parser.add_argument("--split_name", type=str, default="bop_test")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--split_type", type=str, default=None)
    parser.add_argument("--skip", type=int, default=1,
        help="Get 1 out of skip number of test cases")
    parser.add_argument('--model_ids', type=int, default=None, nargs="+",
        help="The model ids that will be processed. If None, all object ids will be take")

    '''About hypotheses'''
    parser.add_argument("--no_sift", action="store_true",
        help="If set, no SIFT hypotheses will be used")
    parser.add_argument("--no_ppf", action="store_true",
        help="If set, no PPF hypotheses will be used")
    parser.add_argument("--ppf_results_file", type=str, default=None,
        help="Path to the csv storing PPF hypotheses. If None, it will be found at default location. ")

    '''About model featurization'''
    parser.add_argument("--grid_dir_name", type=str, default="grid_0.7m")
    parser.add_argument("--sampled_model_dir_name", type=str, default="model_pc")
    parser.add_argument("--real_model_dir_name", type=str, default=None, 
        help="If set, then the model load from this folder will be returned as model_points, \
              and those from the sampled_model_dir_name will be only used for SIFT featurization")
    parser.add_argument("--sampled_model_leaf_size", type=float, default=7)
    parser.add_argument("--sampled_model_n_samples", type=int, default=100000)
    parser.add_argument("--grid_indices_path", type=str, default='/datasets/verts_grid_0.npy')
    parser.add_argument("--model_sift_feature_size", type=int, default=11)
    parser.add_argument("--feature_index_gpu", type=int, default=None)

    '''About obs to model feature matching'''
    parser.add_argument("--feature_match_K", type=int, default=10)
    parser.add_argument("--num_sampled_trans", type=int, default=1000)
    parser.add_argument("--uniform_sampling", action="store_true")
    parser.add_argument("--oracle_sampling", action="store_true",
        help="If set, will add the pose closest to the ground truth into the hypotheses")
    parser.add_argument("--oracle_hypo", action="store_true",
        help="If set, will add the ground truth pose into the hypotheses. ")

    '''About saving intermediate data'''
    parser.add_argument("--save_scoring_results", action="store_true",
        help="If set, save the scoring data. ")
    parser.add_argument("--score_data_save_dir_name", type=str, default=None,
        help="Folder name to save the score data. If None, it will be set according to other information. ")

    '''About scoring model'''
    parser.add_argument("--scoring_gpu", type=int, default=0)

    return parser

def checkArgs(args):
    assert not (args.oracle_sampling and args.oracle_hypo)
