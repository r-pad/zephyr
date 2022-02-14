import argparse

def checkArgs(args):
    assert len(args.dataset_root) == len(args.dataset_name)

def getOptions():
    parser = argparse.ArgumentParser()
    '''Dataset settings'''
    parser.add_argument("--dataset_root", type=str, nargs="+", default=['data/ycb/matches_data_train/'],
        help="paths to the folder containing the dataset. ")
    parser.add_argument("--bop_root", type=str, default="data/bop/",
        help="Path to the BOP datasets root, mainly used for bop_toolkit. ")
    parser.add_argument("--feature_size", type=int, default=11,
        help="The size of the SIFT features. ")
    parser.add_argument("--dataset", type=str, default="feat",
        help="The name of the dataset to be used for scoring model (the featurization of the hypotheses). ")
    parser.add_argument("--no_val", action="store_true",
        help="if set, no validation split and all data will be used for training. ")
    parser.add_argument("--val_obj", type=str, default="even", choices=['odd', 'even'],
        help="Objects with odd or even ID will be used in the validation set. ")
    parser.add_argument("--num_workers", type=int, default=10,
        help="num_workers for DataLoader. ")
    parser.add_argument("--dataset_name", type=str, nargs="+", default=["ycbv"],
        help="the names of the dataset used in testing set. ")
    parser.add_argument("--train_single", action="store_true",
        help="if set, only a single training example will be used. ")

    '''Extra option for loading from a raw BOP dataset'''
    parser.add_argument("--raw_bop_dataset", action="store_true", 
        help="If set, the data will be loaded form the raw BOP dataset instead of a preprocessed dataset")
    parser.add_argument("--ppf_result_file", type=str, default=None,
        help="The path to the PPf result file, used only when raw_bop_dataset")
    parser.add_argument("--split", type=str, default="test", 
        help="BOP dataset splits (train, valid, test), used onlu when raw_bop_dataset")
    parser.add_argument("--split_name", type=str, default="bop_test", 
        help="An extra name to mark bop_test from the full test set, used onlu when raw_bop_dataset")
    

    # parser.add_argument("--bop_dataset", action="store_true",
    #     help='If set, the dataset mode will be on the bop dataset. ')
    parser.add_argument("--test_dataset", action="store_true",
        help="If set, will use getDataloaderBoptest(), otherwise getDatasetsTrain()")
    parser.add_argument("--ppf_only", action="store_true",
        help="If set, only hypotheses from PPF will be used. ")
    parser.add_argument("--sift_only", action="store_true",
        help="If set, only hypotheses from SIFT feature matching will be used. ")
    parser.add_argument("--n_ppf_hypos", type=int, default=100,
        help="The number of the PPF hypotheses. ")
    parser.add_argument("--n_sift_hypos", type=int, default=1000,
        help="The number of the SIFT hypotheses to be used. (SIFT hypotheses after this number will be disgarded. )")
    parser.add_argument("--use_mask_test", action="store_true",
        help="if set, masks of the corresponding type will be used at the test time")
    parser.add_argument("--no_model", action="store_true",
        help="If set, the final predicted scores will not be used but only the order of the hypotheses")

    parser.add_argument("--camera_scale", type=float, default=10000,
        help="If not None, the camera_scale will be modified to be the same to the given value. ")
    parser.add_argument("--norm_downsample", type=int, default=1,
        help="The downsampling ratio when computing normal vectors from depth map. ")

    '''Model settings'''
    parser.add_argument("--model_name", type=str, default='mlp',
        help="The name of the model that will be used for training. ")
    parser.add_argument("--loss", type=str, default="exp",
        help="The name of the loss that will be used. ")
    parser.add_argument("--reg_type", type=str, default='l2',
        help="The type of regularization to be used in training logistic regression and MLP. ")
    parser.add_argument("--reg_weight", type=float, default=0,
        help="(For model_name == lg) If not zero, a L1 regularizer will be added to the logistic regression model")

    '''Training settings'''
    parser.add_argument("--lr", type=float, default=3e-4,
        help="Learning rate. ")
    parser.add_argument("--n_epoch", type=int, default=100,
        help="Number of epoches to run the experiement")
    # parser.add_argument("--gpu_id", type=int, default=0,
    #     help="The GPU ID to be used for the training process")
    parser.add_argument("--n_gpus", type=int, default=1,
        help="Number of GPUs to use to train the model. ")
    parser.add_argument("--update_interval", type=int, default=256,
        help="Interval in iterations between two updates of the visualization. ")
    parser.add_argument("--gradient_interval", type=int, default=32,
        help="Interval in iterations between two gradient descent steps. ")
    parser.add_argument("--no_progress", action="store_true",
        help="If set, no tqdm progress bar will be displayed. ")
    parser.add_argument("--exp_name", type=str, default="exp",
        help="Name of the experiement, used for save figures and models. ")

    parser.add_argument("--debug", action="store_true",
        help="If set, only 1/10 of data will be used for fast debugging. ")
    parser.add_argument("--resume_path", type=str, default=None,
        help="Path to pth file to resume training. ")
    parser.add_argument("--loss_cutoff", type=str, default=None, 
        help="If set, the pose error for loss computation will be set accordingly. ")

    '''Testing settings'''
    parser.add_argument("--oracle_hypo", action="store_true",
        help="Whether to include the oracle hypothesis in testing script. ")
    parser.add_argument("--edge_gpu", type=int, default=None,
        help="If not None, the convolution operation of edge will be on the corresponding GPU. ")
    parser.add_argument("--icp", action="store_true",
        help="If set, a ICP post-processing will be used during validation and testing. ")

    '''feature scoring specific'''
    # parser.add_argument("--feature_inliers", action="store_true",
    #     help="If set, the features related to image feature inliers will also be included in the scoring feature")
    parser.add_argument("--norm_cos_weight", action="store_true",
        help="If set, weight color and depth error according to the cosine of the angle between the normal vector and the vector pointing to the camera. ")
    parser.add_argument("--top_n_feat", type=int, default=None,
        help="If set and selected_features is not specified, top n features on the importance order will be used for training")
    parser.add_argument("--selected_features", type=int, nargs="+", default=None,
        help="The feature index that will be used in ScoreDataset. ")

    '''PointNet specific'''
    parser.add_argument("--pn_pool", type=str, default="max",
        help="The name of type of pooling layer used in PointNetfeat")
    parser.add_argument("--cojitter", type=bool, default=True,
        help="If set, the observed image will be color jittered as data augmentation. ")
    parser.add_argument("--uv_rot", action="store_true",
        help="If set, the uv coordinates will be randomly rotate and flipped as data augmentation. ")
    parser.add_argument("--bottleneck_dim", type=int, default=16,
        help="The dimension of the bottleneck_dim in PointNet (The output per-point dimension of the PointNetfeat)")
    parser.add_argument("--pretrained_pnfeat", type=str, default=None,
        help="the path to the pretrained pointnet feat network. ")
    parser.add_argument("--drop_ratio", type=float, default=0,
        help="ratio of randomly dropping points from the input point cloud. (PointNet++)")
    # parser.add_argument("--use_mask", action="store_true",
    #     help="If set, the mask will be passed and used in the PointNet and PointNet2")
    parser.add_argument("--mask_channel", type=int, nargs="+", default=None,
        help="indices of mask dimension used in PointNet bottleneck. ")
    parser.add_argument("--chunk_size", type=int, default=367,
        help="The chunk size for a single backward pass (To handle out of memory). ")
    parser.add_argument("--max_points", type=int, default=2000,
        help="If not None, the number of points returned by dataset will be at most this number. ")
    parser.add_argument("--max_hypos", type=int, default=None,
        help='If not None, the number of hypotheses will be reduced by randon sampling. ')
    parser.add_argument("--mask_th", type=float, default=None,
        help="If not None, the hypotheses that have percentage of points within given mask lower than the threshold will be rejected regardless. ")
    parser.add_argument("--no_coord", action="store_true",
        help="If set, the xyz channel will not be input as points_channel in PointNet2. ")
    parser.add_argument("--no_valid_proj", action="store_true",
        help="If set, valid_proj will not be used")
    parser.add_argument("--no_valid_depth", action="store_true",
        help="If set, valid_depth will not be used")
    parser.add_argument("--hard_mask", action="store_true", 
        help="If set, the valid proj will be used as a hard mask and to zero out invalid projected points. ")

    parser.add_argument("--inconst_ratio_th", type=float, default=10,
        help="If set, the hypotheses will be filtered according to ratio of inconsistent projected depth. (in percentage)")

    '''DGCNN specific'''
    parser.add_argument("--dgcnn_k", type=int, default=20,
        help="The number of NN used in DGCNN. ")
    # parser.add_argument("--dgcnn_emb_dims", type=int, default=128,
    #     help="The dimension of the embedding vectors of DGCNN. ")
    parser.add_argument("--dgcnn_dropout", action='store_true',
        help="If set, use dropout layer in MLP of DGCNN. ")

    '''Masked Convolution specific'''
    parser.add_argument("--dist_max", default=10,
        help="dist_max input to ConvolutionalPoseModel()")
    parser.add_argument("--masked_pretained", action="store_true",
        help="If set, the pretrained weights of the ResNet will be loaded")
    parser.add_argument("--masked_no_mask", action="store_true",
        help="If set, the unmasked ResNet will be used for maskconv. ")


    return parser
