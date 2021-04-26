import numpy as np

from torch.utils.data import DataLoader

from zephyr.data_util import scanDataset
from zephyr.constants import FEATURE_MEAN, FEATURE_VAR, FEATURE_LIST

from zephyr.datasets.collate import default_collate
from zephyr.datasets.score_dataset import ScoreDataset
from zephyr.datasets.render_dataset import RenderDataset
from zephyr.datasets.concat_dataset import ConcatDataset

'''Separate the RGBD features and SIFT features'''
IMPORTANCE_ORDER = [
    28, 27, 32, 33, 36, 35, 29, 16, 26, 22, 13, 4, 26, 21, 22
]

def getDataloaderBoptest(args, Dataset, dataset_root, dataset_name):
    print("Initializing %s dataset from %s" % (dataset_name, dataset_root))
    datapoints = scanDataset(base_path = dataset_root, split="all")
    datapoints = np.asarray(datapoints)
    if args.debug:
        datapoints = datapoints[::20]
    print("Using BOP dataset format. Total dataset:", len(datapoints))
    boptest_loader = DataLoader(
        Dataset(datapoints, dataset_root=dataset_root, dataset_name=dataset_name, args=args, mode='test'),
        batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=default_collate
        )
    unseen_oids = []
    loaders = [boptest_loader]
    return loaders, unseen_oids

def getDatasetsTrain(args, Dataset, dataset_root, dataset_name):
    print("Initializing %s dataset from %s" % (dataset_name, dataset_root))
    datapoints = scanDataset(base_path = dataset_root, split="train")
    datapoints = np.asarray(datapoints)

    '''Split the train and valid dataset'''
    object_ids = np.arange(1, 22)
    if args.no_val:
        unseen_oids = []
    # elif args.train_single_object is not None:
    #     unseen_oids = [id for id in object_ids if id != args.train_single_object]
    #     print("object Ids for training:", args.train_single_object)
    else:
        if dataset_name == "ycbv":
            if args.val_obj == "odd":
                unseen_oids = [id for id in object_ids if id%2==1]
            elif args.val_obj == "even":
                unseen_oids = [id for id in object_ids if id%2==0]
        elif dataset_name == "lmo":
            unseen_oids = [1, 5, 6, 8, 9, 10, 11, 12]

    print("Object IDs for validation:", unseen_oids)

    '''get data loader from dataset class'''
    # Sample a small portion from the seen object scenes as the seen validation set
    train_set = [_ for _ in datapoints if _[0] not in unseen_oids]
    permute_train_set = np.random.permutation(train_set)
    N = len(permute_train_set)
    train_set = permute_train_set[:int(.9*N)]

    if args.no_val:
        valid_set = permute_train_set[int(.9*N):]
    else:
        # Sample a portion from the unseen objects as the unseen validation set
        valid_set = [_ for _ in datapoints if _[0] in unseen_oids]
        permute_valid_set = np.random.permutation(valid_set)

        valid_set = np.concatenate([permute_valid_set[:int(len(permute_train_set)*0.4)], permute_train_set[int(.9*N):]])

    if args.debug:
        train_set = train_set[::100]
        valid_set = valid_set[::100]
    if args.train_single:
        train_set = train_set[:1]
        valid_set = valid_set[:1]

    print("Dataset: train %d, valid %d" % (len(train_set), len(valid_set)))

    dataset_train = Dataset(train_set, dataset_root=dataset_root, dataset_name=dataset_name, args=args, mode='train')
    dataset_valid = Dataset(valid_set, dataset_root=dataset_root, dataset_name=dataset_name, args=args, mode='valid')
    datasets = [dataset_train, dataset_valid]
    return datasets, unseen_oids

def getDataloader(args):
    Dataset = ScoreDataset

    '''Get the datasets and dataloaders according to dataset format'''
    if args.test_dataset:
        assert len(args.dataset_root) == 1
        # This returns a single dataset containing the test cases in bop test set
        loaders, unseen_oids = getDataloaderBoptest(args, Dataset, args.dataset_root[0], args.dataset_name[0])
        args.unseen_oids = [unseen_oids]
    else:
        datasets_all = []
        unseen_oids_all = []
        for dataset_i in range(len(args.dataset_root)):
            dataset_root = args.dataset_root[dataset_i]
            dataset_name = args.dataset_name[dataset_i]
            # This returns two laoders: one for training set, and another for validation set
            datasets, unseen_oids = getDatasetsTrain(args, Dataset, dataset_root, dataset_name)
            datasets_all.append(datasets)
            unseen_oids_all.append(unseen_oids)

        # datasets_all will be in the form of [[train1, val1], [train2, val2], [train3, val3]]
        # Concate all datasets of different roots into one respectively
        datasets = [ConcatDataset(_) for _ in zip(*datasets_all)]
        loaders = [
            DataLoader(datasets[0], batch_size=1, shuffle=True, num_workers=args.num_workers, collate_fn=default_collate),
            DataLoader(datasets[1], batch_size=1, shuffle=False, num_workers=args.num_workers, collate_fn=default_collate)
        ]
        args.unseen_oids = unseen_oids_all

    '''Set data mean and variance'''
    if args.dataset == "feat" or 'mix' in args.dataset:
        for loader in loaders:
            loader.dataset.setNormalization(FEATURE_VAR, FEATURE_MEAN)

    '''Set extra informatio needed in the args'''
    args.dim_agg = loaders[0].dataset.dim_agg
    args.dim_point = loaders[0].dataset.dim_point
    args.dim_render = loaders[0].dataset.dim_render
    if 'mix' in args.dataset:
        args.extra_bottleneck_dim = args.dim_agg
        print("extra_bottleneck_dim:", args.extra_bottleneck_dim)
    else:
        args.extra_bottleneck_dim = 0
    print("dim_agg:", args.dim_agg, "dim_point:", args.dim_point)

    return loaders
