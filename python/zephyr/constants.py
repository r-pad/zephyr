FEATURE_MEAN = [491.3183898925781, 445.4370422363281, 104.46714782714844, -9.33281421661377, -0.3871251940727234, 15.277295112609863, 96.35192108154297, -21.257665634155273, 244.6168975830078, 454.5150146484375, 452.712890625, 93.33969116210938, 96.92170715332031, 0.4305391013622284, 0.39397644996643066, 0.10189277678728104, 0.2605348229408264, -0.016525093466043472, -0.0010204192949458957, 0.029299981892108917, 0.2030143290758133, -0.03813593089580536, 0.5364543795585632, 0.9343518018722534, 0.9338011741638184, 0.8567478060722351, 0.856406033039093, 14.381174087524414, 39.7275505065918, 0.19100581109523773, 0.2816426157951355, 0.29801905155181885, 32.04430389404297, 15.896907806396484, 0.03398795798420906, 6.787647724151611, 0.43156930804252625, 0.22434939444065094, 0.0131981261074543]

FEATURE_VAR = [104024.1953125, 84253.234375, 10451.4873046875, 7728.88623046875, 0.6716803908348083, 3666.609619140625, 23751.6484375, 3180.625244140625, 57259.7109375, 140852.34375, 82354.09375, 9427.083984375, 9033.4541015625, 0.009170586243271828, 0.009593164548277855, 0.00744457496330142, 0.042461588978767395, 0.015770280733704567, 4.10813800044707e-06, 0.008367528207600117, 0.06150421127676964, 0.004480129573494196, 0.09543216973543167, 0.21136140823364258, 0.005510418675839901, 0.29657745361328125, 0.0726594552397728, 819.4340209960938, 1027.7301025390625, 0.015476326458156109, 0.02970016933977604, 0.036495599895715714, 1238.669677734375, 234.83010864257812, 0.0009443855960853398, 44.11335754394531, 0.014797595329582691, 0.048345230519771576, 9.701783710625023e-06]

FEATURE_LIST = [
    "Number of visible points",
    "Number of points with depth",
    "Number of points close to observed depth",
    "Sum of depth error",
    "Sum of close error",
    "Sum of freespace error",
    "Number of points with freespace error",
    "Sum of occlusion error",
    "Number of points occluded",
    "Sum of L2 color error",
    "Sum of cosine color error",
    "Sum of L2 color error with close depth",
    "Sum of cosine color error with close depth",
    "Percent of model points points visible",
    "Percent of model points with depth",
    "Percent of model points close to observed depth",
    "Percent of model points with depth close to observed depth",
    "Mean of depth error",
    "Mean of close error",
    "Mean of freespace error",
    "Percent of visible points violating freespace",
    "Mean of occlusion error",
    "Percent of visible points occluded",
    "Mean of L2 color error",
    "Mean of cosine color error",
    "Mean of L2 color error with close depth",
    "Mean of cosine color error with close depth ",
]

FEATURE_LIST += [
    "RGB edge score",
    "Depth edge score"
]

FEATURE_LIST += [
    "Mean of Hue Error",
    "Mean of Saturation Error",
    "Mean of Value Error",
]

FEATURE_LIST += [
    "Number of feature matches that are inliers",
    "Number of s features matched to m feature closely",
    "Ratio of s features matched to m feature closely",
    "Sum of feature distances of inlier features",
    "Mean of feature distances of inlier features",
    "Sum of Euclidean distances of inlier features",
    "Mean of Euclidean distances of inlier features",
]

YCBV_TRAIN_SCENE = [
     0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
     24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,41,42,43,44,45,46,47,60,
     61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,
     86
]

YCBV_VALID_SCENE = [40, 85, 87, 88, 89, 90, 91]

YCBV_BOPTEST_SCENE = [48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59]

OBJECT_DIAMETERES = {
    "ycbv":{
        1: 0.172063,
        2: 0.26957299999999995,
        3: 0.198377,
        4: 0.12054300000000001,
        5: 0.196463,
        6: 0.089797,
        7: 0.142543,
        8: 0.114053,
        9: 0.12954,
        10: 0.197796,
        11: 0.259534,
        12: 0.25956599999999996,
        13: 0.161922,
        14: 0.12498999999999999,
        15: 0.22616999999999998,
        16: 0.237299,
        17: 0.20397300000000002,
        18: 0.121365,
        19: 0.174746,
        20: 0.21709399999999998,
        21: 0.10290300000000001,
    },
    "lmo": {
        1: 0.10209900000000001,
        2: 0.247506,
        3: 0.16735499999999998,
        4: 0.17249199999999998,
        5: 0.201404,
        6: 0.154546,
        7: 0.124264,
        8: 0.261472,
        9: 0.108999,
        10: 0.164628,
        11: 0.17588900000000002,
        12: 0.145543,
        13: 0.278078,
        14: 0.282601,
        15: 0.212358,
    }
}