import numpy as np
import cv2

def quadraticPeakInterp(histograms, num_bins=36):
    hist_blur = histograms*(6.0/16) \
                + (np.roll(histograms, 1, axis=-1) + np.roll(histograms, -1, axis=-1)) * (4.0/16) \
                + (np.roll(histograms, 2, axis=-1) + np.roll(histograms, -2, axis=-1)) * (1.0/16)

    peaks = hist_blur > 0.9 * np.expand_dims(hist_blur.max(axis=-1), -1)

    angles = (np.arange(num_bins+2)-0.5)*360/num_bins
    x1 = angles[:-2]
    x2 = angles[1:-1]
    x3 = angles[2:]

    y1 = np.roll(hist_blur, 1, axis=-1)
    y2 = hist_blur
    y3 = np.roll(hist_blur, -1, axis=-1)

    denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
    A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
    B = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom;

    with np.errstate(divide='ignore', invalid='ignore'):
        interp_angle = -B / (2*A);
    quad_peak = ((y2 >= y1) * (y2 >= y3)) * peaks

    flat_peaks = ((y2 == y1) * (y2 == y3)) * peaks
    interp_angle[flat_peaks] = x2[np.nonzero(flat_peaks)[-1]]
    return interp_angle, quad_peak

def computeHOGPeaks(img, cell_size,
                    step_size = None,
                    blur_size = None,
                    sigma = None,
                    num_bins = 36,
                    mask=None):
    # cell_size = 2*cell_size + 1
    # print("computeHOGPeaks")
    # print("cell_size:", cell_size)

    if(type(cell_size) not in [tuple, list, np.ndarray]):
        cell_size = (cell_size, cell_size)

    if(step_size is None):
        step_size = cell_size
    elif(type(step_size) not in [tuple, list, np.ndarray]):
        step_size = (step_size, step_size)

    if(sigma is None):
        sigma = 1.5*cell_size[0]

    if(blur_size is None):
        blur_size = (cell_size[0]//4)*2+1

    if(len(img.shape) == 3 and img.shape[2] == 3):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    win_size = (img.shape[0] - (img.shape[0]-cell_size[0]) % step_size[0],
                img.shape[1] - (img.shape[1]-cell_size[1]) % step_size[1])


    img_blur = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    hog = cv2.HOGDescriptor(_winSize=(win_size[1], win_size[0]),
                            _blockSize=(cell_size[1], cell_size[0]),
                            _blockStride=(step_size[1], step_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=num_bins,
                            #_derivAperture=-1,
                            _winSigma=sigma,
                            #_histogramNormType=0,
                            _L2HysThreshold=1.0,
                            _gammaCorrection=True,
                            #_nlevels=64,
                            _signedGradient=True,
                            )

    n_cells = (int((win_size[0]-cell_size[0])/step_size[0] + 1),
               int((win_size[1]-cell_size[1])/step_size[1] + 1))

    hog_feats = hog.compute(img_blur)
    hog_feats = hog_feats\
                   .reshape(n_cells[1], n_cells[0], num_bins) \
                   .transpose((1, 0, 2))

    interp_angle, quad_peak = quadraticPeakInterp(hog_feats, num_bins)
    y_grid, x_grid = np.mgrid[:n_cells[0], :n_cells[1]]
    x_grid = x_grid*step_size[1] + cell_size[1]//2
    y_grid = y_grid*step_size[0] + cell_size[0]//2

    dense_kps = []
    for u in range(n_cells[0]):
        for v in range(n_cells[1]):
            x = x_grid[u,v]
            y = y_grid[u,v]
            if(mask is not None and not mask[y,x]):
                continue
            for th in np.nonzero(quad_peak[u,v])[0]:
                dense_kps.append(cv2.KeyPoint(x = x, y = y, _size = cell_size[0],
                                              _angle = interp_angle[u,v,th]))

    return dense_kps

def computeHOGKeypoints(img, cell_size,
                        keypoints,
                        blur_size = None,
                        sigma = None,
                        num_bins = 36):
    # cell_size = 2*cell_size + 1
    # print("computeHOGKeypoints")
    # print("cell_size:", cell_size)

    if(type(cell_size) not in [tuple, list, np.ndarray]):
        cell_size = (cell_size, cell_size)

    if(blur_size is None):
        blur_size = (cell_size[0]//4)*2+1

    if(sigma is None):
        sigma = 1.5*cell_size[0]

    if(len(img.shape) == 3 and img.shape[2] == 3):
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    img_blur = cv2.GaussianBlur(img, (blur_size, blur_size), 0)
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1], img.shape[0]),
                            _blockSize=(cell_size[1], cell_size[0]),
                            _blockStride=(1, 1),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=num_bins,
                            #_derivAperture=-1,
                            _winSigma=sigma,
                            #_histogramNormType=0,
                            _L2HysThreshold=1.0,
                            _gammaCorrection=True,
                            #_nlevels=64,
                            _signedGradient=True,
                            )

    n_cells = (img.shape[0]-cell_size[0] + 1,
               img.shape[1]-cell_size[1] + 1)

    hog_feats = hog.compute(img_blur)
    hog_feats = hog_feats\
                   .reshape(n_cells[1], n_cells[0], num_bins) \
                   .transpose((1, 0, 2))

    interp_angle, quad_peak = quadraticPeakInterp(hog_feats, num_bins)
    y_grid, x_grid = np.mgrid[:n_cells[0], :n_cells[1]]
    x_grid = x_grid + cell_size[1]//2
    y_grid = y_grid + cell_size[0]//2

    hog_kps = []
    kp_indices = []
    for j, kp in enumerate(keypoints):
        x,y = kp.pt
        u = int(y-cell_size[0]//2)
        v = int(x-cell_size[1]//2)
        if(u < 0 or v < 0 or u >= n_cells[0] or v >= n_cells[1]):
            continue

        for th in np.nonzero(quad_peak[u,v])[0]:
            kp_indices.append(j)
            hog_kps.append(cv2.KeyPoint(x = x,
                                        y = y,
                                        _size = cell_size[0],
                                        _angle = interp_angle[u,v,th]))

    return hog_kps, kp_indices

def deepcopy_keypoint (kp):
    return cv2.KeyPoint(x = kp.pt[0], y = kp.pt[1],
                        _size = kp.size, _angle = kp.angle,
                        _response = kp.response, _octave = kp.octave,
                        _class_id = kp.class_id)

def orient_keypoints(image, keypoints, radius, num_bins = 36,
                     return_histograms = False,
                     return_indices = False):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    patch_size = 2*radius+1
    dy = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=patch_size)
    dx = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=patch_size)

    mag = np.sqrt(dx**2 + dy**2)
    ori = np.arctan2(dy, dx)
    ori[ori < 0] += 2*np.pi

    w = cv2.getGaussianKernel(patch_size, 1.5*patch_size)
    w = w.dot(w.T)

    height, width = gray.shape

    new_keypoints = []
    histograms = []
    indices = []
    for j, kp in enumerate(keypoints):
        x, y = kp.pt
        y = int(y)
        x = int(x)
        if(y < radius or y >= height-radius-1 or \
           x < radius or x >= width-radius-1):
            histograms.append(None)
            continue
        th = ori[(y-radius):(y+radius+1), (x-radius):(x+radius+1)]
        m = mag[(y-radius):(y+radius+1), (x-radius):(x+radius+1)]
        hist_raw, bin_edges = np.histogram(th, bins=np.linspace(0, 2*np.pi, num_bins+1), weights=m*w)

        # Smooth the histogram according to OpenCV implementation
        hist = hist_raw*(6.0/16) + \
               (np.roll(hist_raw, 1) + np.roll(hist_raw, -1)) * (4.0/16) + \
               (np.roll(hist_raw, 2) + np.roll(hist_raw, -2)) * (1.0/16)

        max_loc = np.argmax(hist)
        max_val = np.max(hist)

        thresh = 0.9*max_val
        if(return_histograms):
            histograms.append(hist_raw)

        for peak_loc in np.nonzero(hist > thresh)[0]:
            peak_neighborhood = peak_loc + np.array([-1,0,1])

            x1,x2,x3 = (peak_neighborhood + .5)*360/num_bins
            y1,y2,y3 = hist[np.remainder(peak_neighborhood, num_bins)]

            # Must be peak
            if(y2 < y1 or y2 < y3):
                continue

            # Quadratic Peak Fit
            denom = (x1 - x2) * (x1 - x3) * (x2 - x3)
            A = (x3 * (y2 - y1) + x2 * (y1 - y3) + x1 * (y3 - y2)) / denom
            B = (x3*x3 * (y1 - y2) + x2*x2 * (y3 - y1) + x1*x1 * (y2 - y3)) / denom;

            with np.errstate(divide='ignore'):
                interp_angle = -B / (2*A);

            new_kp = deepcopy_keypoint(kp)
            new_kp.size = patch_size
            new_kp.angle = interp_angle
            new_kp.response = hist[peak_loc]
            new_keypoints.append(new_kp)
            if(return_indices):
                indices.append(j)

    if(return_histograms or return_indices):
        res = [new_keypoints]
        if(return_histograms):
            res.append(histograms)
        if(return_indices):
            res.append(indices)
        return tuple(res)

    return new_keypoints
