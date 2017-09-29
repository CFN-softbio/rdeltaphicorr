import numpy as np
from scipy.ndimage.filters import gaussian_filter
from skbeam.core.accumulators.binned_statistic import BinnedStatistic1D,\
    BinnedStatistic2D
from skbeam.core.accumulators.binned_statistic import RPhiBinnedStatistic,\
    RadialBinnedStatistic
from skbeam.core.utils import angle_grid, radial_grid

# this function just makes a nice status bar, not necessary
try:
    from tqdm import tqdm
    tqdm_loaded = True
except ImportError:
    tqdm_loaded = False


''' This code relates all angular correlation related code.
'''


# TODO : moake a 1D version (version that allows 1D correlations)
def mkrbinstat(shape, origin, r_map=None, mask=None, statistic='mean',
               bins=800):
    ''' Make the radial binned statistic. Allow for
        an rmap to be supplied
    '''
    if origin is None:
        origin = (shape[0] - 1) / 2, (shape[1] - 1) / 2

    if r_map is None:
        r_map = radial_grid(origin, shape)

    if mask is not None:
        if mask.shape != shape:
            raise ValueError('"mask" has incorrect shape. '
                             ' Expected: ' + str(shape) +
                             ' Received: ' + str(mask.shape))
        mask = mask.reshape(-1)

    rbinstat = BinnedStatistic1D(r_map.reshape(-1), statistic, bins=bins,
                                 mask=mask,
                                 )
    return rbinstat


def mkrphibinstat(shape, origin, r_map=None, mask=None, statistic='mean',
                  bins=(800, 300)):
    ''' Make the radial phi binned statistic. Allow for an rmap to be supplied
    '''
    if origin is None:
        origin = (shape[0] - 1) / 2., (shape[1] - 1) / 2.

    if r_map is None:
        r_map = radial_grid(origin, shape)

    phi_map = angle_grid(origin, shape)

    shape = tuple(shape)
    if mask is not None:
        if mask.shape != shape:
            raise ValueError('"mask" has incorrect shape. '
                             ' Expected: ' + str(shape) +
                             ' Received: ' + str(mask.shape))
        mask = mask.reshape(-1)

    rphibinstat = BinnedStatistic2D(r_map.reshape(-1), phi_map.reshape(-1),
                                    statistic, bins=bins, mask=mask)
    return rphibinstat


class RDeltaPhiCorrelator:
    ''' An angular correlator.

        This routine takes images and correlates them in angle, delta phi.  It
        takes a set of images and converts them to rphi maps (according to
        binning).  Finally, each of these qphi maps it converts to an angular
        correlation.

        This does it in two steps:
            1. Compute the average image from the imgs
            2. Run the average image subtracted angular correlation of each
            frame.

        You need to specify:
            - imgs : a sequence of images, needs to be an iterable, which
            - imgsb : (optional) if specified, will compute angular
            correlations for different sets of frames
                NOTE : these will be randomly offset. This is really either
                meant for comparison of two images (run(imga, imgb)) or
                comparison between images where the offset is known to be
                constant)
            returns a numpy array
                for ex: imgs[0].shape finds the shape
                    len(imgs) gives number of images
            - rbins : 1 number is the number of rs
            - phibins : 1 number is the number of phis
            - r0 : the center (defaults to center of image)

        It is recommended to use some background subtraction method for the
        correlations to reduce error.
    '''
    # TODO : Allow for a second set of mask for the second image in the
    # correlation maybe? (may not b eimportant)
    def __init__(self, shape,  origin=None, mask=None, maskb=None, rbins=800,
                 phibins=360, method='bgsub', saverphis=None, PF=True,
                 sigma=None, r_map=None):
        ''' The initialization routine for the delta phi correlation object

            Parameters
            ----------
            shape : 2 tuple
                the shape of the images
                for ex: imgs[0].shape finds the shape
                    len(imgs) gives number of images

            origin : 2 tuple, optional
                the center (can be float), defaults to image center

            mask : 2d np.ndarray, optional
                The mask for the images

            method : str, optional
                The averaging method. There are currently 3:
                    'bgsub' : subtract the average image of the image series in
                    the correlation
                    'bgest' : estimate the average by averaging each ring of r
                    'symavg' : perform a symmetric averaging
                    None : no averaging

                These may be combined in a list. ex: ['bgsub', 'symavg']

                Note : When using a mask, it is highly recommended to use one
                of the averaging methods to suppress errors.

            rbins : int or list of floats, optional
                This specifies the bins in radius (in units of pixels)
                There are two cases:
                    - If it is an int, the image will be partitioned into that
                      number of bins
                    - If it is a list of floats, the floats will be treated as
                    the edges of the bins used for the partitioning (in pixels)
                Defaults to 800

            phibins : int or list of floats, optional
                There are two cases:
                    - If it is an int, the image will be partitioned into that
                      number of bins
                    - If it is a list of floats, the floats will be treated as
                    the edges of the bins used for the partitioning (in pixels)
                Defaults to 360

            PF : bool, optional
                Print Flag. If True, prints verbose status of the computations.

            sigma : float, optional
                the sigma for the smoothing kernel over the images
                Defaults to None (no smoothing)

            saverphis : bool, optional
                Decide whether to save the rqphi map per image
                If True, after running correlations, an array of rphis is
                saved, as obj.rphis where obj is the reference to this object.

            Returns
            -------
            An RDeltaPhiCorrelator instance.
        '''
        # TODO : add different methods
        self.PF = PF
        self.saverphis = saverphis
        self.method = method

        # the smoothing kernel, set to None to disactivate smoothing
        self.sigma = sigma

        # set up the binned statistic
        self.shape = shape

        if origin is None:
            origin = (self.shape[0] - 1)/2, (self.shape[1] - 1)/2

        # need this to compute the counts for the deltaphi correlation
        if mask is None:
            self.mask = np.ones(self.shape)
        else:
            self.mask = mask

        if maskb is not None:
            self.maskb = maskb
        else:
            # this costs no memory etc...
            self.maskb = self.mask

        # the statistics binner
        # TODO : here we'd need two rphibinstats for two masks
        # self.rphibinstat = mkrphibinstat(self.shape, bins=(rbins, phibins),
                                          #origin=origin, statistic='mean',
                                          #mask=mask, r_map=r_map)
        self.rphibinstat = RPhiBinnedStatistic(self.shape,
                                               bins=(rbins, phibins),
                                               origin=origin, statistic='mean',
                                               mask=mask, r_map=r_map)
        self.rphimask = self.rphibinstat(self.mask)
        self._removenans(self.rphimask)

        if maskb is not None:
            #self.rphibinstatb = mkrphibinstat(self.shape, bins=(rbins,
                                                                 #phibins),
                                                    #origin=origin,
                                                    #statistic='mean',
                                                    #mask=maskb, r_map=r_map)
            self.rphibinstatb = RPhiBinnedStatistic(self.shape, bins=(rbins,
                                                    phibins),
                                                    origin=origin,
                                                    statistic='mean',
                                                    mask=maskb, r_map=r_map)
            self.rphimaskb = self.rphibinstat(self.maskb)
            self._removenans(self.rphimaskb)
        else:
            self.rphibinstatb = self.rphibinstat
            self.rphimaskb = self.rphimask

        # need this to properly count pixels
        self.rdeltaphimask = _convol1d(self.rphimask, axis=-1)
        if maskb is not None:
            self.rdeltaphimaskb = _convol1d(self.rphimaskb, axis=-1)
        else:
            self.rdeltaphimaskb = self.rdeltaphimask

        # get number of pixels per (r,phi) wedge
        # if there are two masks, take product, else just use first mask
        self.rphibinstat.statistic = 'sum'
        self.rphimaskcnts = self.rphibinstat(self.mask)
        self._removenans(self.rphimaskcnts)
        self.rphibinstat.statistic = 'mean'

        if maskb is None:
            self.rphibinstatb.statistic = 'sum'
            self.rphimaskcntsb = self.rphibinstatb(self.maskb)
            self._removenans(self.rphimaskcntsb)
            self.rphibinstatb.statistic = 'mean'

        # where the first moment's number per bin is nonzero
        self.wsel1 = np.where(self.rphimaskcnts > .1)

        # needs to be maskb not self.maskb (already defined)
        if maskb is not None:
            self.wsel1b = np.where(self.rphimaskcntsb > .1)

        # this just finds non zero entry (most of the time this selects
        # everything) allow for numerical errors here from convolution (i.e.
        # using an FFT)
        self.wsel2 = np.where(self.rdeltaphimask*self.rdeltaphimaskb > .1)

        # number of radii and phi
        self.numrs = len(self.rphibinstat.bin_centers[0])
        self.numphis = len(self.rphibinstat.bin_centers[1])

        # save the bin centers, approx (ravg, phiavg)
        self.rvals, self.phivals = self.rphibinstat.bin_centers[0], \
            self.rphibinstat.bin_centers[1]

        # it's relative so just set first points to zero
        self.phivals = self.phivals - self.phivals[0]
        # the degree version
        self.phivalsd = np.degrees(self.phivals)

        # for computing S(q) and S2(q) during the process
        #self.rbinstat = mkrbinstat(self.shape, rbins, origin=origin,
                                              #statistic='mean', mask=mask,
                                    #r_map=r_map)
        self.rbinstat = RadialBinnedStatistic(self.shape, rbins, origin=origin,
                                              statistic='mean', mask=mask,
                                              r_map=r_map)


        # the counts per r bin, no need to use 'sum' this time
        self.Ircnts = self.rbinstat.flatcount

        # this initializes the accumulators for the correlations
        # back to zero
        self.clear_state()

    def clear_state(self):
        # initialize the accumulators for running average and variance
        # for rdeltaphiavg etc
        self.rdphivaravg_state = None
        self.rdphivaravg2_state = None

    def set_method(self, method):
        ''' Set the method to a different method.
        '''
        self.method = method

    def _removenans(self, data):
        w = np.where(np.isnan(data))
        if len(w[0]) > 0:
            data[w] = 0

    def _removeinfs(self, data):
        w = np.where(np.isinf(data))
        if len(w[0]) > 0:
            data[w] = 0

    def _deltaphi_symmetricaverage(self, rphi, rphib=None, rphimask=None,
                                   rphimaskb=None, wsel2=None):
        ''' Compute the qphi correlation assuming mask.
            Try the symmetric average.
            Ib : the second image to compare to

            M : the mask
            Mb : the mask for the second image

            Note : this method could result in divide by zero errors (np.nans).
            Do not use on low count rate data.
            wsel2 : the non zero pixels from the mask in the 2nd moment
                Using this speeds up recurrent calls to this function
        '''
        # this speeds up by not having to recalculate nonzero entries due to
        # mask
        if rphimask is None:
            rphimask = np.ones_like(rphi)
        if rphimaskb is None:
            rphimaskb = np.ones_like(rphimask)

        if rphib is None:
            rphib = rphi

        # this is an attempt to speed up code, the non zero pixels are assumed
        # to be already found from parent object
        if wsel2 is None:
            wsel2 = self.wsel2

        MMK = _convol1d(rphi*rphimask, rphimaskb, axis=-1)
        # in the case of same I same M this is just the reverse of MMK
        MMKp = _convol1d(rphimask, rphib*rphimaskb, axis=-1)
        II = _convol1d(rphi*rphimask, rphib*rphimaskb, axis=-1)
        MM = _convol1d(rphimask, rphimaskb, axis=-1)
        # w = np.where((MM != 0)*(MMK!=0)*(MMKp !=0))
        II[wsel2] *= MM[wsel2]/MMK[wsel2]/MMKp[wsel2]
        # since this is a normalization approach, multiply by S(q) as well
        II[wsel2[0]] *= self.Ir[wsel2[0]][:, np.newaxis] * \
            self.Irb[wsel2[0]][:, np.newaxis]

        return II

    def _deltaphi_corr(self, rphi, rphib=None, rphimask=None, rphimaskb=None,
                       bgest=False, wsel1=None, wsel2=None):
        ''' The regular delta phi correlation, just a convolution,
                normalized by mask.
            wsel1 : the non zero pixels from the mask in the 1st moment
                (average) Using this speeds up recurrent calls to this function
            wsel2 : the non zero pixels from the mask in the 2nd moment Using
                this speeds up recurrent calls to this function
        '''
        if wsel1 is None:
            wsel1 = self.wsel1
        if wsel2 is None:
            wsel2 = self.wsel2
        if rphimask is None:
            rphimask = np.ones_like(rphi)
        if rphib is None:
            rphib = rphi

        if bgest:
            bgestvals = np.sum(rphi, axis=-1) /\
                    np.sum(self.rphimask, axis=-1)
            self._removeinfs(bgestvals)
            rphi = (rphi - bgestvals[:, np.newaxis])*rphimask
            if rphib is None:
                rphib = rphi
            else:
                bgestvals = np.sum(rphib, axis=-1) /\
                        np.sum(self.rphimask, axis=-1)
                self._removeinfs(bgestvals)
                rphib = (rphib - bgestvals[:, np.newaxis])*rphimask

        rdeltaphi = _convol1d(rphi, rphib, axis=-1)
        rdeltaphi[wsel2] = \
            rdeltaphi[wsel2]/self.rdeltaphimask[wsel2]

        return rdeltaphi

    def deltaphicorrelate(self, rphi, rphib=None, rphimask=None,
                          rphimaskb=None, method='bgsub', wsel1=None,
                          wsel2=None):
        ''' Correlate an rphi map

            Parameters
            ----------
            rphi : the rphi map to correlate
            rphib : the rphi map to correlate against (if None, sets equal to
            rphi)

            rphimask : the mask of the correlation map
                if set to None, none is used

            wsel1 : the non zero pixels from the mask in the 1st moment
                (average) Using this speeds up recurrent calls to this function

            wsel2 : the non zero pixels from the mask in the 2nd moment Using
                this speeds up recurrent calls to this function

        '''
        # perform symmetric averaging
        if 'symavg' in method:
            rdeltaphi = self._deltaphi_symmetricaverage(rphi, rphib=rphib,
                                                        rphimask=rphimask,
                                                        rphimaskb=rphimaskb,
                                                        wsel2=wsel2)
        else:
            rdeltaphi = self._deltaphi_corr(rphi, rphib=rphib,
                                            rphimask=rphimask,
                                            rphimaskb=rphimaskb, wsel1=wsel1,
                                            wsel2=wsel2)

        return rdeltaphi

    def run(self, imgs, imgsb=None):
        self(imgs, imgsb=imgsb)

    def __call__(self, imgs, imgsb=None):
        ''' Runs the correlations.

            Parameters
            ----------
            imgs : one image or a sequence of images, needs to be an iterable,

            imgsb : optional, a second set of images to correlate with
            which returns a numpy array
                imgs[0].ndim should give the number of dimensions of an image
                for ex: imgs[0].shape finds the shape
                len(imgs) gives number of images

            Notes
            -----
            Calculation is a two step process. First iterate to obtain average
            image, then calculate correlations. Reason is that some methods
            require average image before hand.
        '''
        # saving rphis is necessary if the method is bgest
        if 'bgest' in self.method and self.saverphis is False:
            raise ValueError("Error, rphis not set to True. Can't use "+\
                             "method bgest")

        # convention:
        # b (ex: imgsb) means the second image batch to compare to
        # 2 (ex: avgimg2 = <img^2>) means the square of the image
        if imgs[0].ndim == 1:
            # it's just one image
            imgs = imgs[np.newaxis, :, :]
        # read first slice
        self.imgs = imgs
        self.nimgs = len(self.imgs)

        if imgsb is None:
            compute_imgb = False
            self.imgsb = imgs  # just a reference, no copy
        else:
            if imgsb[0].ndim == 1:
                # it's just one image
                imgsb = imgsb[np.newaxis, :, :]
            compute_imgb = True
            self.imgsb = imgsb

        # compute average image
        print("Computing a running average")
        self.avgimg, self.avgimg2, \
            self.ivsn = _runningaverage(imgs, PF=self.PF, mask=self.mask,
                                        sigma=self.sigma)
        if compute_imgb:
            print("Computing a running average of the second set of images")
            self.avgimgb, self.avgimg2b, \
                self.ivsnb = _runningaverage(imgs, PF=self.PF, mask=self.maskb,
                                             sigma=self.sigma)
        else:
            self.avgimgb = self.avgimg
            self.avgimg2b = self.avgimg2
            self.ivsnb = self.ivsn

        # self.rvals is the domain
        self.Ir = self.rbinstat(self.avgimg)
        self.Ir2 = self.rbinstat(self.avgimg2)
        self.Irvar = np.sqrt(self.Ir2-self.Ir**2)

        if compute_imgb:
            self.Irb = self.rbinstat(self.avgimgb)
            self.Ir2b = self.rbinstat(self.avgimg2b)
            self.Irvarb = np.sqrt(self.Ir2b-self.Irb**2)
        else:
            self.Irb = self.Ir
            self.Ir2b = self.Ir2
            self.Irvarb = self.Irvar

        self.rphiavg = self.rphibinstat(self.avgimg)
        self._removenans(self.rphiavg)
        self.rphiavg2 = self.rphibinstat(self.avgimg2)
        self._removenans(self.rphiavg2)
        if compute_imgb is not None:
            self.rphiavgb = self.rphibinstat(self.avgimgb)
            self._removenans(self.rphiavgb)
            self.rphiavg2b = self.rphibinstat(self.avgimg2b)
            self._removenans(self.rphiavg2b)
        else:
            self.rphiavgb = self.rphiavg
            self.rphiavg2b = self.rphiavg2

        # mostly for debuggins, save the rphi images
        if self.saverphis:
            self.rphis = np.zeros((self.nimgs, self.numrs, self.numphis))
            self.rphis2 = np.zeros((self.nimgs, self.numrs, self.numphis))
            if compute_imgb is not None:
                self.rphisb = np.zeros((self.nimgs, self.numrs, self.numphis))
                self.rphis2b = np.zeros((self.nimgs, self.numrs, self.numphis))

        self.rdeltaphiavg = np.zeros((self.numrs, self.numphis))
        self.rdeltaphiavg2 = np.zeros((self.numrs, self.numphis))
        # errors in correlations
        self.rdeltaphivar = np.zeros((self.numrs, self.numphis))
        self.rdeltaphivar2 = np.zeros((self.numrs, self.numphis))

        print("Reading rphis")
        # should check if tqdm is available before printing this
        # also should add a print flag (ignore print if not desired)
        if tqdm_loaded and self.PF:
            rangeiter = tqdm(range(self.nimgs))
        else:
            rangeiter = range(self.nimgs)

        for i in rangeiter:
            if self.PF and not tqdm_loaded:
                print("Computing rphi and rdeltaphi, "
                      "{} of {}".format(i+1, self.nimgs))

            img = self.imgs[i]
            imgb = self.imgsb[i]
            # only bg sub if more than one image, else it's ignored
            if 'bgsub' in self.method and len(self.imgs) > 1:
                # copy them to avoid modifying original array
                img = np.copy(img) - self.avgimg
                img2 = (self.imgs[i] - self.avgimg)**2
                if compute_imgb:
                    imgb = np.copy(imgb) - self.avgimgb
                    img2b = (self.imgsb[i] - self.avgimg2b)**2
                else:
                    # since img, img2 were copied, need to redfine imgb
                    imgb = img
                    img2b = img2
            else:
                img2 = img**2
                img2b = imgb**2

            rphi = self.rphibinstat(img)
            rphi2 = self.rphibinstat(img2)
            self._removenans(rphi)
            self._removenans(rphi2)
            if compute_imgb:
                rphib = self.rphibinstat(self.imgsb[i])
                rphi2b = self.rphibinstat(img2b)
                self._removenans(rphib)
                self._removenans(rphi2b)
            else:
                rphib = rphi
                rphi2b = rphi2

            if self.saverphis:
                self.rphis[i] = rphi
                self.rphis2[i] = rphi2
                if compute_imgb:
                    self.rphisb[i] = rphib
                    self.rphis2b[i] = rphi2b


            rdeltaphi = self.deltaphicorrelate(rphi, rphib=rphib,
                                               rphimask=self.rphimask,
                                               rphimaskb=self.rphimaskb,
                                               method=self.method, wsel1=None,
                                               wsel2=None)

            rdeltaphi2 = self.deltaphicorrelate(rphi2, rphimask=self.rphimask,
                                                rphib=rphi2b,
                                                rphimaskb=self.rphimaskb,
                                                method=self.method, wsel1=None,
                                                wsel2=None)

            self.rdphivaravg_state =\
                _running_var_avg(rdeltaphi[self.wsel2][np.newaxis, :],
                                 prev=self.rdphivaravg_state, mask=None)
            self.rdphivaravg2_state = \
                _running_var_avg(rdeltaphi2[self.wsel2][np.newaxis, :],
                                 prev=self.rdphivaravg2_state, mask=None)

        # take the accumulator's current state and compute the moments
        # of the distributions
        n, mean, M2 = self.rdphivaravg_state
        self.rdeltaphiavg[self.wsel2] = mean
        self.rdeltaphivar[self.wsel2] = np.sqrt(M2/(n-1))
        # the variance of the mean itself is reduced by another factor of
        # sqrt(nimgs)
        self.rdeltaphivar /= np.sqrt(n)
        n, mean, M2 = self.rdphivaravg2_state
        self.rdeltaphiavg2[self.wsel2] = mean
        self.rdeltaphivar2[self.wsel2] = np.sqrt(M2/(n-1))
        # the variance of the mean itself is reduced by another factor of
        # sqrt(nimgs)
        self.rdeltaphivar2 /= np.sqrt(n)

        # compute the normalized version (for better viewing)
        self.rdeltaphiavg_n = \
            self.safe_norm(self.rdeltaphiavg,
                           self.rdeltaphiavg[:, 1][:, np.newaxis])
        self.rdeltaphiavg2_n = \
            self.safe_norm(self.rdeltaphiavg2,
                           self.rdeltaphiavg2[:, 1][:, np.newaxis])
        self.rdeltaphivar_n = \
            self.safe_norm(self.rdeltaphivar,
                           self.rdeltaphivar[:, 1][:, np.newaxis])
        self.rdeltaphivar2_n = \
            self.safe_norm(self.rdeltaphivar2,
                           self.rdeltaphivar2[:, 1][:, np.newaxis])

        print("Done. Computed rphi, rdeltaphi")

    def estbgsub(self, rphis, rphimask):
        # estimate background and subtract from rphis using mask
        bgestvals = np.sum(rphis, axis=-1) /\
                np.sum(rphimask, axis=-1)[np.newaxis, :]
        self._removeinfs(bgestvals)

        # make a copy, don't subtract from array
        rphis = (rphis - bgestvals[:, :, np.newaxis]) * \
            rphimask[np.newaxis, :, :]

        return rphis

    def safe_norm(self, rdeltaphi, norm):
        ''' Normalize the rdeltaphi function by row,
            on first axis.
            row must have same length as axis on rdeltaphi
            returns a copy of the array
        '''
        # in case it's a 1 row slice
        norm = np.broadcast_to(norm, rdeltaphi.shape)
        w = np.where((norm != 0)*(~np.isnan(norm)))
        rdeltaphin = np.zeros(rdeltaphi.shape)
        rdeltaphin[w] = rdeltaphi[w]/norm[w]

        return rdeltaphin


''' Helper Functions '''


def _running_var_avg(data, prev=None, mask=None):
    ''' compute a running variance and average.
        Note : this is slower than other methods (since there
        is a division each time), but it allows accumulating.

        data : the data, can be any shape
            It *must* be of form data[iterdim, ...] where iterdim is adimension
            to iterate on

        prev : previous state:
            n, mean, M2
        mask : a boolean mask (or indices into the data arrays)
            useful when data is an iterable
        Returns
            n, mean, M2
            n is the nth number in sequence
            M2 is the partial variance (variance is sqrt(M2/(n-1)))
            mean is the mean
        ex:
            _runing_var_avg(data[:10], n=0)

    '''
    # TODO : make _runningaverage use this
    # for data of length 1
    if not isinstance(data, np.ndarray):
        data = np.array(data)

    # If 0 or 1, give more dimensions so we can fake index
    if data.ndim == 0:
        data = data[np.newaxis, np.newaxis]
    elif data.ndim == 1:
        data = data[:, np.newaxis]

    if mask is None:
        mask = np.ones_like(data[0], dtype=bool)

    wp = np.where(mask == 1)

    if prev is None:
        n = 0
        mean = np.zeros_like(data[0], dtype=float)
        M2 = np.zeros_like(data[0], dtype=float)
    else:
        n, mean, M2 = prev

    # print("Performing a running average/variance calculation")
    # assumes x is an np array
    for x in data:
        n += 1
        delta = x[wp] - mean[wp]
        mean[wp] += delta/n
        delta2 = x[wp] - mean[wp]
        M2[wp] += delta*delta2
    # NOTE: This won't make sense until n > 2
    return n, mean, M2


def _runningaverage(imgs, PF=True, sigma=None, mask=None):
    '''Running average using an image reader.
        See Knuth's book.

        It's a more efficient way to average.
        Iavg_[n-1] = sum(Iavg,n-1)/(n-1)
        Iavg_n = ( (Iavg_[n-1])*(n-1) + I_n)/n
        Iavg_n = Iavg_[n-1] + (I_n - Iavg_[n-1])/n
        rng must be a range iterator
    '''
    rng = range(len(imgs))
    if mask is not None:
        numpixels = np.sum(mask)

    avgimg = np.zeros(imgs[0].shape)
    avgimg2 = np.zeros(imgs[0].shape)

    ivsn = np.zeros(len(imgs))

    if tqdm_loaded and PF:
        rangeiter = tqdm(range(len(rng)))
    else:
        rangeiter = range(len(rng))

    for i in rangeiter:
        rngno = rng[i]
        # note : if sigma is None, _smooth function *must* bypass smoothing,
        # else make two for loops one with one without
        imgn = _smooth2Dgauss(imgs[rngno], sigma=sigma, mask=mask)
        avgimg += (imgn - avgimg)/float(i+1)
        avgimg2 += (imgn**2 - avgimg2)/float(i+1)
        if mask is not None:
            ivsn[i] = np.sum(mask*imgn)/numpixels
        else:
            ivsn[i] = np.average(imgn)

    return avgimg, avgimg2, ivsn


def _smooth2Dgauss(img, mask=None, sigma=30):
    ''' Smooth an image in 2D according to the mask.
        sigma: the sigma to smooth by
    '''
    if mask is None:
        mask = np.ones_like(img)
    if sigma is None:
        return img
    imgf = np.copy(img)
    imgg = gaussian_filter(img*mask, sigma)
    imgmsk = gaussian_filter(mask*1., sigma)
    imgmskrat = gaussian_filter(mask*0.+1, sigma)
    w = np.where(imgmsk > 0)
    imgf[w] = imgg[w]/imgmsk[w]*imgmskrat[w]
    return imgf


def _convol1d(a, b=None, axis=-1):
    ''' convolve a with b. If b not specified, perform
        a self convolution.'''
    from numpy.fft import fft, ifft
    if(b is None):
        b = a
    return ifft(fft(a, axis=axis)*np.conj(fft(b, axis=axis)), axis=axis).real
