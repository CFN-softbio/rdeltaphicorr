Examples
========

QuickStart
----------
Running a simple correlator::
 # assuming some image
 img = ...
 mask = ...
 from rdeltaphicorr.rdpc import RDeltaPhiCorrelator
 rdpc = RDeltaPhiCorrelator(img.shape, mask=mask, origin=origin)
 rdpc.run(img)

 # plotting
 import matplotlib.pyplot as plt
 plt.figure()
 plt.imshow(rdpc.rdeltaphiavg_n)
 plt.clim(0,1)


Averaging
---------

There are different types of averaging you may be interested in.

Symmetric Averaging
*******************

``rdpc = RDeltaPhiCorrelator(img.shape, mask=mask, origin=origin,
method='symavg')``
Symmetric averaging is most useful when large amounts of data are masked or
missing.


