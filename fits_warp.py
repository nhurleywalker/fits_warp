#!/usr/bin/env python

__author__ = "Paul Hancock and Natasha Hurley-Walker"
__date__ = "05/11/2018"

import numpy as np
import os
from scipy import interpolate
from scipy.interpolate import CloughTocher2DInterpolator
from astropy import wcs
from astropy.io import fits
from astropy.io.votable import parse_single_table
import sys
import glob
import argparse
import psutil

def which(program):
    import os
    def is_exe(fpath):
        return os.path.isfile(fpath) and os.access(fpath, os.X_OK)

    fpath, fname = os.path.split(program)
    if fpath:
        if is_exe(program):
            return program
    else:
        for path in os.environ["PATH"].split(os.pathsep):
            path = path.strip('"')
            exe_file = os.path.join(path, program)
            if is_exe(exe_file):
                return exe_file

    return None

def make_pix_models(fname, ra1='ra', dec1='dec', ra2='RAJ2000', dec2='DEJ2000', fitsname=None, plots=False, smooth=300.):
    """
    Read a fits file which contains the crossmatching results for two catalogues.
    Catalogue 1 is the source catalogue (positions that need to be corrected)
    Catalogue 2 is the reference catalogue (correct positions)
    return rbf models for the ra/dec corrections
    :param fname: Filename
    :param ra1: column name for the ra degrees in catalogue 1 (source)
    :param dec1: column name for the dec degrees in catalogue 1 (source)
    :param ra2: column name for the ra degrees in catalogue 2 (reference)
    :param dec2: column name for the dec degrees in catalogue 2 (reference)
    :param plots: True = Make plots
    :param smooth: smoothing radius (in pixels) for the RBF function
    :return: (dxmodel, dymodel)
    """
    filename, file_extension = os.path.splitext(fname)
    if file_extension == ".fits":
        raw_data = fits.open(fname)[1].data
    elif file_extension == ".vot":
        raw_data = parse_single_table(fname).array

    # get the wcs
    hdr = fits.getheader(fitsname)
    imwcs = wcs.WCS(hdr, naxis=2)

    # filter the data to only include SNR>10 sources
    flux_mask = np.where(raw_data['peak_flux']/raw_data['local_rms']>10)
    data = raw_data[flux_mask]

    #calculate the offsets in the ra/dec directions
    # catalog = Longitude(data[ra1], unit=u.degree), Latitude(data[dec1], unit=u.degree)
    # reference = Longitude(data[ra2], unit=u.degree), Latitude(data[dec2], unit=u.degree)

    cat_xy = imwcs.all_world2pix(zip(data[ra1], data[dec1]), 1)
    ref_xy = imwcs.all_world2pix(zip(data[ra2], data[dec2]), 1)

    diff_xy = ref_xy - cat_xy

    dxmodel = interpolate.Rbf(cat_xy[:, 0], cat_xy[:, 1], diff_xy[:, 0], function='linear', smooth=smooth)
    dymodel = interpolate.Rbf(cat_xy[:, 0], cat_xy[:, 1], diff_xy[:, 1], function='linear', smooth=smooth)

    if plots:
        import matplotlib
# Super-computer-safe
        matplotlib.use('Agg')
        from matplotlib import pyplot
        from matplotlib import gridspec
# Perceptually uniform cyclic color schemes
        try:
            import seaborn as sns
            cmap = matplotlib.colors.ListedColormap(sns.color_palette("husl", 256))
        except ImportError:
            print "seaborne not detected; using hsv color scheme"
            cmap = 'hsv'
# Attractive serif fonts
        if which("latex"):
            try:
                from matplotlib import rc
                rc('text', usetex=True)
                rc('font',**{'family':'serif','serif':['serif']})
            except:
                print "rc not detected; using sans serif fonts"
        else:
                print "latex not detected; using sans serif fonts"
        xmin, xmax = 0, hdr['NAXIS1']
        ymin, ymax = 0, hdr['NAXIS2']

        gx, gy = np.mgrid[xmin:xmax:(xmax-xmin)/50., ymin:ymax:(ymax-ymin)/50.]
        mdx = dxmodel(np.ravel(gx), np.ravel(gy))
        mdy = dymodel(np.ravel(gx), np.ravel(gy))

        x = cat_xy[:, 0]
        y = cat_xy[:, 1]

# plot w.r.t. centre of image, in degrees
        try:
            delX = abs(hdr['CD1_1'])
        except:
            delX = abs(hdr['CDELT1'])
        try:
            delY = hdr['CD2_2']
        except:
            delY = hdr['CDELT2']
# shift all co-ordinates and put them in degrees
        x -= hdr['NAXIS1']/2
        gx -= hdr['NAXIS1']/2
        xmin -= hdr['NAXIS1']/2
        xmax -= hdr['NAXIS1']/2
        x *= delX
        gx *= delX
        xmin *= delX
        xmax *= delX
        y -= hdr['NAXIS2']/2
        gy -= hdr['NAXIS2']/2
        ymin -= hdr['NAXIS2']/2
        ymax -= hdr['NAXIS2']/2
        y *= delY
        gy *= delY
        ymin *= delY
        ymax *= delY
        scale = 1

        dx = diff_xy[:, 0]
        dy = diff_xy[:, 1]

        fig = pyplot.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(100,100)
        gs.update(hspace=0,wspace=0)
        kwargs = {'angles':'xy', 'scale_units':'xy', 'scale':scale, 'cmap':cmap, 'clim':[-180,180]}
        angles = np.degrees(np.arctan2(dy, dx))
        ax = fig.add_subplot(gs[0:100,0:48])
        cax = ax.quiver(x, y, dx, dy, angles, **kwargs)
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        ax.set_xlabel("Distance from pointing centre / degrees")
        ax.set_ylabel("Distance from pointing centre / degrees")
        ax.set_title("Source position offsets / arcsec")
#        cbar = fig.colorbar(cax, orientation='horizontal')

        ax = fig.add_subplot(gs[0:100,49:97])
        cax = ax.quiver(gx, gy, mdx, mdy, np.degrees(np.arctan2(mdy, mdx)), **kwargs)
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        ax.set_xlabel("Distance from pointing centre / degrees")
        ax.tick_params(axis='y',labelleft='off')
        ax.set_title("Model position offsets / arcsec")
#        cbar = fig.colorbar(cax, orientation='vertical')
# Color bar
        ax2 = fig.add_subplot(gs[0:100,98:100])
        cbar3 = pyplot.colorbar(cax,cax=ax2,use_gridspec=True)
        cbar3.set_label('Angle CCW from West / degrees')#,labelpad=-75)
        cbar3.ax.yaxis.set_ticks_position('right')

        outname = os.path.splitext(fname)[0] +'.png'
#        pyplot.show()
        pyplot.savefig(outname, dpi=200)

    return dxmodel, dymodel

def correct_images(fnames, dxmodel, dymodel, suffix):
    """
    Read a list of fits image, and apply pixel-by-pixel corrections based on the
    given x/y models. Interpolate back to a regular grid, and then write an
    output file
    :param fname: input fits file
    :param dxodel: x model
    :param dymodel: x model
    :param fout: output fits file
    :return: None
    """
    # Get co-ordinate system from first image
    im = fits.open(fnames[0])
    data = np.squeeze(im[0].data)
    # create a map of (x,y) pixel pairs as a list of tuples
    xy = np.indices(data.shape, dtype=np.float32)
    xy.shape = (2, xy.shape[1]*xy.shape[2])

    x = np.array(xy[1, :])
    y = np.array(xy[0, :])

    mem = int(psutil.virtual_memory().available * 0.75)
    print "Detected memory ~{0}GB".format(mem/2**30)
# 32-bit floats, bit to byte conversion, MB conversion
    print "Image is {0}MB".format(data.shape[0]*data.shape[1]*32/(8*2**20))
    pixmem = 40000
    print "Allowing {0}kB per pixel".format(pixmem/2**10)
    stride = mem / pixmem
# Make sure it is row-divisible
    stride = (stride//data.shape[0])*data.shape[0]
    print stride, len(x)

    if stride < len(x):
        print "Processing {0} rows at a time".format(stride//data.shape[0])
    else:
        print "Processing entire image at once"

    # calculate the corrections in blocks of 100k since the rbf fails on large blocks
    print 'applying corrections to pixel co-ordinates',
    # remember the largest offset
    maxx = maxy = 0
    if len(x) > stride:
        print 'in cycles'
        n = 0
        borders = range(0, len(x)+1, stride)
        if borders[-1] != len(x):
            borders.append(len(x))
        for s1 in [slice(a, b) for a, b in zip(borders[:-1], borders[1:])]:
            off = dxmodel(x[s1], y[s1])
            maxx = max(np.max(np.abs(off)), maxx)
            x[s1] += off
            # the x coords were just changed so we need to refer back to the original coords
            off = dymodel(xy[1, :][s1], y[s1])
            maxy = max(np.max(np.abs(off)), maxy)
            y[s1] += off
            n += 1
            sys.stdout.write("{0:3.0f}%...".format(100*n/len(borders)))
            sys.stdout.flush()
        print ""
    else:
        print 'all at once'
        x += dxmodel(x, y)
        y += dymodel(xy[1, :], y)

# Note that a potential speed-up would be to nest the file loop inside the
# model calculation loop, so you don't have to calculate the model so many times
# However this would require either:
# 1) holding all the files in memory; but the model calculation is done in a loop
#    in order to reduce memory use, so this would be counter-productive; or:
# 2) writing out partly-complete FITS files and then reading them back in again,
#    which is a bit messy and so has not yet been implemented.
# Fancy splines need about this amount of buffer in order to interpolate the data
# if = 5, differences ~ 10^-6 Jy/beam ; if = 15, differences ~ 10^-9; if = 25, 0
    extra_lines = 25
#    extra_lines = int(max(maxx, maxy)) + 1
# Testing shows that this stage is much less memory intensive, and we can do more
# rows per cycle. However there is very little speed-up from processing with fewer
# rows, so if needed this number can be decreased by factors of 1-10 with no ill
# effect on processing time.
    stride *= 40
    for fname in fnames:
        fout = fname.replace(".fits","_"+suffix+".fits")
        im = fits.open(fname)
        im.writeto(fout, overwrite=True, output_verify='fix+warn')
        oldshape = im[0].data.shape
        data = np.squeeze(im[0].data)
# Replace NaNs with zeroes because nans break the interpolator
        nandices = np.isnan(data)
        data[nandices] = 0.0
        squeezedshape = data.shape
        print 'interpolating', fname
# Note that we need a fresh copy of the data because otherwise we will be trying to
# interpolate over the results of our interpolation
        newdata = np.copy(data)

        if len(x) > stride:
            print "Processing {0} rows at a time".format(stride//data.shape[0])
            n = 0
            borders = range(0, len(x)+1, stride)
            if borders[-1] != len(x):
                borders.append(len(x))
            for a, b in zip(borders, borders[1:]):
                # indexes into the image based on the index into the raveled data
                idx = np.unravel_index(range(a, b), data.shape)
                # model using an extra few lines to avoid edge effects
                bp = min(b + data.shape[0]*extra_lines, len(x))
                # also go backwards by a few lines
                ap = max(0, a - data.shape[0]*extra_lines)
                idxp = np.unravel_index(range(ap, bp), data.shape)
                model = CloughTocher2DInterpolator(np.transpose([x[ap:bp], y[ap:bp]]), np.ravel(data[idxp]),fill_value=-1)
                # evaluate the model over this range
                newdata[idx] = model(xy[1, a:b], xy[0, a:b])
                n += 1
                sys.stdout.write("{0:3.0f}%...".format(100*n/len(borders)))
                sys.stdout.flush()
#                break
            print ""
        else:
            print 'all at once'
            model = CloughTocher2DInterpolator(np.transpose([x, y]), np.ravel(data))
            data = model(xy[1, :], xy[0, :])
            print data.shape
        # Float32 instead of Float64 since the precision is meaningless
        print "int64 -> int32"
        data = np.float32(newdata.reshape(squeezedshape))
        # NaN the edges by 10 pixels to avoid weird edge effects
        print "blanking edges"
        data[0:10, :] = np.nan
        data[:, 0:10] = np.nan
        data[:, -10:data.shape[0]] = np.nan
        data[-10:data.shape[1], :] = np.nan
# Reapply any masking from the original data now that we are done interpolating
        data[nandices] = np.nan
        im[0].data = data.reshape(oldshape)
        print "saving..."
        im.writeto(fout, overwrite=True, output_verify='fix+warn')
        print "wrote", fout
        # Explicitly delete potential memory hogs
        del im, data
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group1 = parser.add_argument_group("input/output files")
    group1.add_argument("--xm", dest='xm', type=str, default=None,
                        help='A .fits binary or VO table. The crossmatch between the reference and source catalogue.')
    group1.add_argument("--infits", dest='infits', type=str, default=None,
                        help="The fits image(s) to be corrected; enclose in quotes for globbing.")
    group1.add_argument("--suffix", dest='suffix', type=str, default=None,
                        help="The suffix to append to rename the output (corrected) fits image(s); e.g., specifying \"warp\" will result in an image like image_warp.fits (no default; if not supplied, no correction will be performed).")
    group2 = parser.add_argument_group("catalog column names")
    group2.add_argument("--ra1", dest='ra1', type=str, default='ra',
                        help="The column name for ra  (degrees) for source catalogue.")
    group2.add_argument("--dec1", dest='dec1', type=str, default='dec',
                        help="The column name for dec (degrees) for source catalogue.")
    group2.add_argument("--ra2", dest='ra2', type=str, default='RAJ2000',
                        help="The column name for ra  (degrees) for reference catalogue.")
    group2.add_argument("--dec2", dest='dec2', type=str, default='DEJ2000',
                        help="The column name for dec (degrees) for reference catalogue.")
    group3 = parser.add_argument_group("Other")
    group3.add_argument('--plot', dest='plot', default=False, action='store_true',
                        help="Plot the offsets and models (default = False)")
    group3.add_argument('--smooth', dest='smooth', default=300.0, type=float,
                        help="Smoothness parameter to give to the radial basis function (default = 300 pix)")

    results = parser.parse_args()

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()

    if results.infits is not None:
        fnames = glob.glob(results.infits) 
        # Use the first image to define the model
        dx, dy = make_pix_models(results.xm, results.ra1, results.dec1, results.ra2, results.dec2,
                                 fnames[0], results.plot, results.smooth)
        if results.suffix is not None:
            # Correct all the images
#            for f in fnames:
            correct_images(fnames, dx, dy, results.suffix)
    else:
        print "No output fits file specified; not doing warping"
