#! /usr/bin/env python
from __future__ import print_function
import numpy as np
import os
import textwrap
from scipy import interpolate
from scipy.interpolate import CloughTocher2DInterpolator
import astropy
from astropy import wcs
from astropy.io import fits
from astropy.io.votable import parse_single_table
from astropy.coordinates import SkyCoord, Angle, Latitude, Longitude, SkyOffsetFrame
from astropy.table import Table, hstack
import astropy.units as u
import os
import sys
import glob
import argparse
import psutil
from tqdm import tqdm
from time import time, sleep

# Parallelise the code
import multiprocessing

__author__ = ["Natasha Hurley-Walker", "Paul Hancock"]
__date__ = "2019-08-08"


def is_exe(fpath):
    return os.path.isfile(fpath) and os.access(fpath, os.X_OK)


def which(program):
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


def make_pix_models(
    fname,
    ra1="ra",
    dec1="dec",
    ra2="RAJ2000",
    dec2="DEJ2000",
    fitsname=None,
    plots=False,
    smooth=300.0,
    sigcol=None,
    noisecol=None,
    SNR=10,
    latex=False,
    max_sources=None,
):
    """
    Read a fits file which contains the crossmatching results for two catalogues.
    Catalogue 1 is the source catalogue (positions that need to be corrected)
    Catalogue 2 is the reference catalogue (correct positions)
    return rbf models for the ra/dec corrections
    :param fname: filename for the crossmatched catalogue
    :param ra1: column name for the ra degrees in catalogue 1 (source)
    :param dec1: column name for the dec degrees in catalogue 1 (source)
    :param ra2: column name for the ra degrees in catalogue 2 (reference)
    :param dec2: column name for the dec degrees in catalogue 2 (reference)
    :param fitsname: fitsimage upon which the pixel models will be based
    :param plots: True = Make plots
    :param smooth: smoothing radius (in pixels) for the RBF function
    :param max_sources: Maximum number of sources to include in the construction of the warping model (defaults to None, use all sources)
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
    if sigcol is not None and noisecol is not None:
        flux_mask = np.where(raw_data[sigcol] / raw_data[noisecol] > SNR)
        data = raw_data[flux_mask]
    else:
        data = raw_data

    if max_sources is not None:
        # argsort goes in ascending order, so select from the end
        sort_idx = np.argsort(data[sigcol])[0][-max_sources:]
        data = data[sort_idx]
        print("Selected {0} brightest sources".format(max_sources))

    print("Using {0} sources to construct the pixel offset model".format(len(data)))

    start = time()

    cat_xy = imwcs.all_world2pix(list(zip(data[ra1], data[dec1])), 1)
    ref_xy = imwcs.all_world2pix(list(zip(data[ra2], data[dec2])), 1)

    diff_xy = ref_xy - cat_xy

    global dxmodel
    dxmodel = interpolate.Rbf(
        cat_xy[:, 0], cat_xy[:, 1], diff_xy[:, 0], function="linear", smooth=smooth
    )
    global dymodel
    dymodel = interpolate.Rbf(
        cat_xy[:, 0], cat_xy[:, 1], diff_xy[:, 1], function="linear", smooth=smooth
    )

    print("Model created in {0} seconds".format(time() - start))

    if plots:
        import matplotlib

        # Super-computer-safe
        matplotlib.use("Agg")
        from matplotlib import pyplot
        from matplotlib import gridspec

        # Perceptually uniform cyclic color schemes
        try:
            import seaborn as sns

            cmap = matplotlib.colors.ListedColormap(sns.color_palette("husl", 256))
        except ImportError:
            print("seaborne not detected; using hsv color scheme")
            cmap = "hsv"
        # Attractive serif fonts
        if latex is True:
            if which("latex"):
                try:
                    from matplotlib import rc

                    rc("text", usetex=True)
                    rc("font", **{"family": "serif", "serif": ["serif"]})
                except:
                    print("rc not detected; using sans serif fonts")
            else:
                print("latex not detected; using sans serif fonts")
        xmin, xmax = 0, hdr["NAXIS1"]
        ymin, ymax = 0, hdr["NAXIS2"]

        gx, gy = np.mgrid[
            xmin : xmax : (xmax - xmin) / 50.0, ymin : ymax : (ymax - ymin) / 50.0
        ]
        mdx = dxmodel(np.ravel(gx), np.ravel(gy))
        mdy = dymodel(np.ravel(gx), np.ravel(gy))
        x = cat_xy[:, 0]
        y = cat_xy[:, 1]

        # plot w.r.t. centre of image, in degrees
        try:
            delX = abs(hdr["CD1_1"])
        except:
            delX = abs(hdr["CDELT1"])
        try:
            delY = hdr["CD2_2"]
        except:
            delY = hdr["CDELT2"]
        # shift all co-ordinates and put them in degrees
        x -= hdr["NAXIS1"] / 2
        gx -= hdr["NAXIS1"] / 2
        xmin -= hdr["NAXIS1"] / 2
        xmax -= hdr["NAXIS1"] / 2
        x *= delX
        gx *= delX
        xmin *= delX
        xmax *= delX
        y -= hdr["NAXIS2"] / 2
        gy -= hdr["NAXIS2"] / 2
        ymin -= hdr["NAXIS2"] / 2
        ymax -= hdr["NAXIS2"] / 2
        y *= delY
        gy *= delY
        ymin *= delY
        ymax *= delY
        scale = 1

        dx = diff_xy[:, 0]
        dy = diff_xy[:, 1]

        fig = pyplot.figure(figsize=(12, 6))
        gs = gridspec.GridSpec(100, 100)
        gs.update(hspace=0, wspace=0)
        kwargs = {
            "angles": "xy",
            "scale_units": "xy",
            "scale": scale,
            "cmap": cmap,
            "clim": [-180, 180],
        }
        angles = np.degrees(np.arctan2(dy, dx))
        ax = fig.add_subplot(gs[0:100, 0:48])
        cax = ax.quiver(x, y, dx, dy, angles, **kwargs)
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        ax.set_xlabel("Distance from pointing centre / degrees")
        ax.set_ylabel("Distance from pointing centre / degrees")
        ax.set_title("Source position offsets / arcsec")
        #        cbar = fig.colorbar(cax, orientation='horizontal')

        ax = fig.add_subplot(gs[0:100, 49:97])
        cax = ax.quiver(gx, gy, mdx, mdy, np.degrees(np.arctan2(mdy, mdx)), **kwargs)
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
        ax.set_xlabel("Distance from pointing centre / degrees")
        ax.tick_params(axis="y", labelleft="off")
        ax.set_title("Model position offsets / arcsec")
        #        cbar = fig.colorbar(cax, orientation='vertical')
        # Color bar
        ax2 = fig.add_subplot(gs[0:100, 98:100])
        cbar3 = pyplot.colorbar(cax, cax=ax2, use_gridspec=True)
        cbar3.set_label("Angle CCW from West / degrees")  # ,labelpad=-75)
        cbar3.ax.yaxis.set_ticks_position("right")

        outname = os.path.splitext(fname)[0] + ".png"
        #        pyplot.show()
        pyplot.savefig(outname, dpi=200)


def apply_interp(index, x1, y1, x2, y2, data):
    model = CloughTocher2DInterpolator(
        np.transpose([x1, y1]), np.ravel(data), fill_value=-1
    )
    # evaluate the model over this range
    newdata = model(x2, y2)
    return index, newdata


def pool_fit_dxmodel(index, x, y):
    off = dxmodel(x, y)
    return index, off


def pool_fit_dymodel(index, x, y):
    off = dymodel(x, y)
    return index, off


def _fmm(args):
    """
    A shallow wrapper for apply_interp

    Parameters
    ----------
    args : list
        A list of arguments for apply_interp

    Returns
    -------
    None
    """
    # an easier to debug traceback when multiprocessing
    # thanks to https://stackoverflow.com/a/16618842/1710603
    try:
        return apply_interp(*args)
    except:
        import traceback

        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def _fmx(args):
    """
    A shallow wrapper for pool_fit_dxmodel

    Parameters
    ----------
    args : list
        A list of arguments for pool_fit_dxmodel

    Returns
    -------
    None
    """
    # an easier to debug traceback when multiprocessing
    # thanks to https://stackoverflow.com/a/16618842/1710603
    try:
        return pool_fit_dxmodel(*args)
    except:
        import traceback

        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def _fmy(args):
    """
    A shallow wrapper for pool_fit_dymodel

    Parameters
    ----------
    args : list
        A list of arguments for pool_fit_dymodel

    Returns
    -------
    None
    """
    # an easier to debug traceback when multiprocessing
    # thanks to https://stackoverflow.com/a/16618842/1710603
    try:
        return pool_fit_dymodel(*args)
    except:
        import traceback

        raise Exception("".join(traceback.format_exception(*sys.exc_info())))


def multiprocess_progress(cores, func, args, progress, tqdm_desc=None):
    """
    Provide a single function to decide how to do the pool work submission,
    and wrap with a progress bar if requested
    """
    # start a new process for each task
    with multiprocessing.Pool(processes=cores, maxtasksperchild=1) as pool:
        try:
            if progress:
                results = [
                    i
                    for i in tqdm(
                        pool.imap(func, args, chunksize=1),
                        total=len(args),
                        desc=tqdm_desc,
                    )
                ]
                sleep(4)  # might help with a join timeout error?
            else:
                print("Running {0} stage".format(tqdm_desc))
                results = pool.map_async(func, args, chunksize=1).get(timeout=10000000)

        except KeyboardInterrupt:
            pool.close()
            sys.exit(1)

    return results


def correct_images(fnames, suffix, testimage=False, cores=1, vm=None, progress=False):
    """
    Read a list of fits image, and apply pixel-by-pixel corrections based on the
    given x/y models, which are global variables defined earlier.
    Interpolate back to a regular grid, and then write output files.
    :param fname: input fits file
    :param fout: output fits file
    :param vm: Assume this many GBs are available, which is used when computing stride lengths
    :param progress: use tqdm to provide a progress bar
    :return: None
    """
    # Get co-ordinate system from first image
    # Do not open images at this stage, to save memory
    hdr = fits.getheader(fnames[0])
    nx = hdr["NAXIS1"]
    ny = hdr["NAXIS2"]

    xy = np.indices((ny, nx), dtype=np.float32)
    xy.shape = (2, nx * ny)

    x = np.array(xy[1, :])
    y = np.array(xy[0, :])

    if vm is None:
        mem = int(psutil.virtual_memory().available * 0.75)
    else:
        mem = int(vm * 1e9)  # GB to B
    print("Detected memory ~{0}GB".format(mem / 2 ** 30))
    # 32-bit floats, bit to byte conversion, MB conversion
    print("Image is {0}MB".format(nx * ny * 32 / (8 * 2 ** 20)))
    pixmem = 20000
    print("Allowing {0}kB per pixel".format(pixmem / 2 ** 10))
    stride = mem / pixmem
    # Make sure stride is row-divisible
    stride = (stride // ny) * ny
    # Generally if this is true, the images are so small that you don't want to waste
    # system time setting up a big complex operation
    if len(x) < stride:
        print("Calculating all pixel offsets at once.")
        x += dxmodel(x, y)
        y += dymodel(xy[1, :], y)
    else:
        if cores == 1:
            print(
                "Applying corrections to pixel co-ordinates {0} rows at a time, using a single core".format(
                    stride // ny
                )
            )
            n = 0
            borders = list(range(0, len(x) + 1, int(stride)))
            if borders[-1] != len(x):
                borders.append(len(x))
            for s1 in [slice(a, b) for a, b in zip(borders[:-1], borders[1:])]:
                off = dxmodel(x[s1], y[s1])
                x[s1] += off
                # the x coords were just changed so we need to refer back to the original coords
                off = dymodel(xy[1, :][s1], y[s1])
                y[s1] += off
                n += 1
                sys.stdout.write("{0:3.0f}%...".format(100 * n / len(borders)))
                sys.stdout.flush()
            print("")
        else:
            # Ensure each core gets enough memory
            stride /= cores

            # When using a pool, you need to send the arguments as a tuple through the
            # wrapping function, so we set up the arguments here before running the pool
            args = []
            print(
                "Applying corrections to pixel co-ordinates {0} rows at a time across {1} cores".format(
                    stride // ny, cores
                )
            )
            borders = list(range(0, len(x) + 1, int(stride)))
            if borders[-1] != len(x):
                borders.append(len(x))
            # We need to order our arguments by an index, since the results will be returned
            # in whatever order the pooled tasks finish
            n = 0
            for s1 in [slice(a, b) for a, b in zip(borders[:-1], borders[1:])]:
                args.append((n, xy[1, s1], y[s1]))
                n += 1

            # x-offsets first
            results = multiprocess_progress(
                cores, _fmx, args, progress, tqdm_desc="x-offsets"
            )

            print("Reordering x-offset results")
            indices, offsets = map(list, zip(*results))
            # Order correctly
            ind = np.argsort(indices)
            offsets = np.array(offsets, dtype="object")
            offsets = offsets[ind]
            # Flatten list of lists
            o = [item for sublist in offsets for item in sublist]

            # o = np.concatenate([sublist for sublist in offsets])
            # Make into array and apply
            x += np.array(o)

            # y-offsets are performed now
            results = multiprocess_progress(
                cores, _fmy, args, progress, tqdm_desc="y-offsets"
            )

            print("Reordering y-offset results")
            indices, offsets = map(list, zip(*results))
            # Order correctly
            ind = np.argsort(indices)
            offsets = np.array(offsets, dtype="object")
            offsets = offsets[ind]
            # Flatten list of lists
            o = [item for sublist in offsets for item in sublist]
            # Make into array and apply
            y += np.array(o)

    if testimage is True:
        print("Creating divergence maps")
        start = time()
        # Save the divergence as a fits image
        im = fits.open(fnames[0])
        outputname = fnames[0].replace(".fits", "")
        div = (
            np.gradient((x - np.array(xy[1, :])).reshape((nx, ny)))[0]
            + np.gradient((y - np.array(xy[0, :])).reshape((nx, ny)))[1]
        )
        im[0].data = div
        im.writeto(outputname + "_div.fits", overwrite=True)
        im[0].data = (x - np.array(xy[1, :])).reshape((nx, ny))
        im.writeto(outputname + "_delx.fits", overwrite=True)
        im[0].data = (y - np.array(xy[0, :])).reshape((nx, ny))
        im.writeto(outputname + "_dely.fits", overwrite=True)
        print("finished divergence map in {0} seconds".format(time() - start))

        del im

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
    # Testing shows that this stage is less memory intensive, and we can do more
    # rows per cycle. For a single core there is little speed-up from processing with fewer
    # rows, so if needed this number can be decreased by factors of 1-10 with no ill
    # effect on processing time.
    # However, for multi-core processing we want this number as large as possible
    # without going OOM.
    stride *= 40

    for fname in fnames:
        fout = fname.replace(".fits", "_" + suffix + ".fits")
        im = fits.open(fname)
        im.writeto(fout, overwrite=True, output_verify="fix+warn")
        oldshape = im[0].data.shape
        data = im[0].data
        unsqueezedshape = data.shape
        data = np.squeeze(data)
        squeezedshape = data.shape
        # Replace NaNs with zeroes because otherwise it breaks the interpolation
        nandices = np.isnan(data)
        data[nandices] = 0.0
        print("interpolating {0}".format(fname))
        # We have the "all at once" option inside the cores=1 option this time, because
        # even on a high-memory system, it is faster to parallelise the interpolation than
        # to rely on scipy to do it all at once
        if cores == 1:
            if len(x) > stride:
                print(
                    "Interpolating {0} rows at a time using a single core".format(
                        stride // data.shape[0]
                    )
                )
                # We need a fresh copy of the data because otherwise we will be trying to
                # interpolate over the results of our interpolation
                newdata = np.copy(data)
                print("Remapping data", end="")
                n = 0
                borders = range(0, len(x) + 1, stride)
                if borders[-1] != len(x):
                    borders.append(len(x))
                for a, b in zip(borders, borders[1:]):
                    # indexes into the image based on the index into the raveled data
                    idx = np.unravel_index(range(a, b), data.shape)
                    # model using an extra few lines to avoid edge effects
                    bp = min(b + data.shape[0] * extra_lines, len(x))
                    # also go backwards by a few lines
                    ap = max(0, a - data.shape[0] * extra_lines)
                    idxp = np.unravel_index(range(ap, bp), data.shape)
                    model = CloughTocher2DInterpolator(
                        np.transpose([x[ap:bp], y[ap:bp]]),
                        np.ravel(data[idxp]),
                        fill_value=-1,
                    )
                    # evaluate the model over this range
                    newdata[idx] = model(xy[1, a:b], xy[0, a:b])
                    n += 1
                    sys.stdout.write("{0:3.0f}%...".format(100 * n / len(borders)))
                    sys.stdout.flush()
                print("")
                # Float32 instead of Float64 since the precision is meaningless
                print("int64 -> int32")
                data = np.float32(newdata)
            else:
                print("all at once")
                model = CloughTocher2DInterpolator(np.transpose([x, y]), np.ravel(data))
                newdata = model(xy[1, :], xy[0, :])
                # Float32 instead of Float64 since the precision is meaningless
                print("int64 -> int32")
                data = np.float32(newdata)
        else:
            # Testing shows that larger strides go OOM for the parallel version
            stride /= 4
            # Make sure it is row-divisible
            stride = (stride // ny) * ny
            print(
                "Interpolating {0} rows at a time across {1} cores".format(
                    stride // ny, cores
                )
            )
            args = []
            n = 0
            borders = list(range(0, len(x) + 1, int(stride)))
            if borders[-1] != len(x):
                borders.append(len(x))
            for a, b in zip(borders, borders[1:]):
                # indexes into the image based on the index into the raveled data
                idx = np.unravel_index(range(a, b), data.shape)
                # model using an extra few lines to avoid edge effects
                bp = min(b + data.shape[0] * extra_lines, len(x))
                # also go backwards by a few lines
                ap = max(0, a - data.shape[0] * extra_lines)
                idxp = np.unravel_index(range(ap, bp), data.shape)
                #                # evaluate the model over this range
                #                calc(x1 = x[ap:bp], y1 = y[ap:bp], data = data[idxp], x2 = xy[1, a:b], y2 = xy[0, a:b])
                args.append((n, x[ap:bp], y[ap:bp], xy[1, a:b], xy[0, a:b], data[idxp]))
                n += 1

            #    # start a new process for each task
            results = multiprocess_progress(
                cores, _fmm, args, progress, tqdm_desc="Sky interpolation"
            )

            indices, pixvals = map(list, zip(*results))
            # Order correctly
            ind = np.argsort(indices)
            pixvals = np.array(pixvals, dtype="object")
            pixvals = pixvals[ind]
            # Flatten list of lists
            n = [item for sublist in pixvals for item in sublist]
            # Float32 instead of Float64 since the precision is meaningless
            print("int64 -> int32")
            newdata = np.array(n, dtype="float32")

        data = newdata.reshape(squeezedshape)
        # NaN the edges by 10 pixels to avoid weird edge effects
        print("blanking edges")
        data[0:10, :] = np.nan
        data[:, 0:10] = np.nan
        data[:, -10 : data.shape[0]] = np.nan
        data[-10 : data.shape[1], :] = np.nan
        # Re-apply any previous NaN mask to the data
        data[nandices] = np.nan
        im[0].data = data.reshape(oldshape)
        print("saving...")
        im.writeto(fout, overwrite=True, output_verify="fix+warn")
        print("wrote {0}".format(fout))
        # Explicitly delete potential memory hogs
        del im, data
    return


def warped_xmatch(
    incat=None,
    refcat=None,
    ra1="ra",
    dec1="dec",
    ra2="RAJ2000",
    dec2="DEJ2000",
    radius=2 / 60.0,
):
    """
    Create a cross match solution between two catalogues that accounts for bulk shifts and image warping.
    The warping is done in pixel coordinates, not sky coordinates.

    :param image: Fits image containing the WCS info for sky->pix conversion (Ideally the image which was used
                  to create incat.
    :param incat: The input catalogue which is to be warped during the cross matching process.
    :param ref_cat: The reference image which will remain unwarped during the cross matching process
    :param ra1, dec1: column names for ra/dec in the input catalogue
    :param ra2, dec2: column names for ra/dec in the reference catalogue
    :param radius: initial matching radius in degrees
    :return:
    """
    # check for incat/refcat as as strings, and load the file if it is
    incat = Table.read(incat)
    refcat = Table.read(refcat)

    target_cat = SkyCoord(
        incat[ra1], incat[dec1], unit=(u.degree, u.degree), frame="icrs"
    )
    ref_cat = SkyCoord(
        refcat[ra2], refcat[dec2], unit=(u.degree, u.degree), frame="icrs"
    )

    center = SkyOffsetFrame(
        origin=SkyCoord(np.mean(target_cat.ra), np.mean(target_cat.dec), frame="icrs")
    )

    tcat_offset = target_cat.transform_to(center)
    rcat_offset = ref_cat.transform_to(center)

    # crossmatch the two catalogs
    idx, dist, _ = tcat_offset.match_to_catalog_sky(rcat_offset)

    # accept only matches within radius
    distance_mask = np.where(dist.degree < radius)  # this mask is into tcat_offset
    match_mask = idx[distance_mask]  # this mask is into rcat_offset
    print(len(match_mask))

    # calculate the ra/dec shifts
    dlon = rcat_offset.lon[match_mask] - tcat_offset.lon[distance_mask]
    dlat = rcat_offset.lat[match_mask] - tcat_offset.lat[distance_mask]

    # remake the offset catalogue with the bulk shift included
    tcat_offset = SkyCoord(
        tcat_offset.lon + np.mean(dlon), tcat_offset.lat + np.mean(dlat), frame=center
    )

    # now do this again 3 more times but using the Rbf
    for i in range(3):
        # crossmatch the two catalogs
        idx, dist, _ = tcat_offset.match_to_catalog_sky(rcat_offset)
        # accept only matches within radius
        distance_mask = np.where(dist.degree < radius)  # this mask is into cat
        match_mask = idx[distance_mask]  # this mask is into tcat_offset
        if len(match_mask) < 1:
            break

        # calculate the ra/dec shifts
        dlon = (
            rcat_offset.lon.degree[match_mask] - tcat_offset.lon.degree[distance_mask]
        )
        dlat = (
            rcat_offset.lat.degree[match_mask] - tcat_offset.lat.degree[distance_mask]
        )

        # use the following to make some models of the offsets
        dlonmodel = interpolate.Rbf(
            tcat_offset.lon.degree[distance_mask],
            tcat_offset.lat.degree[distance_mask],
            dlon,
            function="linear",
            smooth=3,
        )
        dlatmodel = interpolate.Rbf(
            tcat_offset.lon.degree[distance_mask],
            tcat_offset.lat.degree[distance_mask],
            dlat,
            function="linear",
            smooth=3,
        )

        # remake/update the tcat_offset with this new model.
        tcat_offset = SkyCoord(
            tcat_offset.lon
            + dlonmodel(tcat_offset.lon.degree, tcat_offset.lat.degree) * u.degree,
            tcat_offset.lat
            + dlatmodel(tcat_offset.lon.degree, tcat_offset.lat.degree) * u.degree,
            frame=center,
        )

    # final crossmatch to make the xmatch file
    idx, dist, _ = tcat_offset.match_to_catalog_sky(rcat_offset)
    # accept only matches within radius
    distance_mask = np.where(dist.degree < radius)  # this mask is into cat
    match_mask = idx[distance_mask]  # this mask is into tcat_offset
    # print("Final mask {0}".format(len(match_mask)))
    xmatch = hstack([incat[distance_mask], refcat[match_mask]])

    # return a warped version of the target catalogue and the final cross matched table
    tcat_corrected = tcat_offset.transform_to(target_cat)
    incat[ra1] = tcat_corrected.ra.degree
    incat[dec1] = tcat_corrected.dec.degree
    return incat, xmatch


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    group1 = parser.add_argument_group("Warping input/output files")
    group1.add_argument(
        "--xm",
        dest="xm",
        type=str,
        default=None,
        help="A .fits binary or VO table. The crossmatch between the reference and source catalogue.",
    )
    group1.add_argument(
        "--infits",
        dest="infits",
        type=str,
        default=None,
        help="The fits image(s) to be corrected; enclose in quotes for globbing.",
    )
    group1.add_argument(
        "--suffix",
        dest="suffix",
        type=str,
        default=None,
        help='The suffix to append to rename the output (corrected) fits image(s); e.g., specifying "warp" will result in an image like image_warp.fits (no default; if not supplied, no correction will be performed).',
    )
    group2 = parser.add_argument_group("catalog column names")
    group2.add_argument(
        "--ra1",
        dest="ra1",
        type=str,
        default="ra",
        help="The column name for ra  (degrees) for source catalogue.",
    )
    group2.add_argument(
        "--dec1",
        dest="dec1",
        type=str,
        default="dec",
        help="The column name for dec (degrees) for source catalogue.",
    )
    group2.add_argument(
        "--ra2",
        dest="ra2",
        type=str,
        default="RAJ2000",
        help="The column name for ra  (degrees) for reference catalogue.",
    )
    group2.add_argument(
        "--dec2",
        dest="dec2",
        type=str,
        default="DEJ2000",
        help="The column name for dec (degrees) for reference catalogue.",
    )
    group3 = parser.add_argument_group("Other")
    group3.add_argument(
        "--plot",
        dest="plot",
        default=False,
        action="store_true",
        help="Plot the offsets and models (default = False)",
    )
    group3.add_argument(
        "--latex",
        dest="latex",
        default=False,
        action="store_true",
        help="Use LaTeX to make nicer axis labels for plots (default = False)",
    )
    group3.add_argument(
        "--testimage",
        dest="testimage",
        default=False,
        action="store_true",
        help="Generate pixel-by-pixel delta_x, delta_y, and divergence maps (default = False)",
    )
    group3.add_argument(
        "--smooth",
        dest="smooth",
        default=300.0,
        type=float,
        help="Smoothness parameter to give to the radial basis function (default = 300 pix)",
    )
    group3.add_argument(
        "--signal",
        dest="sigcol",
        default=None,
        type=str,
        help="Column from which to get the signal for a signal-to-noise cut (e.g. peak_flux) (no default; if not supplied, cut will not be performed",
    )
    group3.add_argument(
        "--noise",
        dest="noisecol",
        default=None,
        type=str,
        help="Column from which to get the noise for a signal-to-noise cut (e.g. local_rms) (no default; if not supplied, cut will not be performed",
    )
    group3.add_argument(
        "--SNR",
        dest="SNR",
        default=10,
        type=float,
        help="Signal-to-noise ratio for a signal-to-noise cut (default = 10)",
    )
    group3.add_argument(
        "--cores",
        dest="cores",
        default=None,
        type=int,
        help="NUmber of cores to use (default = autodetect",
    )
    group3.add_argument(
        "--vm",
        default=None,
        type=float,
        help="Attempt to restrict interpolation algorithms to this man GBs. Data is split based on a stride length computed internally. ",
    )
    group3.add_argument(
        "--nsrcs",
        default=None,
        type=int,
        help="Maximum number of sources used when constructing the distortion model. Default behaviour will use all available matches. ",
    )
    group3.add_argument(
        "--progress",
        default=False,
        action="store_true",
        help="Provide a progress bar for stages that are distributed into work-units",
    )
    group4 = parser.add_argument_group("Crossmatching input/output files")
    group4.add_argument(
        "--incat",
        dest="incat",
        type=str,
        default=None,
        help="Input catalogue to be warped.",
    )
    group4.add_argument(
        "--refcat",
        dest="refcat",
        type=str,
        default=None,
        help="Input catalogue to be warped.",
    )
    group4.add_argument(
        "--xmcat",
        dest="xm",
        type=str,
        default=None,
        help="Output cross match catalogue",
    )
    group4.add_argument(
        "--corrected",
        dest="corrected",
        type=str,
        default=None,
        help="Output corrected version of input catalogue",
    )

    group5 = parser.add_argument_group("Information")
    group5.add_argument(
        "--cite",
        dest="cite",
        default=False,
        action="store_true",
        help="Print citation in BibTeX format.",
    )

    results = parser.parse_args()

    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit()

    if results.cite is True:
        print(
            """
Thanks for using fits_warp! To cite this package, please use the following BibTeX:

                @ARTICLE{2018A&C....25...94H,
                   author = {{Hurley-Walker}, N. and {Hancock}, P.~J.},
                    title = \"{De-distorting ionospheric effects in the image plane}\",
                  journal = {Astronomy and Computing},
                archivePrefix = \"arXiv\",
                   eprint = {1808.08017},
                 primaryClass = \"astro-ph.IM\",
                 keywords = {Astrometry, Radio astronomy, Algorithms, Ionosphere},
                     year = 2018,
                    month = oct,
                   volume = 25,
                    pages = {94-102},
                      doi = {10.1016/j.ascom.2018.08.006},
                   adsurl = {http://adsabs.harvard.edu/abs/2018A%26C....25...94H},
                  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
                }

Other formats can be found here: http://adsabs.harvard.edu/abs/2018A%26C....25...94H
"""
        )
        sys.exit()

    if results.cores is None:
        cores = multiprocessing.cpu_count()
    else:
        cores = results.cores

    if results.incat is not None:
        if results.refcat is not None:
            corrected, xmcat = warped_xmatch(
                incat=results.incat,
                refcat=results.refcat,
                ra1=results.ra1,
                dec1=results.dec1,
                ra2=results.ra2,
                dec2=results.dec2,
            )
            xmcat.write(results.xm, overwrite=True)
            print("Wrote {0}".format(results.xm))
            if results.corrected is not None:
                corrected.write(results.corrected, overwrite=True)
                print("Wrote {0}".format(results.corrected))

    if results.infits is not None:
        if results.xm is not None:
            fnames = glob.glob(results.infits)
            # Use the first image to define the model
            make_pix_models(
                results.xm,
                results.ra1,
                results.dec1,
                results.ra2,
                results.dec2,
                fnames[0],
                results.plot,
                results.smooth,
                results.sigcol,
                results.noisecol,
                results.SNR,
                results.latex,
                max_sources=results.nsrcs,
            )
            if results.suffix is not None:
                correct_images(
                    fnames,
                    results.suffix,
                    results.testimage,
                    cores,
                    vm=results.vm,
                    progress=results.progress,
                )
            else:
                print("No output fits file specified via --suffix; not doing warping")
        else:
            print(
                "Must specify a cross-matched catalogue via --xm to perform the warping."
            )

