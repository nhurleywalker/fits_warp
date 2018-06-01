# fits_warp
Aim: Warp catalogues and images to remove the distorting affect of the ionosphere.

Authors: Natasha Hurley-Walker and Paul Hancock


## Usage
```
usage: fits_warp.py [-h] [--xm XM] [--infits INFITS] [--suffix SUFFIX]
                    [--ra1 RA1] [--dec1 DEC1] [--ra2 RA2] [--dec2 DEC2]
                    [--plot] [--smooth SMOOTH] [--incat INCAT]
                    [--refcat REFCAT] [--xmcat XM] [--corrected CORRECTED]

optional arguments:
  -h, --help            show this help message and exit

Warping input/output files:
  --xm XM               A .fits binary or VO table. The crossmatch between the
                        reference and source catalogue.
  --infits INFITS       The fits image(s) to be corrected; enclose in quotes
                        for globbing.
  --suffix SUFFIX       The suffix to append to rename the output (corrected)
                        fits image(s); e.g., specifying "warp" will result in
                        an image like image_warp.fits (no default; if not
                        supplied, no correction will be performed).

catalog column names:
  --ra1 RA1             The column name for ra (degrees) for source catalogue.
  --dec1 DEC1           The column name for dec (degrees) for source
                        catalogue.
  --ra2 RA2             The column name for ra (degrees) for reference
                        catalogue.
  --dec2 DEC2           The column name for dec (degrees) for reference
                        catalogue.

Other:
  --plot                Plot the offsets and models (default = False)
  --smooth SMOOTH       Smoothness parameter to give to the radial basis
                        function (default = 300 pix)
  --signal SIGCOL       Column from which to get the signal for a signal-to-
                        noise cut (e.g. peak_flux) (no default; if not
                        supplied, cut will not be performed
  --noise NOISECOL      Column from which to get the noise for a signal-to-
                        noise cut (e.g. local_rms) (no default; if not
                        supplied, cut will not be performed
  --SNR SNR             Signal-to-noise ratio for a signal-to-noise cut
                        (default = 10)

Crossmatching input/output files:
  --incat INCAT         Input catalogue to be warped.
  --refcat REFCAT       Input catalogue to be warped.
  --xmcat XM            Output cross match catalogue
  --corrected CORRECTED
                        Output corrected version of input catalogue
```

## Bugs/Questions
Please use the GitHub issue tracker to submit bug reports, feature requests, or questions.

## Credit
If you use fits_warp in your work please Cite Hurley-Walker and Hancock 2018 (in prep)
