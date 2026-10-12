# %% [markdown]
# # IMSNG_Rougine.ipynb
# - Automatically Process the image data from GECKO facilities and search for transients in the subtracted images with `gpPy`
# - Author: Gregory S.H. Paek (23.04.24)
# - Updated 2026.05.08. by Sophia Kim
# - Updated 2026.05.20. by YoungPyo Hong
#   - Upgrade Logging / implement multiprocessing / delete hardcoded paths / modify it for RASA36

# %% [markdown]
# ## Library

# %%
from __future__ import print_function, division, absolute_import
import os, sys, glob, subprocess
import numpy as np
import astropy.io.ascii as ascii
import matplotlib.pyplot as plt
plt.ioff()
from astropy.nddata import CCDData
from preprocess import calib
from util import tool
from astropy.io import fits
from astropy.table import Table, vstack
from astropy import units as u
from ccdproc import ImageFileCollection
from astropy.visualization import ZScaleInterval
import warnings
warnings.filterwarnings(action='ignore')
# from itertools import product
from itertools import repeat, chain
import multiprocessing
mp_fork = multiprocessing.get_context('fork')
import time
from tqdm import tqdm
import shutil
from datetime import datetime
from astropy.time import Time

# Warm up astropy's lazily-initialized machinery (wcslib, coordinate transform
# graph, ERFA) in the parent BEFORE any fork: first-touch init grabs module-level
# locks, and a fork landing while another thread holds one wedges the child
# forever on first use (2026-07-13: a stack worker froze in check_wcs_sanity and
# the run hung >10 h with no error). Shrinks the window; imap_collect is the net.
from astropy.wcs import WCS  # noqa: F401
from astropy.coordinates import SkyCoord
_wc = SkyCoord(10.0 * u.deg, -20.0 * u.deg)
_ = _wc.separation(SkyCoord(11.0 * u.deg, -21.0 * u.deg)).deg
_ = Time('2000-01-01T00:00:00', format='isot').mjd
del _wc, _

# Add for logging (25.01.31.)
import logging
from util.gppylogging import setup_global_logging, setup_worker_logging, get_log_queue, stop_listener

start_localtime = time.strftime('%Y-%m-%d %H:%M:%S (%Z)', time.localtime())

# %%
#	plot setting
import matplotlib.pyplot as plt
import matplotlib as mpl
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "last_expr"

mpl.rcParams["axes.titlesize"] = 14
mpl.rcParams["axes.labelsize"] = 20
plt.rcParams['savefig.dpi'] = 500
plt.rc('font', family='serif')

#%% Functions for ....
def _read_head(headfile, keyword):
	with open(headfile, 'r') as f:
		for line in f:
			if line[:8].strip() == keyword:
				return line.split('=')[1].split('/')[0].strip().strip("'").strip()
	return None

def save_fits_as_png(fitsfile, outpng=None):
	"""Save a FITS image as PNG with ZScale normalization for quick visual inspection.

	Previews are downsampled to <=4096 px on the long side and kept in float32:
	up to ncores of these run concurrently, and an oversized coadd (bad-WCS SWarp
	canvas) must not balloon into GBs of full-resolution float copies here.
	"""
	if outpng is None:
		outpng = os.path.splitext(fitsfile)[0] + '.png'
	try:
		from PIL import Image
		with fits.open(fitsfile) as hdul:
			data = hdul[0].data
			if data is None and len(hdul) > 1:
				data = hdul[1].data
			if data is None:
				return
			stride = max(1, (max(data.shape) + 4095) // 4096)
			data = np.array(data[::stride, ::stride], dtype=np.float32)
		finite = np.isfinite(data)
		if not np.any(finite):
			return
		interval = ZScaleInterval()
		vmin, vmax = interval.get_limits(data[finite])
		if vmax > vmin:
			data_norm = np.clip((data - vmin) / (vmax - vmin), 0, 1)
		else:
			data_norm = np.zeros_like(data)
		data_norm[~finite] = 0.0
		# PIL is thread-safe; flip vertically for astronomical orientation (origin='lower')
		img_uint8 = (data_norm[::-1] * 255).astype(np.uint8)
		Image.fromarray(img_uint8, mode='L').save(outpng)
	except Exception as e:
		logging.getLogger().warning(f"PNG generation failed for {os.path.basename(fitsfile)}: {e}")


def run_shell_checked(cmd, label):
	"""Run a shell command via os.system and raise RuntimeError if it didn't exit 0.

	os.system() return codes are easy to ignore, which is how a silently-failed
	SWarp/SCAMP/MissFITS call used to surface as a confusing FileNotFoundError two
	lines later instead of a clear error pointing at the actual failed command.
	"""
	status = os.system(cmd)
	if os.WIFSIGNALED(status):
		raise RuntimeError(f"{label} was killed by signal {os.WTERMSIG(status)} (likely OOM-killed): {cmd}")
	exitcode = os.WEXITSTATUS(status)
	if exitcode != 0:
		raise RuntimeError(f"{label} failed with exit code {exitcode}: {cmd}")


POOL_RESULT_TIMEOUT_SEC = 1800  # max wait for the NEXT pool result before declaring the stage stalled

# Records (stage_desc, n_abandoned) for every imap_collect stall in this run, so a
# wedged-worker abandonment is surfaced in the final Slack summary — not just buried
# in the log (2026-07-15: the WCS-update stall dropped 1 image silently for the user).
POOL_ABANDONED = []


def imap_collect(pool, func, argslist, desc, unit, timeout=POOL_RESULT_TIMEOUT_SEC):
	"""list(tqdm(pool.imap_unordered(...))) with a per-result timeout.

	A forked worker can inherit a lock that another thread held at the fork
	instant and wedge forever the first time it touches it (2026-07-13: the
	T00145 stack worker froze inside check_wcs_sanity and the run hung >10 h
	at 40/41 groups with no error). imap_unordered itself never times out, so
	guard result consumption: if no new result lands within `timeout`, log the
	stall and return what completed — the stage loses the outstanding task(s)
	instead of hanging the whole run, and exiting the Pool context terminates
	the wedged worker.
	"""
	results = []
	it = pool.imap_unordered(func, argslist)
	with tqdm(total=len(argslist), desc=desc, unit=unit) as pbar:
		for _ in range(len(argslist)):
			try:
				results.append(it.next(timeout=timeout))
			except multiprocessing.TimeoutError:
				n_left = len(argslist) - len(results)
				POOL_ABANDONED.append((desc, n_left))
				logger.error(
					f"{desc}: no result for {timeout}s with {n_left} task(s) outstanding - "
					f"likely a wedged worker (fork-inherited lock); abandoning the remaining task(s)"
				)
				break
			pbar.update(1)
	return results


def run_pre_sextractor(inim, outcat, conf_simple, param_simple, conv_simple, nnw_simple, pixscale, gain):
	# Pre-Source Extractor (FITS_LDAC for SCAMP)
	sexcom = (
		f"source-extractor -c {conf_simple} {inim} "
		f"-CATALOG_NAME {outcat} -CATALOG_TYPE FITS_LDAC "
		f"-PARAMETERS_NAME {param_simple} -FILTER_NAME {conv_simple} "
		f"-STARNNW_NAME {nnw_simple} -PIXEL_SCALE {pixscale} -GAIN {gain}"
	)
	logging.getLogger().info(sexcom)
	os.system(sexcom)


# Single-arg starmap adapters so these stages can run through imap_collect
# (per-result stall timeout) instead of bare pool.starmap, which (a) never
# times out on a wedged worker and (b) re-raises the FIRST worker exception
# in the parent, aborting the stage for every remaining frame (one frame
# with a bad header used to kill renaming/astrometry for the whole night).
def _astrometry_star(args):
	try:
		return calib.astrometry_RASA36(*args)
	except Exception as e:
		logger.error(f"astrometry worker failed for {os.path.basename(args[0])}: {e}")
		return None


def _fnamechange_star(args):
	try:
		return calib.fnamechange(*args)
	except Exception as e:
		logger.error(f"fnamechange worker failed for {os.path.basename(args[0])}: {e}")
		return None


def _pre_sextractor_star(args):
	try:
		return run_pre_sextractor(*args)
	except Exception as e:
		logger.error(f"pre-SExtractor worker failed for {os.path.basename(args[0])}: {e}")
		return None

#------------------------------------------------------------
def extract_date_and_time(date_obs_str, round_seconds=False):
	"""
	Extract date and time strings from FITS 'DATE-OBS' (ISOT).
	Returns:
		date_str: 'YYYYMMDD'
		time_str: 'HHMMSS'
	"""
	time_obj = Time(date_obs_str)
	date_str = time_obj.strftime('%Y%m%d')
	if round_seconds:
		time_str = time_obj.strftime('%H%M%S')
	else:
		time_str = f"{time_obj.datetime.hour:02}{time_obj.datetime.minute:02}{int(time_obj.datetime.second):02}"
	return date_str, time_str

keywords_to_add = [
	"IMAGETYP",
	"XBINNING", "YBINNING",
	"GAIN", "EGAIN",
	"XPIXSZ", "YPIXSZ",
	"INSTRUME",
	"SET-TEMP", "CCD-TEMP",
	"TELESCOP",
	"FOCALLEN", "FOCRATIO",
	"RA", "DEC",
	"PIERSIDE",
	"SITEELEV", "SITELAT", "SITELONG",
	"FWHEEL", "FILTER",
	"OBJECT",
	"OBJCTRA", "OBJCTDEC", "OBJCTROT",
	"FOCNAME", "FOCPOS", "FOCUSPOS", "FOCUSSZ",
	"ROWORDER",
	"_QUINOX",
	"SWCREATE",
	# Observing metadata preserved through calibration into stacked images
	"OBSERVER", "NOTES",
	"OBJTYPE", "IS_TOO", "is_rapidToO",
	"OBSID", "PROGID",
]


#------------------------------------------------------------------------
def get_radec_from_header(hdr):
	ra = (
		hdr.get('RA', None)
		or hdr.get('OBJCTRA', None)
		or hdr.get('OBJECTRA', None)
	)
	dec = (
		hdr.get('DEC', None)
		or hdr.get('OBJCTDEC', None)
		or hdr.get('OBJECTDEC', None)
	)

	if ra is not None:
		ra = str(ra).replace(' ', ':')
	if dec is not None:
		dec = str(dec).replace(' ', ':')

	return ra, dec


def check_wcs_sanity(im, pixscale_arcsec, fov_deg):
	"""Return None if the image's WCS looks sane, else a short reason string.

	Two bogus-WCS modes have produced multi-GB SWarp coadds (T00008 2026-06-24:
	3 GB from 5 frames instead of ~84 MB), whose float copies downstream (PNG
	previews, photometry, subtraction) blow past the per-user cgroup memory limit:
	  1. A blind solve-field retry "succeeds" degrees away from the real pointing
	     → caught by comparing the WCS center to the commanded RA/DEC header.
	  2. SCAMP fits garbage PV distortion terms on star-poor (polar) fields; the
	     center stays fine but the projected footprint spans several degrees
	     → caught by comparing the footprint diagonal to the physical
	       pixscale × image diagonal (bogus case measured 5.66 vs 3.79 deg).
	"""
	from astropy.wcs import WCS
	from astropy.coordinates import SkyCoord
	try:
		hdr = fits.getheader(im)
		w = WCS(hdr)
		if not w.has_celestial:
			return None
		wc = w.celestial
		nx, ny = hdr['NAXIS1'], hdr['NAXIS2']

		#	Footprint diagonal vs physical size (catches crazy distortion terms)
		corners = wc.pixel_to_world([0.5, nx + 0.5, nx + 0.5, 0.5], [0.5, 0.5, ny + 0.5, ny + 0.5])
		diag = max(corners[0].separation(corners[2]).deg, corners[1].separation(corners[3]).deg)
		expected_diag = pixscale_arcsec * np.hypot(nx, ny) / 3600.
		if not np.isfinite(diag):
			return "non-finite footprint (broken projection)"
		if diag > 1.25 * expected_diag:
			return f"footprint diagonal {diag:.2f} deg vs physical {expected_diag:.2f} deg"

		#	Center vs commanded pointing (catches translated blind solutions)
		ra, dec = get_radec_from_header(hdr)
		if ra is not None and dec is not None:
			try:
				cmd = SkyCoord(float(ra) * u.deg, float(dec) * u.deg)
			except ValueError:
				cmd = SkyCoord(ra, dec, unit=(u.hourangle, u.deg))
			cen = wc.pixel_to_world(nx / 2., ny / 2.)
			offset = float(cmd.separation(cen).deg)
			if offset > fov_deg:
				return f"center {offset:.2f} deg from commanded pointing"
		return None
	except Exception:
		return None


# %% [markdown]
# ## Ready =========================================================================

# %%
obs = 'RASA36'
print(f'# Observatory : {obs.upper()}')

no_stack = '--no-stack' in sys.argv

# --stack-sec <sec>: cap each stack group at <sec> total EXPTIME (e.g. 300 builds
# 300-s epochs from 100-s frames). Time-gap grouping (1 h) still applies first;
# each gap group is then chunked in time order. Ignored when --no-stack is given.
stack_sec = None
if '--stack-sec' in sys.argv:
	_i = sys.argv.index('--stack-sec')
	stack_sec = float(sys.argv[_i + 1])

try:
	path = sys.argv[1]
except Exception:
	# path = "/data6/obsdata/RASA36/2026-05-28_gain25"
	path = input("INPUT FOLDER PATH: ").strip()
print(f"- Input folder: {path}")

try:
	ncores = int(sys.argv[2])
except Exception:
	ncores = 4
print(f"- Number of Cores: {ncores}")

# %% [markdown]
# ### Path =========================================================================

# %%
HERE = os.path.dirname(os.path.abspath(__file__))

#------------------------------------------------------------
#   I/O  —  edit these paths for your environment
#------------------------------------------------------------
path_result  = '/data6/processed_RASA'               # output processed images
path_refimg  = '/data6/obsdata/RASA36/reference_img' # pre-staged external subtraction templates (Ref-RASA36-<tile>-<filter>-<survey>.fits)

#------------------------------------------------------------
#   Internal  —  pipeline code & reference data (relative to HERE)
#------------------------------------------------------------
path_factory       = os.path.normpath(os.path.join(HERE, '..', 'factory'))
path_ref           = os.path.join(path_factory, 'ref_frames', obs.upper())
path_mframe        = os.path.join(path_factory, 'master_frames')
path_calib         = os.path.join(path_factory, 'calib')

path_keys          = HERE
path_config        = os.path.join(HERE, 'config')
path_default_gphot = os.path.join(path_config, f'gphot.{obs.lower()}.config')
path_obs           = os.path.join(path_config, 'obs.dat')
path_changehdr     = os.path.join(HERE, 'table', 'changehdr.dat')
# path_alltarget     = os.path.join(HERE, 'table', 'alltarget.dat')
path_astrometry    = os.path.join(path_config, 'astrometry_gppy_rasa36.config')

conf_simple        = os.path.join(path_config, 'simple.sex')
param_simple       = os.path.join(path_config, 'simple.param')
nnw_simple         = os.path.join(path_config, 'simple.nnw')
conv_simple        = os.path.join(path_config, 'simple.conv')

path_phot_mp       = os.path.join(HERE, 'phot', 'gregoryphot_mp_2021.py')
path_phot_sub      = os.path.join(HERE, 'phot', 'gregoryphot_sub_2021.py')
path_find          = os.path.join(HERE, 'phot', 'gregoryfind_bulk_mp_2021.py')

# %% [markdown]
# - Folder setting =========================================================================

# %%
if not os.path.exists(path_factory):
	os.makedirs(path_factory)
if not os.path.exists(path_ref):
	os.makedirs(path_ref)
os.makedirs(path_result, exist_ok=True)


def atomic_cp(src, dst):
	"""Copy into a shared directory atomically (tmp + rename) so a concurrent
	pipeline run (e.g. ToO fast lane) never reads a half-written master frame."""
	tmp = f'{dst}.tmp{os.getpid()}'
	shutil.copy(src, tmp)
	os.replace(tmp, dst)


def scamp_output_opts(tag):
	"""SCAMP's default XML/check-plot names are relative, so they land in the
	shared cwd and collide across parallel workers and concurrent runs —
	redirect them into this run's factory dir with a unique tag."""
	pfx = f'{path_data}/scamp.{tag}'
	plots = ','.join(f'{pfx}.{p}' for p in (
		'fgroups', 'distort', 'astr_interror2d', 'astr_interror1d',
		'astr_referror2d', 'astr_referror1d', 'astr_chi2', 'psphot_error'))
	return f'-XML_NAME {pfx}.xml -CHECKPLOT_NAME {plots}'

# %% [markdown]
# ### Table =========================================================================

# %%
obstbl = ascii.read(path_obs)
hdrtbl = ascii.read(path_changehdr)
# alltbl = ascii.read(path_alltarget)
keytbl = ascii.read(f'{path_keys}/keys.dat')

#%% Added 25.01.31.
# Enter a new obs information if it does not in path_obs
if obs.upper() not in obstbl['obs']:
	print(f"'{obs.upper()}' information does not found in the obstbl."\
			"\n Please enter the following details to add it:")

	# Prompt for each required attribute
	ccd = input("Enter CCD name (e.g., 'FLI'): ")
	gain = float(input("Enter gain value (e.g., 1.5): "))
	rdnoise = float(input("Enter read noise value (e.g., 12.3): "))
	dark = float(input("Enter dark current value (e.g., 0.0): "))
	pixelscale = float(input("Enter pixel scale value (e.g., 0.5): "))
	fov = float(input("Enter field of view value (e.g., 18.0): "))
 
	new_row = [obs, ccd, gain, rdnoise, dark, pixelscale, fov]
	obstbl.add_row(new_row)
 
	today = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

	ascii.write(obstbl, path_obs, overwrite=True, delimiter='\t')
	with open(path_obs, 'a') as file:
		file.write(f'# Last update {today}\n')
	print(f"New entry for '{obs}' has been added to {path_obs}")

else: 
	print(f"'{obs}' information is already present.")

# %% [markdown]
# ## Process Summary Status =========================================================================

# %%
protbl = Table()
protbl['process'] = ['master_frame', 'pre_process', 'astrometry', 'cr_removal', 'wcs_refine', 'defringe', 'photometry', 'image_stack', 'photometry_com', 'subtraction', 'photometry_sub', 'transient_search', 'total']
protbl['status'] = False
protbl['time'] = 0.0 * u.second

# Processing statistics — defaults overwritten during run, used for final summary
biasnumb = 0
darknumb = 0
darkexptimelist = []
flatfilterlist = []
nobj = 0
objfilterlist = []
objexptimelist = []
n_calib_images = 0
n_astro_ok = 0
n_astro_fail = 0
n_combined = 0
n_subtracted = 0
sub_records = []  # per-image subtraction outcome (tile/filter/ref/status) for log + Slack reporting
objlist = []
filterlist = []

# %% [markdown]
# ## Main Body ==================================================================================

# %%
tdict = dict()
starttime = time.time()

# Detect obs type per object — prefer .head sidecars, fall back to FITS headers.
# obj_meta: {object_name: {'is_too': bool, 'objtype': str, 'too_name': str}}
date_str  = os.path.basename(path).split('_')[0]   # '2026-04-22' from '2026-04-22_gain25'

# ToO flag: True if any of these conditions hold:
#   1) IS_TOO or is_rapidToO header is True
#   2) OBJTYPE header is 'Request'
#   3) "too" appears in the input folder name or a raw filename
_path_has_too = 'too' in os.path.basename(path).lower()

def _check_too(is_too_str, is_rapidtoo_str, objtype_str, filename=''):
	return (
		is_too_str.strip().upper() == 'TRUE'
		or is_rapidtoo_str.strip().upper() == 'TRUE'
		or objtype_str.strip().lower() == 'request'
		or _path_has_too
		or 'too' in os.path.basename(filename).lower()
	)

headfiles = sorted(glob.glob(f'{path}/*.head'))
obj_meta  = {}

if headfiles:
	for hf in headfiles:
		name         = (_read_head(hf, 'OBJECT')      or 'unknown').strip()
		is_too_str   = (_read_head(hf, 'IS_TOO')      or 'False')
		rapidtoo_str = (_read_head(hf, 'is_rapidToO') or 'False')
		objtype      = (_read_head(hf, 'OBJTYPE')      or 'TOS').strip()
		notes_val    = (_read_head(hf, 'NOTES') or _read_head(hf, 'NOTE') or '').strip()
		is_too       = _check_too(is_too_str, rapidtoo_str, objtype, hf)
		too_name     = notes_val if notes_val else name
		obj_meta[name] = {'is_too': is_too, 'objtype': objtype, 'too_name': too_name}
else:
	fitsfiles = sorted(
		glob.glob(f'{path}/*.fts') +
		glob.glob(f'{path}/*.fits') +
		glob.glob(f'{path}/*.fit')
	)
	seen = set()
	for ff in fitsfiles:
		hdr  = fits.getheader(ff)
		name = str(hdr.get('OBJECT', 'unknown')).strip()
		if name in seen:
			continue
		seen.add(name)
		is_too_str   = str(hdr.get('IS_TOO',      'False'))
		rapidtoo_str = str(hdr.get('is_rapidToO', 'False'))
		objtype      = str(hdr.get('OBJTYPE',      'TOS')).strip()
		notes_val    = str(hdr.get('NOTES', hdr.get('NOTE', ''))).strip()
		is_too       = _check_too(is_too_str, rapidtoo_str, objtype, ff)
		too_name     = notes_val if notes_val else name
		obj_meta[name] = {'is_too': is_too, 'objtype': objtype, 'too_name': too_name}

if not obj_meta:
	obj_meta['unknown'] = {'is_too': False, 'objtype': 'TOS', 'too_name': 'unknown'}

# Each night gets its own subdirectory so concurrent runs don't collide
path_data = os.path.join(path_factory, obs.upper(), os.path.basename(path))

# Logging start
input_folder = os.path.basename(os.path.normpath(path))
logstart = datetime.now().strftime('%Y%m%d%H%M%S')

logdir = os.path.normpath(os.path.join(HERE, 'log'))
logger = setup_global_logging(logdir, obs.upper(), logstart, obs_subdir=input_folder, log_level=logging.INFO)
logger.setLevel(logging.INFO)
_log_queue = get_log_queue()
_logfile_path = os.path.join(logdir, obs.upper(), input_folder, f'{logstart}.log')

def _worker_log_init():
	setup_worker_logging(_log_queue)

#	Remove old folder and re-copy folder
rmcom = f'rm -rf {path_data}'
logger.info(rmcom)
os.system(rmcom)
cpcom = f'cp -r {path} {path_data}'
logger.info(cpcom)
os.system(cpcom)

logger.info(f"Pipeline start timestamp: {logstart}")

logger.info('Pipeline from Hearder Correction started')
#%% [markdown]
# ### Header Correction =========================================================================

# %%
try:
	imgfiles = sorted(glob.glob(path_data+'/*.fits'))
	imgfiles += sorted(glob.glob(path_data+'/*.fts'))
	ic0 = ImageFileCollection(path_data, keywords='*.fits', filenames=imgfiles) # TCSpy
	ic0.summary.write('{}/hdr.raw.dat'.format(path_data), format='ascii.tab', overwrite=True) 
	#	RASA Mode
	if obs == 'RASA36':
		if 'hdr' in path:
			mode = 'hdr'
			badmode = True
		elif 'high' in path:
			mode = 'high'
			badmode = True
		else:
			mode = 'high'
			badmode = True
			pass
		logger.info(f'Master Frame Mode:{mode} [Bad Mode:{badmode}]')

	calib.correcthdr_routine(path_data, hdrtbl, obs)
	logger.info("Correction Done")

	# With Slack noticifation (2025.01.23.)
	# - Slack notification

	slack_token = keytbl['key'][keytbl['name']=='slack'].item()
	channel = '#pipeline--imsng'

	obsinfo = calib.getobsinfo(obs, obstbl)
	gain = obsinfo['gain'].value       # e-/ADU, used for EGAIN and SExtractor
	rdnoise = obsinfo['rdnoise'].value # e-, informational
	fov = obsinfo['fov'] * u.arcmin    # angular FOV, used by transient search
	fov_deg = fov.to(u.deg).value      # WCS sanity threshold: solved center must be within 1 FOV of commanded pointing
	# objfilterlist, objexptimelist, flatfilterlist, darkexptimelist, obstime = calib.correcthdr_routine(path_data, hdrtbl)
	ic1 = ImageFileCollection(path_data, keywords='*')
	ic1.summary.write('{}/hdr.cor.dat'.format(path_data), format='ascii.tab', overwrite=True)

	# Saturated-core rejection (2026-07-13): a bright-moon sky can clip the
	# vignetting-boosted frame center at full well (2026-06-26: 62% of frames).
	# Clipped centers are unrecoverable, so drop these frames before calibration
	# and quarantine their original raw files to fail_img/<date>/.
	SATCORE_LEVEL = 60000  # ADU — central-box median at/near 16-bit full well
	SATCORE_BOX = 400      # half-size (px) of the central box
	path_failimg = f'/data6/obsdata/RASA36/fail_img/{date_str}'
	n_satcore = 0
	_satcands = [fc.files for fc in (ic1.filter(imagetyp='OBJECT'),
	                                 ic1.filter(imagetyp='object'),
	                                 ic1.filter(imagetyp='Light Frame')) if fc.files]
	# basename() is load-bearing: filter().files returns ABSOLUTE factory paths,
	# and os.path.join(path, <absolute>) discards path — so without it _rawfits
	# pointed at the factory copy and the "raw" quarantine moved that instead
	# (2026-05-28: gate's first live trigger crashed at the follow-up os.rename).
	for _fn in tqdm(sorted({os.path.basename(_f) for _f in chain.from_iterable(_satcands)}), desc="Saturated-core check", unit="img"):
		_fac = os.path.join(path_data, _fn)
		try:
			_dat = fits.getdata(_fac, memmap=False)
			_ny, _nx = _dat.shape
			_cen = float(np.median(_dat[_ny//2-SATCORE_BOX:_ny//2+SATCORE_BOX,
			                            _nx//2-SATCORE_BOX:_nx//2+SATCORE_BOX]))
			del _dat
		except Exception as _e:
			logger.warning(f'  SATCORE check failed ({_e}) - {_fn}')
			continue
		if _cen < SATCORE_LEVEL:
			continue
		n_satcore += 1
		os.makedirs(path_failimg, exist_ok=True)
		_rawfits = os.path.join(path, _fn)
		_rawhead = os.path.splitext(_rawfits)[0] + '.head'
		for _src in (_rawfits, _rawhead):
			if os.path.exists(_src):
				shutil.move(_src, os.path.join(path_failimg, os.path.basename(_src)))
		os.rename(_fac, _fac + '.satcore')  # drop factory copy from all later globs
		logger.warning(f'  SATURATED CORE ({_cen:.0f} ADU) - {_fn} -> raw moved to {path_failimg}')
	if n_satcore:
		logger.warning(f'  {n_satcore} frame(s) rejected for saturated core; raw files quarantined in {path_failimg}')
		ic1 = ImageFileCollection(path_data, keywords='*')
		ic1.summary.write('{}/hdr.cor.dat'.format(path_data), format='ascii.tab', overwrite=True)
except Exception as e:
	logger.error(f'Unexpected error occured: {e}')
	logger.exception('Full error traceback:')
	raise

#%% [markdown]
# Added 2025.01.23.
# ### Send Slack message to alarm the start of pipeline processing

#%%
slack_ts = None
try:
	# testobj = 'S190425z'
	project = "IMSNG"
	obsmode = "MONITORING" # Default
	'''
	for obj in ic1.filter(imagetyp='object').summary['object']:
		if 'MS' in obj[:2]: # MS230425 (test event)
			print(obj)
			project = "GECKO"
			obsmode = "TEST"
		elif 'S2' in obj[:2]: # S230425 (super event)
			print(obj)
			project = "GECKO"
			obsmode = "FOLLOWUP" # Follow-up
		else:
	'''

	#		pass
	#print(f"[{project}] {obsmode}")

	#filtered_ic1 = ic1.filter(imagetype='OBJECT') or ic1.filter(imagetype='object') or ic1.filter(imagetyp='Light Frame') or ic1.filter(imagetyp='object')

	filtered_hdr = [ic1.filter(imagetyp='OBJECT'),
					ic1.filter(imagetyp='object'),
					ic1.filter(imagetyp='Light Frame')]
	# dedupe: the imagetyp filters overlap, so the same file can match more than once
	filtered_ic1 = list(dict.fromkeys(chain.from_iterable(fc.files for fc in filtered_hdr if fc.files)))

	if filtered_ic1 is None or len(filtered_ic1) ==0: 
	#if filtered_ic1 is None or filtered_ic1.summary is None len(filtered_ic1.summary) ==0:

		text_only_calib = f"[`gpPy` / RASA36] Calibration only /n Path: /data6/obsdata/{obs.upper()}/{path_data.split('/')[-1]} have only calibration images. No object files found."
		if n_satcore:
			text_only_calib += f"\n:no_entry_sign: {n_satcore} science frame(s) rejected for saturated core — raw files moved to fail_img/{date_str}"
		logger.info(text_only_calib)

		param_slack = dict(
			token = slack_token,
			channel = channel,
			text = text_only_calib,
		)

		tool.slack_bot(**param_slack)

		proceed_obj_calib=False

	else: 
		#try:
			#nobj = len(ic1.filter(imagetyp='OBJECT').summary)		
		#except:
			#nobj = len(ic1.filter(imagetyp='object').summary)	
		nobj = len(filtered_ic1)

		try:
			obj_filters = sorted(list(set(ic1.filter(imagetyp='object').summary['filter'])))
			obj_exptimes = sorted(list(set(ic1.filter(imagetyp='object').summary['exptime'])))
		except Exception:
			obj_filters, obj_exptimes = [], []
		_fmt_filters = ", ".join(str(f) for f in obj_filters)
		_fmt_exptimes = ", ".join(str(int(e)) for e in obj_exptimes)
		try:
			_sci_files = [os.path.join(path_data, f) for f in ic1.filter(imagetyp='object').files]
			_total_size_mb = sum(os.path.getsize(f) for f in _sci_files if os.path.exists(f)) / 1024**2
			_per_file_mb = _total_size_mb / len(_sci_files) if _sci_files else 0
			if _total_size_mb >= 1000:
				_size_info = f"{_total_size_mb/1024:.2f} GB total ({_per_file_mb:.1f} MB/frame)"
			else:
				_size_info = f"{_total_size_mb:.1f} MB total ({_per_file_mb:.1f} MB/frame)"
		except Exception:
			_size_info = "N/A"
		_text_lines = [
			f":rocket: [`gpPy`/RASA36] *Pipeline Starting* — {os.path.basename(path)}",
			f"*Start:* {start_localtime}",
			f"*Raw science:* {nobj} frames | *Filters:* {_fmt_filters} | *ExpTime:* {_fmt_exptimes}s | *Cores:* {ncores}",
			f"*Data size:* {_size_info}",
		]
		if n_satcore:
			_text_lines.append(f":no_entry_sign: *Saturated-core frames:* {n_satcore} rejected — raw files moved to fail_img/{date_str}")
		text = "\n".join(_text_lines)
		logger.info(text)

		param_slack = dict(
			token = slack_token,
			channel = channel,
			text = text,
		)

		slack_ts = tool.slack_bot(**param_slack)

		proceed_obj_calib = True
except Exception as e:
	logger.error(f'Unexpected error occured: {e}')
	logger.exception('Full error traceback:')
	raise

# %% [markdown]
# ### Master Frame =========================================================================

# %% [markdown]
# - Bias

# %%
#	Track master frames reused from previous dates (populated in the borrow branches below)
borrowed_frames = []
try:
	st = time.time()
	#------------------------------------------------------------
	#	BIAS
	#------------------------------------------------------------
	zero_in = f"{path_data}/zero.fits"
	logger.info(f"START bias masterframe - input: {zero_in}")

	try:
		biasnumb = len(ic1.filter(imagetyp='Bias').summary)
	except Exception:
		biasnumb = 0
	# if len(ic1.filter(imagetyp='Bias').summary) != 0:
	if biasnumb != 0:
		mzero = calib.master_zero(ic1, fig=False)
		# print(zeroim)
		date = fits.getheader(f'{path_data}/zero.fits')['date-obs'][:10].replace('-', '')
	
		zeroim = f'{path_mframe}/{obs}/zero/{date}-zero_{mode}.fits'
		
		if not os.path.exists(os.path.dirname(zeroim)):
			os.makedirs(os.path.dirname(zeroim), exist_ok=True)
		logger.info(f'atomic_cp {path_data}/zero.fits -> {zeroim}')
		atomic_cp(f'{path_data}/zero.fits', zeroim)
		plt.close('all')

		logger.info(f"END   bias masterframe ? wrote: {zeroim}")

	else:
		#	IF THERE IS NO FLAT FRAMES, BORROW FROM CLOSEST OTHER DATE
		logger.warning('\nNO BIAS FRAMES\n')
		
		pastzero = np.array(glob.glob(f'{path_mframe}/{obs}/zero/*zero_{mode}.fits'))
		
		#	CALCULATE CLOSEST ONE FROM TIME DIFFERENCE
		deltime = []
		for date in pastzero:
			zeromjd = calib.isot_to_mjd((os.path.basename(date)).split('-')[0])
			deltime.append(np.abs(ic1.summary['mjd'][0]-zeromjd))
		indx_closet = np.where(deltime == np.min(deltime))
		# tmpzero = path_data+'/'+os.path.basename(np.asscalar(pastzero[indx_closet]))
		tmpzero = f"{path_data}/{os.path.basename(pastzero[indx_closet][0])}"
		# cpcom = 'cp {} {}'.format(np.asscalar(pastzero[indx_closet]), tmpzero)
		cpcom = f'cp {pastzero[indx_closet][0]} {tmpzero}'
		logger.info(cpcom)
		os.system(cpcom)

		if (mode == 'high') & (badmode == True):
			mzero = CCDData.read(tmpzero, hdu=0).multiply(20)
			logger.warning('[Bad mode] Multiply 20 on the high mode bias.')
		else:
			mzero = CCDData.read(tmpzero, hdu=0)

		mzero.meta['FILENAME'] = os.path.basename(tmpzero)
		borrowed_frames.append(f"Bias ({os.path.basename(tmpzero)})")
except Exception as e:
	logger.error(f'Unexpected error occured: {e}')
	logger.exception('Full error traceback:')
	raise

# %% [markdown]
# - Dark

# %%
try:
	#------------------------------------------------------------
	#	DARK (ITERATION FOR EACH EXPOSURE TIMES)
	#------------------------------------------------------------
	logger.info("START Dark masterframe")

	try:
		darkexptimelist = sorted(list(set(ic1.filter(imagetyp='dark').summary['exptime'])))
		darknumb = len(darkexptimelist)
	except Exception:
		darknumb = 0

	darkdict = dict()
	# if len(darkexptimelist) != 0:
	if darknumb != 0:
		dark_process = True

		def _make_master_dark(exptime):
			calib.master_dark(ic1, mzero=mzero, exptime=exptime, fig=False)
			date = fits.getheader(f'{path_data}/dark-{int(exptime)}.fits')['date-obs'][:10].replace('-', '')
			darkim = f'{path_mframe}/{obs}/dark/{int(exptime)}-{date}-dark_{mode}.fits'
			os.makedirs(os.path.dirname(darkim), exist_ok=True)
			atomic_cp(f'{path_data}/dark-{int(exptime)}.fits', darkim)
			plt.close('all')

		# Serial on purpose: one forked Process per exptime deadlocked on a
		# fork-inherited lock on multi-exptime dark-lab nights (2026-06-18,
		# 7 exptimes — wedged at p.join twice). Each combine takes seconds.
		for exptime in tqdm(darkexptimelist, desc="Dark frames", unit="exptime"):
			_make_master_dark(exptime)

		for exptime in darkexptimelist:
			darkdict[str(int(exptime))] = CCDData.read(f'{path_data}/dark-{int(exptime)}.fits')
		exptime = darkexptimelist[-1]
		logger.info('END making Dark master frames')

	else:
		#	Borrow
		logger.warning('\nNO DARK FRAMES\n')
		exptime = None
		try:
			objsum = ic1.filter(imagetyp='objec').summary
			objexptimelist = sorted(list(set(objsum['exptime'])))
			if len(objexptimelist) > 0:
				exptime = objexptimelist[-1]
		except Exception:
			pass

		if exptime is None:
			try:
				exptimelist = sorted(list(set(ic1.summary['exptime'])))
				if len(exptimelist) > 0:
					exptime = exptimelist[-1]
			except Exception:
				pass
			
		if exptime is None:
			try:
				hdr0 = fits.getheader(ic1.files[0])
				exptime = hdr0.get('EXPTIME', hdr0.get('exptime', None))
			except Exception:
				exptime = None
		
		if exptime is None:
			raise RuntimeError("Cannot determine exptime (no 'object' frames, no summary exptime, and header fallback failed).")


		#objexptimelist = sorted(list(set(ic1.filter(imagetyp='object').summary['exptime'])))
		#exptime = objexptimelist[-1]
		# pastdark = np.array(glob.glob('{}/{}/dark/{}*dark*.fits'.format(path_mframe, obs, int(exptime))))
		pastdark = np.array(glob.glob(f'{path_mframe}/{obs}/dark/{int(exptime)}*dark_{mode}.fits'))
		
		if len(pastdark) == 0:
			pastdark = np.array(glob.glob('{}/{}/dark/*dark*.fits'.format(path_mframe, obs)))
		else:
			pass
		#	CALCULATE CLOSEST ONE FROM TIME DIFFERENCE
		deltime = []
		delexptime = []
		darkexptimes = []
		for date in pastdark:
			# darkmjd = calib.isot_to_mjd((os.path.basename(date)).split('-')[0])
			darkmjd = calib.isot_to_mjd((os.path.basename(date)).split('-')[1])
			darkexptime = int( os.path.basename(date).split('-')[0] )
			# darkexptime = delexptime.append(int( os.path.basename(date).split('-')[1] ))
			darkexptimes.append(darkexptime)
			deltime.append(np.abs(ic1.summary['mjd'][0]-darkmjd))
		if 'KCT' in obs:
			indx_closet = np.where(
				(np.abs(np.array(darkexptimes)-exptime) == np.min(np.abs(np.array(darkexptimes)-exptime)))
			)
		else:
			indx_closet = np.where(
				(deltime == np.min(deltime)) &
				(darkexptimes == np.max(darkexptimes))
			)
		if len(indx_closet[0]) == 0:
			indx_closet = np.where(
				(deltime == np.min(deltime))
			)
		else:
			pass
		# tmpdark = path_data+'/'+os.path.basename(pastdark[indx_closet].item())
		# tmpdark = '{}/{}'.format(path_data, os.path.basename(pastdark[indx_closet[0]].item()))
		# tmpdark = pastdark[indx_closet[0]].item()
		tmpdark = pastdark[indx_closet][-1]
		exptime = int(fits.getheader(tmpdark)['exptime'])

		# cpcom = 'cp {} {}/dark-{}.fits'.format(tmpdark, path_data, int(exptime))
		cpcom = 'cp {} {}/dark-{}.fits'.format(tmpdark, path_data, int(exptime))
		logger.info(cpcom)
		os.system(cpcom)

		if (mode == 'high') & (badmode == True):
			mdark = CCDData.read(tmpdark, hdu=0).multiply(20)
			logger.warning('[Bad mode] Multiply 20 on the high mode dark.')
		else:
			mdark = CCDData.read(tmpdark, hdu=0)
		
		mdark.meta['FILENAME'] = os.path.basename(tmpdark)
		mdark.meta['EXPTIME'] = exptime
		darkdict['{}'.format(int(exptime))] = mdark
		borrowed_frames.append(f"Dark ({os.path.basename(tmpdark)})")
except Exception as e:
	logger.error(f'Unexpected error occured: {e}')
	logger.exception('Full error traceback:')
	raise

# %% [markdown]
# - Flat

# %%
try:
	logger.info("START making master flat frames")

	flatdict = dict()

	try:
		flatfilterlist = list(set(ic1.filter(imagetyp='flat').summary['filter']))

		def _make_master_flat(filte):
			flatsum = ic1.filter(imagetyp='flat', filter=filte).summary
			flat_exptimelist = sorted(list(set(flatsum['exptime'])))
			flat_exptime = flat_exptimelist[-1]
			mdark_use = darkdict.get(str(int(flat_exptime)), False)
			logger.info('MAKING MASTER FLAT IN {}-BAND'.format(filte))
			calib.master_flat(ic1, mzero, filte, mdark=mdark_use, fig=True)
			date = fits.getheader(f'{path_data}/dark-{int(exptime)}.fits')['date-obs'][:10].replace('-', '')
			flatim = f'{path_mframe}/{obs}/flat/{date}-n{filte}_{mode}.fits'
			os.makedirs(os.path.dirname(flatim), exist_ok=True)
			atomic_cp(f'{path_data}/n{filte}.fits', flatim)
			plt.close('all')

		# Serial on purpose — same fork-inherited-lock wedge risk as the dark stage.
		for filte in tqdm(flatfilterlist, desc="Flat filters", unit="filter"):
			_make_master_flat(filte)

		for filte in flatfilterlist:
			flatdict[filte] = CCDData.read(f'{path_data}/n{filte}.fits')
	except Exception:
		logger.warning('No flat calibration image.')
		# flatdict['None'] = None
		pass
	# tdict['masterframe'] = time.time() - st
	protbl['status'][protbl['process']=='master_frame'] = True
	protbl['time'][protbl['process']=='master_frame'] = int(time.time() - st)
except Exception as e:
	logger.error(f'Unexpected error occured: {e}')
	logger.exception('Full error traceback:')
	raise
logger.info("END flat masterframes")
logger.info(
	f"CALIB SUMMARY | Bias: {biasnumb} frames | "
	f"Dark: {darknumb} exp-times {[int(e) for e in darkexptimelist]} | "
	f"Flat: {len(flatfilterlist)} filters {flatfilterlist}"
)
try:
	_step_secs = int(protbl['time'][protbl['process']=='master_frame'].item())
	_fmt_flat = ", ".join(str(f) for f in flatfilterlist) or "none"
	_msg_lines = [
		f":white_check_mark: *[1/10] Master frames done* — {_step_secs}s",
		f"Bias: {biasnumb} frames | Dark: {darknumb} exp-times {[int(e) for e in darkexptimelist]} | Flat: {len(flatfilterlist)} filter(s) [{_fmt_flat}]",
	]
	if borrowed_frames:
		_msg_lines.append(f":recycle: Reused existing master frame(s) from a previous date: {', '.join(borrowed_frames)}")
	tool.slack_bot(
		token=slack_token, channel=channel, thread_ts=slack_ts,
		text="\n".join(_msg_lines)
	)
except Exception as _se:
	logger.warning(f"Slack notification failed: {_se}")

# %% [markdown]
#------------------------------------------------------------
#	OBJECT CALIBARTION (ZERO, DARK, FLAT)
#------------------------------------------------------------
#%%
try:
	if proceed_obj_calib == False:
		logger.info(f"Processing completed - {obs.upper()} {path_data.split('-')[-1]} contains only calibration images.")
		total_time = round(protbl['time'][protbl['process']=='total'].item()/60., 1)
		text = f'[`gpPy`/Processing Complete] {obs} {os.path.basename(path)} Calibration Data'
		try:
			tool.slack_bot(token=slack_token, channel=channel, thread_ts=slack_ts, text=text)
		except Exception as _se:
			logger.warning(f"Slack notification failed: {_se}")
		# Calibration-only night: master frames are done, nothing else to reduce.
		# Without this exit the script falls through to astrometry and dies on
		# the undefined fzimlist (NameError).
		sys.exit(0)

	else:
		logger.info("START object calibration")

		st_ = time.time()
		objfilterlist = sorted(list(set(ic1.filter(imagetyp='object').summary['filter'])))
		objexptimelist = sorted(list(set(ic1.filter(imagetyp='object').summary['exptime'])))

		# Phase 1 (serial): ensure flatdict is fully populated before forking
		for filte in objfilterlist:
			if filte in flatdict:
				continue
			logger.warning('\nNO {} FLAT FRAMES\n'.format(filte))
			deltime = []
			pastflat = np.array(glob.glob(f'{path_mframe}/{obs}/flat/*-n{filte}_{mode}.fits'))
			for date in pastflat:
				flatmjd = fits.getheader(date)['MJD']
				deltime.append(np.abs(ic1.summary['mjd'][0]-flatmjd))
			indx_closet = np.where(deltime == np.min(deltime))
			tmpflat = '{}/{}'.format(path_data, os.path.basename(pastflat[indx_closet][0].item()))
			cpcom = 'cp {} {}'.format(pastflat[indx_closet][0].item(), tmpflat)
			logger.info(cpcom)
			os.system(cpcom)
			if ('KCT' not in obs) & ('LSGT_SNUCAMII' not in obs):
				mflat = CCDData.read(tmpflat, hdu=0)
			elif 'LSGT_SNUCAMII' in obs:
				mflat = CCDData.read(tmpflat, hdu=0, unit='adu')
			else:
				#KCT Exception
				mflat = CCDData.read(tmpflat, hdu=0, unit='adu')
				mflat.meta['FILENAME'] = os.path.basename(tmpflat)
			mflat.meta['FILENAME'] = os.path.basename(tmpflat)
			flatdict[filte] = mflat
		logger.info(f"Flatdict: {flatdict}")
	
		# Phase 2 (parallel): each filter is independent — fork one process per filter.
		# mp_fork.Process uses fork so closures and CCDData objects need no pickling.
		def _calibrate_filter(filte):
			setup_worker_logging(_log_queue)
			for expt in objexptimelist:
				if str(int(expt)) in darkdict:
					mdark = darkdict[str(int(expt))]
				else:
					mdark = darkdict[list(darkdict.keys())[-1]]
				# exptime= fixes the bug where all images of a filter were re-processed
				# once per exptime value, with the last dark overwriting earlier results.
				calib.calibration(ic1, mzero, flatdict[filte], filte, mdark=mdark, exptime=expt)

		if len(objfilterlist) == 1:
			# Single filter: run in-process to avoid fork doubling memory usage on low-RAM machines
			_calibrate_filter(objfilterlist[0])
			logger.info(f"DONE object calib - filter: {objfilterlist[0]}")
		else:
			processes = [mp_fork.Process(target=_calibrate_filter, args=(filte,)) for filte in objfilterlist]
			for p in processes:
				p.start()
			# Bounded join: this is the last untimed fork in the pipeline, and forking a
			# Pool/Process from the multi-threaded parent can inherit a lock and wedge a
			# worker forever (the fork-inherited-lock wedge that hit dark/flat + stack).
			# An untimed p.join() would hang the whole run with no recovery, so cap it and,
			# on a wedge, terminate the worker and count that filter as failed — the frames
			# it didn't calibrate simply drop out of fzimlist / n_calib_images, same
			# skip-and-count handling as the imap_collect stages. A genuine nonzero exit
			# (a real calibration crash, not a wedge) still raises.
			for filte, p in tqdm(zip(objfilterlist, processes), total=len(objfilterlist), desc="Filters", unit="filter"):
				p.join(timeout=POOL_RESULT_TIMEOUT_SEC)
				if p.is_alive():
					logger.error(
						f"Calibration worker for filter '{filte}' did not finish in {POOL_RESULT_TIMEOUT_SEC}s "
						f"- likely a wedged worker (fork-inherited lock); terminating and skipping this filter"
					)
					POOL_ABANDONED.append(("Object calibration", 1))
					p.terminate()
					p.join()
					continue
				if p.exitcode != 0:
					raise RuntimeError(f"Calibration worker for filter '{filte}' failed (exit code {p.exitcode})")
				logger.info(f"DONE object calib - filter: {filte}")

		protbl['status'][protbl['process']=='pre_process'] = True
		protbl['time'][protbl['process']=='pre_process'] = int(time.time() - st_)
		logger.info(f"END object calibration - total {round(time.time()-st_,1)}s")

		#	Corrected image list
		fzimlist = []
		for ims in ('{}/fz*.fits'.format(path_data), '{}/fz*.fit'.format(path_data), '{}/fz*.fts'.format(path_data)):
			fzimlist.extend(sorted(glob.glob(ims)))

		n_calib_images = len(fzimlist)
		logger.info(
			f"PRE_PROCESS SUMMARY | Calibrated: {n_calib_images} images | "
			f"Filters: {objfilterlist} | ExpTimes: {[int(e) for e in objexptimelist]}s"
		)
		try:
			_step_secs = int(protbl['time'][protbl['process']=='pre_process'].item())
			_fmt_objfilters = ", ".join(str(f) for f in objfilterlist)
			tool.slack_bot(
				token=slack_token, channel=channel, thread_ts=slack_ts,
				text="\n".join([
					f":white_check_mark: *[2/10] Pre-process done* — {_step_secs}s",
					f"Calibrated: {n_calib_images} images | Filters: {_fmt_objfilters} | ExpTimes: {[int(e) for e in objexptimelist]}s",
				])
			)
		except Exception as _se:
			logger.warning(f"Slack notification failed: {_se}")
except Exception as e:
	logger.error(f'Unexpected error occured: {e}')
	logger.exception('Full error traceback:')
	raise
# %% [markdown]
# # Astrometry ----------------------------------------------------------------------------
 
#%%
try:
	st_ = time.time()
	logger.info('ASTROMETRY START')
	logger.info('='*60)
	astrometryfailist = []
	# fzimlist = sorted(glob.glob(path_data+'/fz*.fits'))
	astimlist = []
	#astotlist = []
	astralist = []
	astdelist = []
	astblindlist = []


	for inim in tqdm(fzimlist, desc="Classify astrometry", unit="img"):
		outim = f'{path_data}/a{os.path.basename(inim)}'

		# ÀÌ¹Ì astrometry °á°ú°¡ ÀÖÀ¸¸é skip
		if os.path.exists(outim) and os.path.getsize(outim) > 0:
			logger.warning(f"  SKIP astrometry already exists - {os.path.basename(outim)}")
			continue

		hdr = fits.getheader(inim)
		obj = hdr.get('OBJECT', '').upper()

		ra, dec = None, None

		# if obj in alltbl['obj']:
		# 	indx_target = np.where(obj == alltbl['obj'])[0][0]
		# 	ra = alltbl['ra'][indx_target].item()
		# 	dec = alltbl['dec'][indx_target].item()
		# else:
		ra, dec = get_radec_from_header(hdr)

		if (ra is not None) and (dec is not None):
			astimlist.append(inim)
			astralist.append(ra)
			astdelist.append(dec)
		else:
			astblindlist.append(inim)


	# Astrometry with RA/Dec prior
	if __name__ == '__main__':
		logger.info(f"  QUEUE constrained astrometry - {len(astimlist)} images")
		with mp_fork.Pool(processes=ncores, initializer=_worker_log_init) as pool:
			results = imap_collect(
				pool, _astrometry_star,
				list(zip(
					astimlist,
					repeat(obsinfo['pixscale']),
					astralist,
					astdelist,
					repeat(obsinfo['fov'] / 60.),
					repeat(60),
					repeat(path_astrometry)
				)),
				desc="Constrained astrometry", unit="img"
			)
		logger.info("  DONE constrained astrometry")


	# Astrometry without RA/Dec prior
	if __name__ == '__main__':
		logger.info(f"  QUEUE blind astrometry - {len(astblindlist)} images")
		with mp_fork.Pool(processes=ncores, initializer=_worker_log_init) as pool:
			results = imap_collect(
				pool, _astrometry_star,
				list(zip(
					astblindlist,
					repeat(obsinfo['pixscale']),
					repeat(None),
					repeat(None),
					repeat(None),
					repeat(120),
					repeat(path_astrometry)
				)),
				desc="Blind astrometry", unit="img"
			)
		logger.info("  DONE blind astrometry")


	# Retry failed images, regardless of IMSNG/non-IMSNG
	astfailist = []

	for inim in tqdm(fzimlist, desc="Check astrometry failures", unit="img"):
		outim = f'{path_data}/a{os.path.basename(inim)}'
		if not os.path.exists(outim):
			astfailist.append(inim)

	if __name__ == '__main__':
		logger.info(f"  QUEUE failed astrometry retry - {len(astfailist)} images")
		with mp_fork.Pool(processes=ncores, initializer=_worker_log_init) as pool:
			results = imap_collect(
				pool, _astrometry_star,
				list(zip(
					astfailist,
					repeat(obsinfo['pixscale']),
					repeat(None),
					repeat(None),
					repeat(None),
					repeat(180),
					repeat(path_astrometry)
				)),
				desc="Astrometry retry", unit="img"
			)
		logger.info("  DONE failed astrometry retry")


	# Reject bogus solutions (translated blind solves, crazy distortion fits)
	n_badwcs = 0
	for inim in tqdm(fzimlist, desc="Validate astrometry", unit="img"):
		outim = f'{path_data}/a{os.path.basename(inim)}'
		if not os.path.exists(outim):
			continue
		reason = check_wcs_sanity(outim, obsinfo['pixscale'], fov_deg)
		if reason is not None:
			os.rename(outim, outim + '.badwcs')
			n_badwcs += 1
			logger.warning(f"  BAD WCS ({reason}) - {os.path.basename(outim)}")
	if n_badwcs:
		logger.warning(f"  {n_badwcs} image(s) rejected for bogus WCS solutions")


	# Final failed list
	astrometryfailist = []

	for inim in tqdm(fzimlist, desc="Astrometry final check", unit="img"):
		outim = f'{path_data}/a{os.path.basename(inim)}'
		if not os.path.exists(outim):
			astrometryfailist.append(outim)
			logger.warning(f"  ASTROMETRY FINAL FAIL - {os.path.basename(inim)}")

	os.system('rm '+path_data+'/*.axy '+path_data+'/*.corr '+path_data+'/*.xyls '+path_data+'/*.match '+path_data+'/*.rdls '+path_data+'/*.solved '+path_data+'/*.wcs ')
	n_astro_ok = len(fzimlist) - len(astrometryfailist)
	n_astro_fail = len(astrometryfailist)
	logger.info(
		f"ASTROMETRY SUMMARY | OK: {n_astro_ok}/{len(fzimlist)} | Failed: {n_astro_fail}"
		+ (f" | Failed: {[os.path.basename(f) for f in astrometryfailist]}" if astrometryfailist else "")
	)
	logger.info('ASTROMETRY COMPLETE\n'+'='*60)
	# tdict['astronomy'] = time.time() - st - tdict[list(tdict.keys())[-1]]
	protbl['status'][protbl['process']=='astrometry'] = True
	protbl['time'][protbl['process']=='astrometry'] = int(time.time() - st_)
	try:
		_step_secs = int(protbl['time'][protbl['process']=='astrometry'].item())
		_astro_pct = f"{100*n_astro_ok//n_calib_images}%" if n_calib_images > 0 else "N/A"
		tool.slack_bot(
			token=slack_token, channel=channel, thread_ts=slack_ts,
			text="\n".join([
				f":white_check_mark: *[3/10] Astrometry done* — {_step_secs}s",
				f"OK: {n_astro_ok}/{n_calib_images} ({_astro_pct}) | Failed: {n_astro_fail}",
			])
		)
	except Exception as _se:
		logger.warning(f"Slack notification failed: {_se}")

	# --- Astrometry total-failure gate (Fix 2a, 2026-07-24) ---
	# A night that solves ZERO of many real science frames is a silent total failure
	# (e.g. 2026-06-22: bad master flat -> 0 stars detected -> 0/427 solved, yet the run
	# still went on to post a clean "COMPLETE"). Treat it as a hard failure: alert loudly
	# and stop, instead of grinding through empty stack/subtraction to a misleading summary.
	MIN_FRAMES_FOR_ASTRO_GATE = 20
	if n_astro_ok == 0 and n_calib_images >= MIN_FRAMES_FOR_ASTRO_GATE:
		logger.critical(
			f"ASTROMETRY TOTAL FAILURE - 0/{n_calib_images} science frames solved. "
			f"Likely bad calibration (e.g. master flat) or non-astrometric data. "
			f"Stopping run: nothing downstream can produce output."
		)
		try:
			tool.slack_bot(
				token=slack_token, channel=channel, thread_ts=slack_ts,
				text="\n".join([
					f":rotating_light: *Astrometry TOTAL failure - {os.path.basename(path)}*",
					f"0/{n_calib_images} science frames solved (SExtractor found no usable stars).",
					f"Run stopped before stacking/subtraction - check the master flat / sky conditions, then re-run.",
				])
			)
		except Exception as _se:
			logger.warning(f"Slack notification failed: {_se}")
		sys.exit(1)
except Exception as e:
	logger.error(f'Unexpected error occured: {e}')
	logger.exception('Full error traceback:')
	raise
logger.info('Astrometry complete')

# ### Cosmic-ray Removal - SKIP for RASA
try:
	st_ = time.time()
	logger.info('Skip seeing measurement & CR removal for RASA36')
	logger.info('='*60)

	afzimlist = []
	for ims in (
		f'{path_data}/a*.fits',
		f'{path_data}/a*.fit',
		f'{path_data}/a*.fts',
	):
		afzimlist.extend(sorted(glob.glob(ims)))

	outimlist = []
	for inim in afzimlist:
		outim = f"{os.path.dirname(inim)}/cr{os.path.basename(inim)}"
		outimlist.append(outim)

	logger.info(f"RASA36 copy a* -> cra*: {len(afzimlist)} images")

	import shutil
	from concurrent.futures import ThreadPoolExecutor

	def _copy_if_needed(pair):
		inim, outim = pair
		if os.path.exists(outim) and os.path.getsize(outim) > 0:
			return
		shutil.copy2(inim, outim)

	with ThreadPoolExecutor(max_workers=ncores) as executor:
		list(tqdm(
			executor.map(_copy_if_needed, zip(afzimlist, outimlist)),
			total=len(afzimlist), desc="CR copy a*->cra*", unit="img"
		))

	protbl['status'][protbl['process']=='cr_removal'] = True
	protbl['time'][protbl['process']=='cr_removal'] = int(time.time() - st_)
except Exception as e:
	logger.error(f'Unexpected error occured: {e}')
	logger.exception('Full error traceback:')
	raise
logger.info(f"END RASA36 a* -> cra* copy - total {round(time.time()-st_, 1)}s")

### Rename to IMSNG/GECKO Convention
try:
	crafzimlist = []
	for ims in ('{}/cra*.fits'.format(path_data), '{}/cra*.fit'.format(path_data), '{}/cra*.fts'.format(path_data)):
		crafzimlist.extend(sorted(glob.glob(ims)))
	# Serial in-process: ~7 ms/image (header read + mv), so a fork pool buys
	# nothing while exposing this stage to the fork-inherited-lock wedge — same
	# reasoning as the RACENT/DECCENT update and the dark/flat combines. Never
	# pool millisecond-scale work behind imap_collect's 1800 s timeout.
	for inim in tqdm(crafzimlist, desc="Rename to IMSNG", unit="img"):
		_fnamechange_star((inim, obs))

	caliblist = sorted(glob.glob(path_data+'/Calib*.fits'))
	ic_cal = ImageFileCollection(path_data, glob_include='Calib*0.fits', keywords='*')
	os.system('chmod 777 {}'.format(path_data))
	os.system('chmod 777 {}/*'.format(path_data))
	#	Calib-*.fits TO SAVE PATH
	f = open(path_data+'/object.txt', 'a')
	f.write('obs obj dateobs filter exptime\n')
	for inim in caliblist:
		img = os.path.basename(inim)
		part = img.split('-')
		line = '{} {} {} {} {}\n'.format(part[1], part[2], part[3]+'T'+part[4], part[5], part[6])
		logger.debug(line.rstrip())
		f.write(line)
	f.close()

	os.system(f'rm {path_data}/*fz*.f*')
	plt.close('all')
except Exception as e:
	logger.error(f'Unexpected error occured: {e}')
	logger.exception('Full error traceback:')
	raise

# %% [markdown]
# ## SCAMP + MissFITS only (update WCS on single images first)

#%%
try:
	logger.info("START SCAMP + MissFITS WCS update)")
	st_ = time.time()

	combined_images = []
	step = 1e-10 if no_stack else (1/24/60)*60  # 1e-10: each frame its own group (no-stack); default: 1 hour
	ic_cal_phot = ImageFileCollection(path_data, glob_include='Calib*0.fits', keywords='*')
	passa_skipped = []  # (group_label, size) of Pass A SCAMP/MissFITS groups skipped on failure (one bad tile must not abort the night)
	calist = sorted(glob.glob('{}/Calib*.fits'.format(path_data)))
	if ic_cal_phot.summary is None:
		logger.warning(f"No Calib*0.fits files found in {path_data}, skipping SCAMP pass A")
		objlist, filterlist = [], []
	else:
		objlist = sorted(list(set(ic_cal_phot.summary['object'])))
		filterlist = sorted(list(set(ic_cal_phot.summary['filter'])))


	for obj in tqdm(objlist, desc="SCAMP Pass A (obj)", unit="obj"):
		for filte in tqdm(filterlist, desc=f"  SCAMP [{obj}]", unit="filter", leave=False):
			imlist_tmp = sorted(glob.glob('{}/Calib*-{}-*-{}-*.fits'.format(path_data, obj, filte)))
			logger.info(f"IMAGE REGISTERING & COMBINE process on {imlist_tmp}")

			if len(imlist_tmp) == 0:
				pass
			elif len(imlist_tmp) == 1:
				# Nothing to SCAMP/MissFITS for a single frame
				pass
			else:
				logger.info(f"{obj} {filte}")

				jds = np.array([fits.getheader(inim)['jd'] for inim in imlist_tmp])
				delts = jds - np.min(jds)

				grouplist = []
				grouplists = []

				for i in range(len(delts)):
					if i == 0:
						t0 = delts[i]
					#elif i == len(delts)-1:
					#	grouplists.append(grouplist)

					t1 = delts[i]
					dif = np.abs(t0 - t1)

					if dif < step:
						grouplist.append(imlist_tmp[i])
					else:
						grouplists.append(grouplist)
						grouplist = [imlist_tmp[i]]
					t0 = t1
					
				grouplists.append(grouplist)
				
				# PASS A loop: SCAMP/MissFITS only
				for gidx, group in tqdm(enumerate(grouplists, 1), desc=f"    SCAMP groups [{obj}/{filte}]", unit="grp", total=len(grouplists), leave=False):
					logger.info(f"  [{gidx}/{len(grouplists)}] PASS A group size={len(group)}")

					if len(group) > 1:
						files_to_stack = group
						baseim = files_to_stack[0]

						# 1) Pre-SExtractor catalogs
						outcatlist = []
						outheadlist = []

						for inim in files_to_stack:
							outcat = inim.replace('.fits', '.cat')
							outhead = inim.replace('.fits', '.head')
							outcatlist.append(outcat)
							outheadlist.append(outhead)

						if __name__ == '__main__':
							with mp_fork.Pool(processes=ncores, initializer=_worker_log_init) as pool:
								imap_collect(pool, _pre_sextractor_star, [
									(inim, outcat, conf_simple, param_simple, conv_simple, nnw_simple, obsinfo['pixscale'], gain)
									for inim, outcat in zip(files_to_stack, outcatlist)
								], desc="Pre-SExtractor", unit="img")

						# 2) SCAMP catalog list
						path_cat_scamp_list = f"{path_data}/cat.scamp.{obj}.{filte}.{gidx}.list"
						logger.info(f"Generate Catalog list for SCAMP: {path_cat_scamp_list}")
						with open(path_cat_scamp_list, 'w') as s:
							for incat in outcatlist:
								s.write(f"{incat}\n")

						# 3) MissFITS list (keep your variable name as-is)
						path_head_missfits_list = f"{path_data}/head.missfits.{obj}.{filte}.{gidx}.list"
						logger.info(f"Generate Header missfits list for SCAMP: {path_head_missfits_list}")
						with open(path_head_missfits_list, 'w') as i:
							for inim in files_to_stack:
								i.write(f"{inim}\n")

						# SCAMP + MissFITS — guarded per group so one bad tile can't abort the whole night.
						# SCAMP occasionally segfaults (exit 139, transient — e.g. a glitched Gaia refcat
						# download) or errors (exit 1) on a single group; skip that group and continue.
						try:
							# SCAMP
							scampcom = f"scamp -c {path_config}/rasa36.scamp @{path_cat_scamp_list} {scamp_output_opts(f'A.{obj}.{filte}.{gidx}')}"
							logger.info(scampcom)
							run_shell_checked(scampcom, "SCAMP (pass A)")

							# MissFITS
							missfitscom = f"missfits @{path_head_missfits_list}"
							logger.info(missfitscom)
							run_shell_checked(missfitscom, "MissFITS (pass A)")
						except Exception as _passa_e:
							_grp_label = f"{obj}/{filte}/g{gidx}"
							logger.error(
								f"SCAMP/MissFITS Pass A FAILED on group {_grp_label} "
								f"(size={len(group)}) — skipping this tile group, run continues: {_passa_e}"
							)
							passa_skipped.append((_grp_label, len(group)))
							continue

					else:
						# Nothing to do
						pass

	logger.info(f"END PASS A: SCAMP/MissFITS WCS update - total {round(time.time()-st_,1)}s")
except Exception as e:
	logger.error(f'Unexpected error occured: {e}')
	logger.exception('Full error traceback:')
	raise

		# ------------------------------------------------------------
		# Update WCS center / footprint keywords after SCAMP + MissFITS
		# ------------------------------------------------------------
try:
	caliblist = sorted(glob.glob(f'{path_data}/Calib*.fits'))

	logger.info(f"START updating RACENT/DECCENT/RAPOLY after SCAMP - {len(caliblist)} images")

	def _update_wcs_keywords(calibim):
		try:
			center, vertices = tool.get_wcs_coordinates(calibim)
			fits.setval(calibim, "RACENT", value=round(center[0].item(), 3), comment="RA CENTER [deg]")
			fits.setval(calibim, "DECCENT", value=round(center[1].item(), 3), comment="DEC CENTER [deg]")
			for ii, (_ra, _dec) in enumerate(vertices):
				fits.setval(calibim, f"RAPOLY{ii}", value=round(_ra, 3), comment=f"RA POLYGON {ii} [deg]")
				fits.setval(calibim, f"DEPOLY{ii}", value=round(_dec, 3), comment=f"DEC POLYGON {ii} [deg]")
			return None
		except Exception as e:
			return (calibim, str(e))

	# Serial in-process: ~40 ms/image, so a pool buys nothing while exposing the
	# stage to the fork-inherited-lock wedge (2026-07-15: one worker wedged and
	# imap_collect waited the full 1800 s for a millisecond task).
	for calibim in caliblist:
		r = _update_wcs_keywords(calibim)
		if r is not None:
			logger.warning(f"FAIL WCS footprint update - {os.path.basename(r[0])}: {r[1]}")

	logger.info("END updating RACENT/DECCENT/RAPOLY after SCAMP")
except Exception as e:
	logger.error(f'Unexpected error occured: {e}')
	logger.exception('Full error traceback:')
	raise

# The SCAMP + MissFITS + WCS-keyword-update block above (st_ set at its start,
# ~line 1394) is its own pipeline stage — it used to record no protbl time and
# post no Slack step, so its wall-clock (~45% of a nightly run; 625 s even on this
# 10-frame ToO) vanished into the gap between the [3/N] Astrometry and photometry
# Slack messages, making the photometry step look like it took the whole ~18 min.
# Attribute it explicitly so every Slack step's "— Ns" matches the elapsed time
# since the previous step and the stage times sum to the total.
protbl['status'][protbl['process']=='wcs_refine'] = True
protbl['time'][protbl['process']=='wcs_refine'] = int(time.time() - st_)
try:
	_step_secs = int(protbl['time'][protbl['process']=='wcs_refine'].item())
	_wcs_lines = [
		f":white_check_mark: *[4/10] WCS refine (SCAMP+MissFITS) done* — {_step_secs}s",
		f"{n_astro_ok} frames registered against Gaia",
	]
	if passa_skipped:
		_n_skip_frames = sum(_sz for _, _sz in passa_skipped)
		_wcs_lines.append(
			f":warning: Skipped {len(passa_skipped)} tile group(s) "
			f"({_n_skip_frames} frame(s)) on SCAMP/MissFITS failure — run continued"
		)
		for _lbl, _sz in passa_skipped[:25]:
			_wcs_lines.append(f"  • {_lbl} ({_sz} frames)")
		if len(passa_skipped) > 25:
			_wcs_lines.append(f"  … +{len(passa_skipped) - 25} more (see log)")
	tool.slack_bot(
		token=slack_token, channel=channel, thread_ts=slack_ts,
		text="\n".join(_wcs_lines)
	)
except Exception as _se:
	logger.warning(f"Slack notification failed: {_se}")

# %% [markdown]
# ## Photometry
# %%
try:
	logger.info("START PHOTOMETRY")
	st_ = time.time()
	path_infile = f'{path_data}/{os.path.basename(path_default_gphot)}'
	path_new_gphot = f'{os.path.dirname(path_infile)}/gphot.config'

	#	Copy default photometry configuration
	cpcom = f'cp {path_default_gphot} {path_new_gphot}'
	logger.info(cpcom)
	os.system(cpcom)

	#	Read default photometry configuration
	f = open(path_default_gphot, 'r')
	lines = f.read().splitlines()
	f.close()

	#	Write photometry configuration
	g = open(path_new_gphot, 'w')
	for line in lines:
		if 'imkey' in line:
			line = f'imkey\t{path_data}/C*0.fits'
		else:
			pass
		g.write(line+'\n')
	g.close()


	path_phot = path_phot_mp

	#	Execute
	com = f'python {path_phot} {path_data} {ncores}'
	# com = f'python {path_phot} {path_data} 1'
	logger.info(com)
	os.system(com)
	#	gregoryphot's exit code is not propagated and its stdout is not in this
	#	log, so an all-frames failure is otherwise invisible (2026-07-14: NGC292
	#	ToO got zero phot.cat because PS1 has no coverage at Dec -72).
	_n_pin = len(glob.glob(f'{path_data}/C*0.fits'))
	_n_pout = len(glob.glob(f'{path_data}/C*0.phot.cat'))
	if _n_pin > 0 and _n_pout == 0:
		logger.error(f"Photometry produced no phot.cat for any of {_n_pin} frames - check refcat coverage / gregoryphot output")
	logger.info(f"END   photometry - total {round(time.time()-st_,1)}s")

	protbl['status'][protbl['process']=='photometry'] = True
	protbl['time'][protbl['process']=='photometry'] = int(time.time() - st_)
	try:
		_step_secs = int(protbl['time'][protbl['process']=='photometry'].item())
		tool.slack_bot(
			token=slack_token, channel=channel, thread_ts=slack_ts,
			text="\n".join([
				f":white_check_mark: *[5/10] Photometry done* — {_step_secs}s",
				f"{n_astro_ok} individual frames processed",
			])
		)
	except Exception as _se:
		logger.warning(f"Slack notification failed: {_se}")
except Exception as e:
	logger.error(f'Unexpected error occured: {e}')
	logger.exception('Full error traceback:')
	raise

# %% [markdown]
# ## Image registering & combine

def _split_group_by_exptime(group, max_sec):
	"""Split one time-contiguous group into subgroups whose cumulative EXPTIME
	stays <= max_sec, keeping frame time order (for --stack-sec)."""
	subs, cur, cur_sec = [], [], 0.0
	for inim in group:
		expt = float(fits.getheader(inim).get('EXPTIME', 0.0) or 0.0)
		if cur and cur_sec + expt > max_sec:
			subs.append(cur)
			cur, cur_sec = [], 0.0
		cur.append(inim)
		cur_sec += expt
	if cur:
		subs.append(cur)
	return subs


def _stack_group_worker(args):
	"""Process one (obj, filte, gidx, group) stacking task.

	All path/config globals are inherited via fork — no need to pass them.
	Pre-SExtractor runs sequentially to avoid nested-pool deadlocks.
	"""
	obj, filte, gidx, group = args

	if len(group) == 1:
		inim = group[0]
		comim = inim.replace('.fits', '.com.fits')
		os.system(f'cp {inim} {comim}')
		logger.info(f"    SKIP combine (single image): {os.path.basename(inim)}")
		return comim

	files_to_stack = group
	baseim = files_to_stack[0]

	# 1) Pre-SExtractor catalogs (sequential within worker)
	outcatlist = []
	for inim in files_to_stack:
		outcat = inim.replace('.fits', '.cat')
		run_pre_sextractor(inim, outcat, conf_simple, param_simple, conv_simple, nnw_simple, obsinfo['pixscale'], gain)
		outcatlist.append(outcat)

	# 2) SCAMP list files
	path_cat_scamp_list = f"{path_data}/cat.scamp.{obj}.{filte}.{gidx}.list"
	with open(path_cat_scamp_list, 'w') as s:
		for incat in outcatlist:
			s.write(f"{incat}\n")

	path_head_missfits_list = f"{path_data}/head.missfits.{obj}.{filte}.{gidx}.list"
	with open(path_head_missfits_list, 'w') as fp:
		for inim in files_to_stack:
			fp.write(f"{inim}\n")

	# 3) SCAMP (relative astrometric refinement)
	scampcom = f"scamp -c {path_config}/rasa36.scamp @{path_cat_scamp_list} {scamp_output_opts(f'B.{obj}.{filte}.{gidx}')}"
	logger.info(scampcom)
	run_shell_checked(scampcom, f"SCAMP ({obj}/{filte}/{gidx})")

	# 4) MissFITS (apply SCAMP solution to FITS headers)
	missfitscom = f"missfits @{path_head_missfits_list}"
	logger.info(missfitscom)
	run_shell_checked(missfitscom, f"MissFITS ({obj}/{filte}/{gidx})")

	# 4.5) Drop frames whose post-SCAMP WCS is bogus — one bad solution (shifted
	# center or garbage PV distortion terms) makes SWarp's AUTOMATIC canvas span
	# degrees (3 GB coadd from 5 frames, T00008 2026-06-24). SWarp reads the
	# .head sidecars too, so also quarantine those for dropped frames.
	kept = []
	for inim in files_to_stack:
		reason = check_wcs_sanity(inim, obsinfo['pixscale'], fov_deg)
		if reason is not None:
			logger.warning(f"  [{gidx}] DROP from stack - bad WCS ({reason}): {os.path.basename(inim)}")
			sidecar = inim.replace('.fits', '.head')
			if os.path.exists(sidecar):
				os.rename(sidecar, sidecar + '.badwcs')
		else:
			kept.append(inim)
	if len(kept) == 0:
		logger.error(f"  [{gidx}] SKIP stack - all {len(files_to_stack)} frame(s) have bogus WCS ({obj}/{filte})")
		return None
	if len(kept) == 1:
		inim = kept[0]
		comim = inim.replace('.fits', '.com.fits')
		os.system(f'cp {inim} {comim}')
		logger.info(f"  [{gidx}] SKIP combine (single valid frame after WCS filter): {os.path.basename(inim)}")
		return comim
	files_to_stack = kept
	baseim = files_to_stack[0]

	# 5) SWarp combine
	path_imagelist = f"{os.path.dirname(baseim)}/{os.path.basename(baseim).replace('fits', f'{obj}.{filte}.{gidx}.image.list')}"
	with open(path_imagelist, 'w') as f:
		for inim in files_to_stack:
			f.write(f"{inim}\n")

	mjdlist, exptimelist, airmasslist = [], [], []
	for inim in files_to_stack:
		with fits.open(inim) as hdulist:
			hdr = hdulist[0].header
			mjdlist.append(Time(hdr['DATE-OBS'], format='isot').mjd)
			exptimelist.append(hdr['EXPTIME'])
			airmasslist.append(hdr['AIRMASS'])

	exptime_combined = int(np.sum(exptimelist))
	mjd_combined = np.mean(mjdlist)
	jd_combined = Time(mjd_combined, format='mjd').jd
	dateobs_combined = Time(mjd_combined, format='mjd').isot
	airmass_combined = np.mean(airmasslist)

	with fits.open(baseim) as hdulist:
		hdr0 = hdulist[0].header
	objra = float(hdr0.get('RA', hdr0['CRVAL1']))
	objdec = float(hdr0.get('DEC', hdr0['CRVAL2']))
	center = f"{objra},{objdec}"

	datestr, timestr = extract_date_and_time(dateobs_combined)
	comim = f"{path_data}/Calib-{obs}-{obj}-{datestr}-{timestr}-{filte}-{exptime_combined}.com.fits"

	swarpcom = (
		f"SWarp -c {path_config}/rasa36.swarp @{path_imagelist} "
		f"-IMAGEOUT_NAME {comim} -RESAMPLE_DIR {path_data} "
		f"-WEIGHTOUT_NAME {path_data}/weight.{obj}.{filte}.{gidx}.fits "
		f"-XML_NAME {path_data}/swarp.{obj}.{filte}.{gidx}.xml "
		f"-CENTER_TYPE MANUAL -CENTER {center} -GAIN_KEYWORD EGAIN "
	)
	logger.info(swarpcom)
	run_shell_checked(swarpcom, f"SWarp ({obj}/{filte}/{gidx})")

	# Canvas sanity: a normal coadd is ~4300x4300 (~18 Mpix); much larger means a
	# bad WCS slipped past the filter and downstream stages will eat huge memory.
	_hdr_out = fits.getheader(comim)
	npix_out = _hdr_out['NAXIS1'] * _hdr_out['NAXIS2']
	if npix_out > 40_000_000:
		logger.error(f"  [{gidx}] combined canvas suspiciously large ({npix_out/1e6:.0f} Mpix) - check WCS of group: {[os.path.basename(f) for f in files_to_stack]}")

	# 6) Copy general header keywords from base image
	with fits.open(baseim) as hdulist:
		basehdr = hdulist[0].header
		chdr = {key: basehdr.get(key, None) for key in keywords_to_add}

	with fits.open(comim, mode='update') as hdul:
		hdr = hdul[0].header
		for key in chdr.keys():
			hdr[key] = chdr[key]
		hdul.flush()

	N_combine = len(files_to_stack)
	effgain = (2/3) * N_combine * gain  # gain is a global from obsinfo['gain'].value

	# SCAMP's achieved astrometric RMS (written to each frame's .head by SCAMP, merged in
	# by MissFITS) is otherwise dropped by SWarp. Propagate the median over the stacked
	# frames onto the coadd so per-image astrometric accuracy is auditable (vs Gaia).
	_astrom_acc = {'ASTRRMS1': [], 'ASTRRMS2': [], 'ASTIRMS1': [], 'ASTIRMS2': []}
	for _inim in files_to_stack:
		try:
			_fh = fits.getheader(_inim)
		except Exception:
			continue
		for _k in _astrom_acc:
			try:
				if _k in _fh:
					_astrom_acc[_k].append(float(_fh[_k]))
			except (ValueError, TypeError):
				pass
	_astrom_kw = {_k: (float(np.median(_v)), 'SCAMP astrom RMS [deg], median over frames')
	              for _k, _v in _astrom_acc.items() if _v}
	if 'ASTRRMS1' in _astrom_kw and 'ASTRRMS2' in _astrom_kw:
		_astrom_kw['ASTRMSAS'] = (float(np.hypot(_astrom_kw['ASTRRMS1'][0], _astrom_kw['ASTRRMS2'][0]) * 3600.0),
		                          'SCAMP ref astrom RMS vs Gaia [arcsec]')
	if any(_astrom_acc.values()):
		_astrom_kw['NASTRMS'] = (max(len(_v) for _v in _astrom_acc.values()), 'N frames with SCAMP astrom RMS')

	keywords_to_update = {
		'EGAIN'    : (effgain,          'Effective EGAIN, [e-/ADU] Electrons per A/D unit'),
		'FILTER'   : (filte,            'Active filter name'),
		'DATE-OBS' : (dateobs_combined,  'Time of observation (UTC) for combined image'),
		'EXPTIME'  : (exptime_combined,  '[s] Total exposure duration for combined image'),
		'EXPOSURE' : (exptime_combined,  '[s] Total exposure duration for combined image'),
		'AIRMASS'  : (airmass_combined,  'Average airmass at frame center for combined image (Gueymard 1993)'),
		'MJD'      : (mjd_combined,      'Modified Julian Date at start of observations for combined image'),
		'JD'       : (jd_combined,       'Julian Date at start of observations for combined image'),
		'NCOMBINE' : (N_combine,         'Number of frames combined'),
	}
	keywords_to_update.update(_astrom_kw)

	with fits.open(comim, mode='update') as hdul:
		header = hdul[0].header
		for key, (value, comment) in keywords_to_update.items():
			header[key] = (value, comment)
		# Add NOTE keyword for ToO images with the ToO event/folder name
		meta = obj_meta.get(obj, {'is_too': False, 'too_name': obj})
		if meta['is_too']:
			too_note = meta.get('too_name') or obj
			header['NOTE'] = (too_note, 'ToO event name / processed folder name')
		hdul.flush()

	# Write .head sidecar file so downstream tools (SCAMP, MissFITS, etc.) can read it
	head_path = comim.replace('.fits', '.head')
	with fits.open(comim) as hdul:
		hdul[0].header.totextfile(head_path, overwrite=True)

	logger.info(f"  [{gidx}] DONE stack -> {os.path.basename(comim)} (head: {os.path.basename(head_path)})")
	return comim


def _stack_group_worker_star(args):
	obj, filte, gidx, group = args
	try:
		return _stack_group_worker(args)
	except Exception as e:
		logger.error(f"stack worker failed for {obj}/{filte}/{gidx}: {e}")
		return None


# %%
try:
	logger.info("START image stack")
	st_ = time.time()
	combined_images = []
	step = 1e-10 if no_stack else (1/24/60)*60  # 1e-10: each frame its own group (no-stack); default: 1 hour
	ic_cal_phot = ImageFileCollection(path_data, glob_include='Calib*0.fits', keywords='*')
	calist = sorted(glob.glob('{}/Calib*0*fits'.format(path_data)))
	if ic_cal_phot.summary is None:
		logger.warning(f"No Calib*0.fits files found in {path_data}, skipping stack")
		objlist, filterlist = [], []
	else:
		objlist = sorted(list(set(ic_cal_phot.summary['object'])))
		filterlist = sorted(list(set(ic_cal_phot.summary['filter'])))
	# Collect all (obj, filte, gidx, group) tasks for parallel dispatch
	all_stack_args = []
	for obj in objlist:
		for filte in filterlist:
			imlist_tmp = sorted(glob.glob('{}/Calib*-{}-*-{}-*.fits'.format(path_data, obj, filte)))
			logger.info(f"IMAGE REGISTERING & COMBINE process on {imlist_tmp}")
			if len(imlist_tmp) == 0:
				pass
			elif len(imlist_tmp) == 1:
				all_stack_args.append((obj, filte, 1, imlist_tmp))
			else:
				logger.info(f"{obj} {filte}")
				jds = np.array([fits.getheader(inim)['jd'] for inim in imlist_tmp])
				delts = jds - np.min(jds)

				grouplist = []
				grouplists = []

				for i in range(len(delts)):
					if i == 0:
						t0 = delts[i]
					t1 = delts[i]
					dif = np.abs(t0-t1)

					if dif < step:
						grouplist.append(imlist_tmp[i])
					else:
						grouplists.append(grouplist)
						grouplist = [imlist_tmp[i]]
					t0 = t1
				grouplists.append(grouplist)

				if stack_sec:
					grouplists = [sub for g in grouplists for sub in _split_group_by_exptime(g, stack_sec)]
					logger.info(f"--stack-sec {stack_sec:g}: {obj} {filte} -> {len(grouplists)} stack group(s)")

				for gidx, group in enumerate(grouplists, 1):
					all_stack_args.append((obj, filte, gidx, group))

	logger.info(f"START parallel image stack — {len(all_stack_args)} group(s) across {len(objlist)} object(s) × {len(filterlist)} filter(s)")
	if __name__ == '__main__':
		if len(all_stack_args) <= 1:
			# Single group (typical ToO / most GRB runs): run in-process, no fork.
			# A one-task fork pool can't parallelize anything, yet every fork is a
			# chance to inherit a poisoned astropy/ERFA lock and wedge the worker in
			# check_wcs_sanity/the header-read window (2026-07-15 M8_ToO hung the full
			# 1800 s imap_collect timeout on its single group). In-process = no fork =
			# that wedge cannot happen. _stack_group_worker_star already returns None
			# on failure, matching the pool path's filter below.
			stack_results = [_stack_group_worker_star(a) for a in all_stack_args]
		else:
			with mp_fork.Pool(processes=ncores, initializer=_worker_log_init) as pool:
				stack_results = imap_collect(pool, _stack_group_worker_star, all_stack_args, desc="Stack groups", unit="grp")
	combined_images = [r for r in stack_results if r is not None]

	n_combined = len(combined_images)
	protbl['status'][protbl['process']=='image_stack'] = True
	protbl['time'][protbl['process']=='image_stack'] = int(time.time() - st_)
	logger.info(
		f"STACK SUMMARY | Combined: {n_combined} images | "
		f"Targets ({len(objlist)}): {objlist} | Filters: {filterlist}"
	)
	logger.info(f"END   image registration & combine - total {round(time.time()-st_,1)}s")
	try:
		_step_secs = int(protbl['time'][protbl['process']=='image_stack'].item())
		_fmt_objlist = ", ".join(str(o) for o in objlist) or "none"
		_fmt_filterlist = ", ".join(str(f) for f in filterlist) or "none"
		_n_stack_failed = len(all_stack_args) - n_combined
		_stack_lines = [
			f":white_check_mark: *[6/10] Stacking done* — {_step_secs}s",
			f"Combined: {n_combined}/{len(all_stack_args)} groups | Failed: {_n_stack_failed} | Targets ({len(objlist)}): {_fmt_objlist} | Filters: {_fmt_filterlist}",
		]
		tool.slack_bot(
			token=slack_token, channel=channel, thread_ts=slack_ts,
			text="\n".join(_stack_lines)
		)
	except Exception as _se:
		logger.warning(f"Slack notification failed: {_se}")
except Exception as e:
	logger.error(f'Unexpected error occured: {e}')
	logger.exception('Full error traceback:')
	raise

# PNG previews for stacked images
try:
	logger.info("START PNG generation for stacked images")
	st_ = time.time()
	from concurrent.futures import ThreadPoolExecutor
	with ThreadPoolExecutor(max_workers=ncores) as executor:
		list(tqdm(
			executor.map(save_fits_as_png, combined_images),
			total=len(combined_images), desc="PNG (stacked)", unit="img"
		))
	logger.info(f"END   PNG generation for stacked images - total {round(time.time()-st_,1)}s")
except Exception as e:
	logger.warning(f"PNG generation for stacked images failed: {e}")

# PNG previews for non-stacked calibrated images
try:
	logger.info("START PNG generation for calibrated (non-stacked) images")
	st_ = time.time()
	# Individual calibrated science frames (Calib*0.fits); .com.fits stacks are excluded
	# by the trailing "0.fits" pattern. These get moved to the output dir alongside the
	# stacked/subtracted PNGs by the file-transfer catch-all.
	calib_png_list = sorted(glob.glob(f'{path_data}/Calib*0.fits'))
	from concurrent.futures import ThreadPoolExecutor
	with ThreadPoolExecutor(max_workers=ncores) as executor:
		list(tqdm(
			executor.map(save_fits_as_png, calib_png_list),
			total=len(calib_png_list), desc="PNG (calibrated)", unit="img"
		))
	logger.info(f"END   PNG generation for calibrated (non-stacked) images - total {round(time.time()-st_,1)}s")
except Exception as e:
	logger.warning(f"PNG generation for calibrated (non-stacked) images failed: {e}")

	# %%
	# images_to_align = group
	# ref_image = images_to_align[0]

	# outim = tool.imcombine_routine(images_to_align, ref_image)

	# %% [markdown]
	# ## Photometry for combined images

	# %%
try:
	logger.info("START photometry for combined images")
	st_ = time.time()
	#	Write photometry configuration
	h = open(path_new_gphot, 'w')
	for line in lines:
		if 'imkey' in line:
			line = '{}\t{}/C*com.fits'.format('imkey', path_data)
		else:
			pass
		h.write(line+'\n')
	h.close()
	#	Execute
	path_phot = path_phot_mp
	com = 'python {} {} {}'.format(path_phot, path_data, ncores)
	logger.info(com)
	os.system(com)
	#	Same silent-failure guard as the single-frame photometry stage.
	_n_pin = len(glob.glob(f'{path_data}/C*com.fits'))
	_n_pout = len(glob.glob(f'{path_data}/C*com.phot.cat'))
	if _n_pin > 0 and _n_pout == 0:
		logger.error(f"Combined-image photometry produced no phot.cat for any of {_n_pin} coadds - check refcat coverage / gregoryphot output")

	# tdict['photometry_com'] = time.time() - st - tdict[list(tdict.keys())[-1]]
	protbl['status'][protbl['process']=='photometry_com'] = True
	protbl['time'][protbl['process']=='photometry_com'] = int(time.time() - st_)
	try:
		_step_secs = int(protbl['time'][protbl['process']=='photometry_com'].item())
		tool.slack_bot(
			token=slack_token, channel=channel, thread_ts=slack_ts,
			text="\n".join([
				f":white_check_mark: *[7/10] Photometry (combined) done* — {_step_secs}s",
				f"{n_combined} combined frames processed",
			])
		)
	except Exception as _se:
		logger.warning(f"Slack notification failed: {_se}")

	ic_com_phot = ImageFileCollection(path_data, glob_include='Calib*com.fits', keywords='*')
	logger.info('Draw observation summary plots')
	# for filte in list(set(ic_cal_phot.summary['filter'])):
	for filte in tqdm(filterlist, desc="Summary plots", unit="filter"):
		try:
			tool.obs_summary(filte, ic_cal_phot, ic_com_phot, path_save=path_data)
		except Exception as _pe:
			logger.warning(f'Fail to make summary plots: {_pe}')
		plt.close('all')
	logger.info(f"END   photometry for combined images - total {round(time.time()-st_,1)}s")
except Exception as e:
	logger.error(f'Unexpected error occured: {e}')
	logger.exception('Full error traceback:')
	raise

	# %% [markdown]
	# #	Image subtraction ==================================================================

	# %%
try:
	logger.info("START image subtraction")
	st_ = time.time()
	def _prestaged_to_cd(refpath):
		# Becker wcsremap reads only CDELT and misreads the pre-staged templates' PC+CDELT
		# WCS as 3600"/px ("scaled by 0.00000"). Rewrite the WCS as a CD matrix into the
		# factory (wiped per run) so wcsremap registers it correctly. No-op if already CD.
		try:
			with fits.open(refpath) as _hl:
				_h = _hl[0].header
				if 'CD1_1' in _h:
					return refpath
				_d = _hl[0].data
				_cd1 = _h.pop('CDELT1', 1.0); _cd2 = _h.pop('CDELT2', 1.0)
				_p11 = _h.pop('PC1_1', 1.0); _p12 = _h.pop('PC1_2', 0.0)
				_p21 = _h.pop('PC2_1', 0.0); _p22 = _h.pop('PC2_2', 1.0)
				_h['CD1_1'] = _cd1 * _p11; _h['CD1_2'] = _cd1 * _p12
				_h['CD2_1'] = _cd2 * _p21; _h['CD2_2'] = _cd2 * _p22
				_out = f'{path_data}/cd_{os.path.basename(refpath)}'
				fits.writeto(_out, _d, _h, overwrite=True)
				return _out
		except Exception as _e:
			logger.warning(f"  CD-normalize failed for {os.path.basename(refpath)}: {_e}")
			return refpath

	def _img_center(hdr):
		#	SkyCoord of an image's center from its WCS (None if not computable).
		try:
			_c = WCS(hdr).pixel_to_world(hdr['NAXIS1'] / 2.0, hdr['NAXIS2'] / 2.0)
			return _c if hasattr(_c, 'separation') else None
		except Exception:
			return None

	def _resolve_refim(obj, filte, sci_center=None):
		# Two-tier reference lookup, shared by subtraction and transient search.
		# Use the earliest prior TOS observation of the same tile as reference.
		# OBJECT is always the tile ID, so this works for both TOS and ToO modes.
		def _overlaps(refpath):
			#	Guard against name-matched but wrong-sky templates. Reference lookup
			#	matches Ref-/TOS files by the OBJECT string only; legacy off-grid nights
			#	(obsolete tile numbering) whose OBJECT maps to a DIFFERENT position in the
			#	current displaycenter_new grid got paired with a survey template ~80 deg
			#	away -> empty wcsremap overlap -> all-zero hd + junk candidates, silently
			#	(found 2026-07-20 on T00464: science Dec -20.4, template Dec -59.1). Reject
			#	a template whose center is >0.5 deg from the science field center. Fail-open
			#	when a center can't be computed so a WCS read hiccup never regresses the
			#	normal path (a real overlap is <~0.3 deg; the failure mode is tens of deg).
			if sci_center is None:
				return True
			try:
				with fits.open(refpath) as _hl:
					_rc = _img_center(_hl[0].header)
				if _rc is None:
					return True
				_sep = sci_center.separation(_rc).deg
				if _sep > 0.5:
					logger.warning(f"  SKIP reference {os.path.basename(refpath)}: center {_sep:.2f} deg from {obj} field (wrong-sky / legacy tile?)")
					return False
				return True
			except Exception as _e:
				logger.warning(f"  overlap check failed for {os.path.basename(refpath)}: {_e}")
				return True
		refimlist = sorted(glob.glob(f'{path_result}/TOS/{obj}/*/{filte}/Calib*.com.fits'))
		refimlist = [r for r in refimlist if f'/{date_str}/' not in r]
		refimlist = [r for r in refimlist if _overlaps(r)]
		if refimlist:
			return refimlist[0], 'TOS'
		# No prior TOS epoch of this tile -> fall back to a pre-staged external
		# survey template (DESI/SkyMapper), named Ref-RASA36-<tile>-<filter>-<survey>.fits.
		prestaged = sorted(glob.glob(f'{path_refimg}/Ref-{obs.upper()}-{obj}-{filte}-*.fits'))
		# Ignore .wcsremap.fits intermediates that may linger from an interrupted run.
		prestaged = [r for r in prestaged if '.wcsremap.' not in os.path.basename(r)]
		prestaged = [r for r in prestaged if _overlaps(r)]
		if prestaged:
			return prestaged[0], 'PRESTAGED'
		return None, None

	def _target_label(obj, filte):
		#	Label for subtraction/transient-search summaries: ToO targets show
		#	their name (NOTES header) alongside the tile ID, e.g. "GRB260421B (T02094) r".
		meta = obj_meta.get(obj, {'is_too': False, 'too_name': obj})
		if meta['is_too'] and meta.get('too_name') and meta['too_name'] != obj:
			return f"{meta['too_name']} ({obj}) {filte}"
		return f"{obj} {filte}"

	def _subtract_one(inim):
		#	Returns (subim, ds9com, obj, filte, ref_src, ref_base) so the parent can
		#	report per-tile outcomes (which tiles were subtracted, against which
		#	reference) in the log and Slack. subim is None when no reference exists
		#	(ref_base None too) or when wcsremap/hotpants produced nothing usable.
		obj = filte = '?'
		ref_src = ref_base = None
		try:
			hdr = fits.getheader(inim)
			obj = hdr['object']   # tile ID (e.g. T00123) for both TOS and ToO
			filte = hdr['filter']
			refim, ref_src = _resolve_refim(obj, filte, sci_center=_img_center(hdr))
			if refim is None:
				logger.warning(f"	NO reference image for {os.path.basename(inim)} in TOS/{obj}/ or {path_refimg}")
				return (None, None, obj, filte, None, None)
			ref_base = os.path.basename(refim)   # original template name, not the cd_ factory copy
			if ref_src == 'PRESTAGED':
				refim = _prestaged_to_cd(refim)
			logger.info(f"  REF[{ref_src}] {os.path.basename(refim)} -> {os.path.basename(inim)}")
			# wcsremap-registered reference + hotpants difference imaging.
			# subtraction_routine2 returns the wcsremap'd reference (intermediate);
			# the actual difference image written by hotpants is hd<inim>.
			wcsremap_ref, ds9com = tool.subtraction_routine2(inim, refim)
			if wcsremap_ref is not None and os.path.exists(wcsremap_ref):
				os.system(f"rm {wcsremap_ref}")
			subim = f"{os.path.dirname(inim)}/hd{os.path.basename(inim)}"
			if os.path.exists(subim) and os.path.getsize(subim) != 0:
				#	Record the template on the difference image so downstream stages
				#	(transient search) use the same reference without re-deriving it.
				fits.setval(subim, 'REFIM', value=refim, comment='subtraction template image')
				fits.setval(subim, 'REFSRC', value=ref_src, comment='template source: TOS or PRESTAGED')
				return (subim, ds9com, obj, filte, ref_src, ref_base)
			#	hotpants failed or was killed by its timeout: drop any partial hd/hc so
			#	no empty FITS reaches the transfer / transient-search stages (2026-07-14:
			#	an abandoned M8 subtraction shipped a 0-byte hd to the ToO output dir).
			for _f in (subim, f"{os.path.dirname(inim)}/hc{os.path.basename(inim)}"):
				if os.path.exists(_f):
					os.remove(_f)
			logger.warning(f"  Subtraction produced no usable difference image for {os.path.basename(inim)}")
			return (None, None, obj, filte, ref_src, ref_base)
		except Exception as _e:
			#	Per-task isolation: imap_collect re-raises worker exceptions on
			#	.next(), so an unguarded frame would abort the whole stage.
			logger.error(f"  Subtraction failed for {os.path.basename(inim)}: {_e}")
			return (None, None, obj, filte, ref_src, ref_base)

	subtracted_images = []
	ds9comlist = []
	if __name__ == '__main__':
		with mp_fork.Pool(processes=ncores) as pool:
			sub_results = imap_collect(pool, _subtract_one, combined_images, desc="Subtraction", unit="img")
	sub_records = [r for r in sub_results if r is not None]
	for _subim, _ds9com, _obj, _filte, _src, _refb in sub_records:
		if _subim is not None:
			subtracted_images.append(_subim)
			ds9comlist.append(_ds9com)

	#	Per-tile outcome summary — which tiles were subtracted against which
	#	reference (and which weren't, and why). Aggregated over frames sharing
	#	the same tile/filter/reference so ToO --no-stack runs stay one line.
	_sub_groups = {}
	for _subim, _ds9com, _obj, _filte, _src, _refb in sub_records:
		_outcome = 'OK' if _subim is not None else ('NOREF' if _refb is None else 'FAIL')
		_key = (_obj, _filte, _outcome, _src, _refb)
		_sub_groups[_key] = _sub_groups.get(_key, 0) + 1
	sub_summary_lines = []
	for (_obj, _filte, _outcome, _src, _refb), _n in sorted(_sub_groups.items()):
		_cnt = f' ({_n} images)' if _n > 1 else ''
		if _outcome == 'OK':
			sub_summary_lines.append(f"{_target_label(_obj, _filte)}: subtracted, ref [{_src}] {_refb}{_cnt}")
		elif _outcome == 'NOREF':
			sub_summary_lines.append(f"{_target_label(_obj, _filte)}: NOT subtracted - no reference image{_cnt}")
		else:
			sub_summary_lines.append(f"{_target_label(_obj, _filte)}: FAILED (wcsremap/hotpants), ref [{_src}] {_refb}{_cnt}")
	if len(sub_records) < n_combined:
		sub_summary_lines.append(f"{n_combined - len(sub_records)} image(s) abandoned (worker stall timeout)")

	rmcom = 'rm {}/*Ref*gregister.fits'.format(path_data)
	logger.info(rmcom)
	os.system(rmcom)
	# tdict['subtraction'] = time.time() - st - tdict[list(tdict.keys())[-1]]
	n_subtracted = len(subtracted_images)
	protbl['status'][protbl['process']=='subtraction'] = True
	protbl['time'][protbl['process']=='subtraction'] = int(time.time() - st_)
	logger.info(f"SUBTRACTION SUMMARY | Subtracted: {n_subtracted}/{n_combined}")
	for _l in sub_summary_lines:
		logger.info(f"  {_l}")
	logger.info(f"END image subtraction - total {round(time.time()-st_,1)}s")
	try:
		_step_secs = int(protbl['time'][protbl['process']=='subtraction'].item())
		_slack_lines = [
			f":white_check_mark: *[8/10] Subtraction done* — {_step_secs}s",
			f"Subtracted: {n_subtracted}/{n_combined} images",
		]
		#	Per-tile reference detail; capped to keep the thread message readable.
		_MAX_SUB_LINES = 25
		_slack_lines += [f"• {_l}" for _l in sub_summary_lines[:_MAX_SUB_LINES]]
		if len(sub_summary_lines) > _MAX_SUB_LINES:
			_slack_lines.append(f"… and {len(sub_summary_lines) - _MAX_SUB_LINES} more tiles (see log)")
		tool.slack_bot(
			token=slack_token, channel=channel, thread_ts=slack_ts,
			text="\n".join(_slack_lines)
		)
	except Exception as _se:
		logger.warning(f"Slack notification failed: {_se}")
except Exception as e:
	logger.error(f'Unexpected error occured: {e}')
	logger.exception('Full error traceback:')
	raise

# PNG previews for subtracted images
try:
	logger.info("START PNG generation for subtracted images")
	st_ = time.time()
	sub_png_list = [f for f in subtracted_images if f and os.path.exists(f)]
	from concurrent.futures import ThreadPoolExecutor
	with ThreadPoolExecutor(max_workers=ncores) as executor:
		list(tqdm(
			executor.map(save_fits_as_png, sub_png_list),
			total=len(sub_png_list), desc="PNG (subtracted)", unit="img"
		))
	logger.info(f"END   PNG generation for subtracted images - total {round(time.time()-st_,1)}s")
except Exception as e:
	logger.warning(f"PNG generation for subtracted images failed: {e}")

	# %% [markdown]
	# ##	Photometry for subtracted images

	# %%
try:
	logger.info("START photometry on subtracted images")
	st_ = time.time()
	#	Write photometry configuration

	s = open(path_new_gphot, 'w')
	for line in lines:
		if 'imkey' in line:
			# line = '{}\t{}/hd*com.fits'.format('imkey', path_data)
			line = '{}\t{}/hd*.fits'.format('imkey', path_data)
		else:
			pass
		if 'photfraction' in line:
			line = '{}\t{}'.format('photfraction', 1.0)
		else:
			pass
		if 'DETECT_MINAREA' in line:
			line = '{}\t{}'.format('DETECT_MINAREA', 10)
		else:
			pass
		if 'DETECT_THRESH' in line:
			line = '{}\t{}'.format('DETECT_THRESH', 1.25)
		else:
			pass
		s.write(line+'\n')
	s.close()
	#	Execute
	hdimlist = sorted(glob.glob('{}/hd*.fits'.format(path_data)))
	if len(hdimlist) > 0:
		com = 'python {} {}'.format(path_phot_sub, path_data)
		logger.info(com)
		os.system(com)
		# tdict['photometry_sub'] = time.time() - st - tdict[list(tdict.keys())[-1]]
		logger.info(f"  PROCESS photometry_sub - input: {os.path.basename(path_data)}")
	else:
		logger.info('No subtracted image')
		pass
	protbl['status'][protbl['process']=='photometry_sub'] = True
	protbl['time'][protbl['process']=='photometry_sub'] = int(time.time() - st_)
	logger.info(f"END   photometry on subtracted images - total {round(time.time()-st_,1)}s")
	try:
		_step_secs = int(protbl['time'][protbl['process']=='photometry_sub'].item())
		tool.slack_bot(
			token=slack_token, channel=channel, thread_ts=slack_ts,
			text="\n".join([
				f":white_check_mark: *[9/10] Photometry (subtracted) done* — {_step_secs}s",
				f"{n_subtracted} subtracted frames processed",
			])
		)
	except Exception as _se:
		logger.warning(f"Slack notification failed: {_se}")
except Exception as e:
	logger.error(f'Unexpected error occured: {e}')
	logger.exception('Full error traceback:')
	raise

	# %% [markdown]
	# ##	Transient Search
	# %%
try:
	logger.info("START Transient Search")
	st_ = time.time()
	fovval = fov.value
	#	Input table for transient search. Searched: every ToO target, and TOS tiles
	#	whose subtraction used a prior TOS epoch. TOS tiles subtracted against a
	#	survey (PRESTAGED) template are skipped — those difference images carry
	#	~46% stellar dipole residuals and flood the search with junk candidates;
	#	the hd image itself is still kept.
	hdimlist = sorted(glob.glob(f'{path_data}/hd*.fits'))
	ts_rows = []
	ts_meta = []      # (hdcat, group key) per searched image, to audit gregoryfind output
	n_ts_skip = 0
	n_ts_err = 0
	_ts_groups = {}   # (obj, filte, ref basename) -> N searched images, for log/Slack
	for hdim in tqdm(hdimlist, desc="Transient table", unit="img"):
		hdcat = hdim.replace('.fits','.phot_sub.cat')
		hcim = hdim.replace('hdCalib', 'hcCalib')
		inim = hdim.replace('hdCalib', 'Calib')
		scicat = inim.replace('.fits', '.phot.cat')
		try:
			hdr = fits.getheader(hdim)
			obj = hdr['object']
			filte = hdr['filter']
			#	Template recorded by the subtraction stage; fall back to the shared
			#	two-tier lookup for hd images written before REFIM stamping existed.
			refim = hdr.get('REFIM') or None
			ref_src = hdr.get('REFSRC', '')
			if refim is None:
				refim, ref_src = _resolve_refim(obj, filte, sci_center=_img_center(hdr))
			if refim is None:
				raise FileNotFoundError(f'no reference image for {obj} {filte} in TOS or {path_refimg}')
			is_too = obj_meta.get(obj, {'is_too': False})['is_too']
			if not is_too and ref_src != 'TOS':
				logger.info(f"  Skip transient search ({ref_src or 'unknown'} template on TOS tile): {os.path.basename(hdim)}")
				n_ts_skip += 1
				continue
			ts_rows.append([hdim, hdcat, hcim, inim, scicat, refim, fovval])
			#	Strip the cd_ prefix of the factory CD-converted copy so the summary
			#	names the actual template in reference_img/.
			_key = (obj, filte, os.path.basename(refim).removeprefix('cd_'))
			_ts_groups[_key] = _ts_groups.get(_key, 0) + 1
			ts_meta.append((hdcat, _key))
		except Exception as e:
			n_ts_err += 1
			logger.error(f"[ERROR] {hdim}: {e}")
			l = open(hdim+'.error.txt', 'w')
			l.write(str(e))
			l.close()

	n_ts_fail = 0
	_ts_cand = {}   # group key -> total transient candidates (flag==False rows)
	_ts_fail = {}   # group key -> N images whose gregoryfind routine failed
	if ts_rows:
		tstbl = Table(rows=ts_rows, names=['hdim', 'hdcat', 'hcim', 'inim', 'scicat', 'refim', 'fovval'])
		out_tstbl = f'{path_data}/transient_search.txt'
		tstbl.write(out_tstbl, format='ascii.tab', overwrite=True)

		com = f'python {path_find} {out_tstbl} {ncores}'
		logger.info(com)
		subprocess.call(com, shell=True)

		#	Audit gregoryfind output: it writes <hd>.transients.cat for every image
		#	it processes, even source-free ones (empty-catalog guard, 2026-07-15).
		#	A missing catalog therefore means the per-image routine failed — its
		#	traceback only goes to gregoryfind stdout, invisible to this log, so
		#	count it here instead of reporting a misleading "errors: 0".
		for _hdcat, _key in ts_meta:
			_outcat = _hdcat.replace('phot_sub.cat', 'transients.cat')
			if os.path.exists(_outcat):
				try:
					_tct = ascii.read(_outcat)
					#	flag column reads back as str ('True'/'False') from ascii.tab
					_ncand = int(np.sum(_tct['flag'].astype(str) == 'False')) if len(_tct) else 0
				except Exception as _ce:
					logger.warning(f"  Could not count candidates in {os.path.basename(_outcat)}: {_ce}")
					_ncand = 0
				_ts_cand[_key] = _ts_cand.get(_key, 0) + _ncand
			else:
				n_ts_fail += 1
				_ts_fail[_key] = _ts_fail.get(_key, 0) + 1
	n_ts_cand = sum(_ts_cand.values())

	protbl['status'][protbl['process']=='transient_search'] = True
	protbl['time'][protbl['process']=='transient_search'] = int(time.time() - st_)
	#	Per-tile searched summary (which difference images, against which reference),
	#	so a silent no-op search is distinguishable from an error.
	ts_summary_lines = []
	for _grpkey, _n in sorted(_ts_groups.items()):
		_obj, _filte, _refb = _grpkey
		_cnt = f' ({_n} images)' if _n > 1 else ''
		_line = f"{_target_label(_obj, _filte)}: searched vs {_refb}{_cnt} — {_ts_cand.get(_grpkey, 0)} candidates"
		if _grpkey in _ts_fail:
			_line += f", {_ts_fail[_grpkey]} image(s) FAILED in gregoryfind"
		ts_summary_lines.append(_line)
	logger.info(
		f"TRANSIENT SEARCH SUMMARY | searched: {len(ts_rows)}/{len(hdimlist)} difference images, "
		f"candidates: {n_ts_cand}, skipped (survey template on TOS tile): {n_ts_skip}, "
		f"errors: {n_ts_err}, gregoryfind failures: {n_ts_fail}"
	)
	for _l in ts_summary_lines:
		logger.info(f"  {_l}")
	logger.info(f"END   Transient Search:  total {round(time.time()-st_,1)}s")
	try:
		_step_secs = int(protbl['time'][protbl['process']=='transient_search'].item())
		_slack_lines = [
			f":white_check_mark: *[10/10] Transient search done* — {_step_secs}s",
			f"Searched: {len(ts_rows)}/{len(hdimlist)} difference images "
			f"(skipped: {n_ts_skip}, errors: {n_ts_err}, gregoryfind failures: {n_ts_fail})",
			f"Transient candidates: {n_ts_cand}",
		]
		_MAX_TS_LINES = 25
		_slack_lines += [f"• {_l}" for _l in ts_summary_lines[:_MAX_TS_LINES]]
		if len(ts_summary_lines) > _MAX_TS_LINES:
			_slack_lines.append(f"… and {len(ts_summary_lines) - _MAX_TS_LINES} more tiles (see log)")
		tool.slack_bot(
			token=slack_token, channel=channel, thread_ts=slack_ts,
			text="\n".join(_slack_lines)
		)
	except Exception as _se:
		logger.warning(f"Slack notification failed: {_se}")
except Exception as e:
	logger.error(f'Unexpected error occured: {e}')
	logger.exception('Full error traceback:')
	raise

# %% [markdown]
# #	Summary file

# %%
#------------------------------------------------------------
#------------------------------------------------------------
protbl['status'][protbl['process']=='total'] = True
protbl['time'][protbl['process']=='total'] = int(time.time() - st)
total_time = round(protbl['time'][protbl['process']=='total'].item()/60., 1)
end_localtime = time.strftime('%Y-%m-%d %H:%M:%S (%Z)', time.localtime())
summary_lines = [
	'='*60,
	f'Pipeline start : {start_localtime}',
	f'Pipeline end   : {end_localtime}',
	f'Total time     : {total_time} min',
	'='*60,
	f'PATH   : {path}',
	f'Cores  : {ncores}',
	'='*60,
	'CALIBRATION FRAMES',
	f'  Bias : {biasnumb} frames',
	f'  Dark : {darknumb} exp-times {[int(e) for e in darkexptimelist]}',
	f'  Flat : {len(flatfilterlist)} filters {flatfilterlist}',
	'='*60,
	'SCIENCE IMAGES',
	f'  Raw object frames  : {nobj}',
	f'  Calibrated images  : {n_calib_images}',
	f'  Astrometry OK      : {n_astro_ok}/{n_calib_images}',
	f'  Astrometry failed  : {n_astro_fail}',
	f'  Stacked (combined) : {n_combined}',
	f'  Subtracted         : {n_subtracted}',
	'='*60,
	f'TARGETS ({len(objlist)}) : {objlist}',
	f'FILTERS : {filterlist}',
]
try:
	summary_lines.append('='*60)
	objkind = sorted(set(ic_cal.summary['object']))
	for obj in objkind:
		summary_lines.append('-'*60)
		for filte in list(set(ic_cal.summary['filter'])):
			indx_tmp = ic_cal.files_filtered(filter=filte, object=obj)
			if len(indx_tmp) > 0:
				summary_lines.append(f'  {obj}\t{filte}\t{len(indx_tmp)} frames')
except Exception:
	pass
summary_lines.append('='*60)
summary_lines.append('STAGE TIMES')
for row in protbl:
	if row['process'] != 'total' and row['status']:
		summary_lines.append(f"  {row['process']}\t: {row['time']:.0f}s")
summary_lines.append('='*60)
logger.info('\n' + '\n'.join(summary_lines))

# %% [markdown]
# ## File Transfer

# %%
_processed_dirs = set()
try:
	rmcom = 'rm {}/inv*.*'.format(path_data, path_data)
	logger.info(rmcom)
	os.system(rmcom)
	# Regenerate each single frame's .head sidecar with its FULL merged header.
	# SCAMP writes a WCS-only .head in Pass A; MissFITS merges it into the .fits,
	# but the .head itself stays a WCS-only stub. Photometry has already consumed
	# it (SExtractor HEADER_SUFFIX='.head') by this point, so it is safe to
	# overwrite with the complete header — this makes the single/ .head
	# self-descriptive and consistent with the coadd .com.head. Excludes .com.*
	# (coadds keep their own full .com.head, written at stack time).
	single_fits = [
		f for f in glob.glob(f'{path_data}/*Calib*.fits')
		if '.com.' not in os.path.basename(f)
	]
	_nhead = 0
	for _sf in single_fits:
		_hp = _sf.replace('.fits', '.head')
		try:
			with fits.open(_sf) as _hl:
				_hl[0].header.totextfile(_hp, overwrite=True)
			_nhead += 1
		except Exception as _e:
			logger.warning(f"Could not regenerate full .head for {os.path.basename(_sf)}: {_e}")
	logger.info(f"Regenerated {_nhead}/{len(single_fits)} single-frame .head sidecars with full header")
	# Move order matters: transient products first, then stacked (.com.) products
	# to the top level, then the '' catch-all sweeps per-frame files into single/
	tails = ['.transients.', '.new.', '.ref.', '.sub.', '.com.', '']
	for obj in tqdm(objlist, desc="File transfer", unit="obj"):
		meta = obj_meta.get(obj, {'is_too': False, 'objtype': 'TOS', 'too_name': obj})
		for filte in tqdm(filterlist, desc=f"  Transfer [{obj}]", unit="filter", leave=False):
			for tail in tails:
				if meta['is_too']:
					# ToO: /data6/processed_RASA/ToO/{date}/{too_name}/{filte}
					too_folder = meta.get('too_name') or obj
					base = f'{path_result}/ToO/{date_str}/{too_folder}/{filte}'
				else:
					base = f'{path_result}/{meta["objtype"]}/{obj}/{date_str}/{filte}'
				if tail == '.com.':
					pathto = base
				elif tail == '':
					pathto = f'{base}/single'
				else:
					pathto = f'{base}/transients'
				files = f'{path_data}/*Calib*-{obj}-*-{filte}-*{tail}*'
				nfiles = len(glob.glob(files))
				if nfiles != 0:
					if not os.path.exists(pathto):
						os.makedirs(pathto)
					mvcom = f'mv {files} {pathto}'
					logger.info(mvcom)
					os.system(mvcom)
					_processed_dirs.add(base)
			# Also move .head sidecar files for stacked images (same destination as the images)
			head_files = f'{path_data}/*Calib*-{obj}-*-{filte}-*.com.head'
			if len(glob.glob(head_files)) > 0:
				if meta['is_too']:
					too_folder = meta.get('too_name') or obj
					head_dest = f'{path_result}/ToO/{date_str}/{too_folder}/{filte}'
				else:
					head_dest = f'{path_result}/{meta["objtype"]}/{obj}/{date_str}/{filte}'
				os.makedirs(head_dest, exist_ok=True)
				os.system(f'mv {head_files} {head_dest}')
			# Move PNG previews (stacked and subtracted) for this object/filter
			for png_pattern in (
				f'{path_data}/*Calib*-{obj}-*-{filte}-*.com.png',
				f'{path_data}/hd*Calib*-{obj}-*-{filte}-*.png',
			):
				if glob.glob(png_pattern):
					if meta['is_too']:
						too_folder = meta.get('too_name') or obj
						png_dest = f'{path_result}/ToO/{date_str}/{too_folder}/{filte}'
					else:
						png_dest = f'{path_result}/{meta["objtype"]}/{obj}/{date_str}/{filte}'
					os.makedirs(png_dest, exist_ok=True)
					os.system(f'mv {png_pattern} {png_dest}')
	#	Delete raw duplicates (originals remain in the input folder)
	for ext in ('*.fts', '*.fits', '*.fit'):
		os.system(f'rm -f {path_data}/{ext}')

	logger.info(f'DONE processing {path}')
	shutil.rmtree(path_data, ignore_errors=True)
	logger.info(f'Cleaned up factory directory: {path_data}')

	_stage_times = " | ".join(
		f"{row['process']}: {row['time']:.0f}s"
		for row in protbl
		if row['process'] != 'total' and row['status']
	)
	_fmt_objlist_fin = ", ".join(str(o) for o in objlist) if objlist else "N/A"
	_fmt_filterlist_fin = ", ".join(str(f) for f in filterlist) if filterlist else "N/A"
	_fmt_flatlist_fin = ", ".join(str(f) for f in flatfilterlist) if flatfilterlist else "none"
	# Surface any wedged-worker abandonments (imap_collect stalls) so a silently
	# dropped frame/group is visible in Slack, not just the log (2026-07-15).
	if POOL_ABANDONED:
		_abandon_by_stage = {}
		for _stage, _n in POOL_ABANDONED:
			_abandon_by_stage[_stage] = _abandon_by_stage.get(_stage, 0) + _n
		_abandon_str = ", ".join(f"{_s} {_n}" for _s, _n in _abandon_by_stage.items())
		_abandon_line = f":warning: *Abandoned {sum(_abandon_by_stage.values())} task(s)* to wedged workers — {_abandon_str}"
	else:
		_abandon_line = None
	text = "\n".join(line for line in [
		f":checkered_flag: [`gpPy`/RASA36] *Pipeline Complete* — {os.path.basename(path)}",
		f"*Start:* {start_localtime}  →  *End:* {end_localtime}  |  *Total:* {total_time} min  |  *Cores:* {ncores}",
		f"",
		f"*Calibration:*  Bias {biasnumb} frames | Dark {darknumb} exp-times {[int(e) for e in darkexptimelist]} | Flat {len(flatfilterlist)} filters [{_fmt_flatlist_fin}]",
		f"*Science:*  {nobj} raw → {n_calib_images} calibrated → {n_astro_ok}/{n_calib_images} astrometry → {n_combined} stacked → {n_subtracted} subtracted",
		f"*Targets ({len(objlist)}):*  {_fmt_objlist_fin}  |  *Filters:*  {_fmt_filterlist_fin}",
		_abandon_line,
		f"",
		f"*Stage times:*  {_stage_times}",
	] if line is not None)
	logger.info(f"\n{'='*60}\nPIPELINE COMPLETE SUMMARY\n{text}\n{'='*60}")

except Exception as e:
	logger.error(f'Unexpected error occured: {e}')
	logger.exception('Full error traceback:')
	raise

# %% [markdown]
# ##	Slack message

# %%
#total_time = round(protbl['time'][protbl['process']=='total'].item()/60., 1)
logger.info(f'Total processing time: {total_time} minutes')

channel = '#pipeline--imsng'

param_slack = dict(
	token = slack_token,
	channel = channel,
	text = text,
	thread_ts = slack_ts,
)

tool.slack_bot(**param_slack)

logger.info(f'Process ended at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
stop_listener()
logging.shutdown()

# Copy the completed log file into each processed filter directory
for dest_dir in _processed_dirs:
	try:
		os.makedirs(dest_dir, exist_ok=True)
		shutil.copy2(_logfile_path, dest_dir)
	except Exception:
		pass  # log copy failure is non-fatal


# %%
