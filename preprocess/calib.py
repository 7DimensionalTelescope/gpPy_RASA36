#============================================================
#   CALIBTRATION ROUTINE FUNCTION FOR IMSNG TELESCOPES
#	LOAO, DOAO, SOAO, ...
#   18.12.04	UPDATED BY Gregory S.H. Paek
#	19.08.08	CREATED BY Gregory S.H. Paek	(FUNCTIONALIZATION)
#	25.12.23
#============================================================
#	FUNCTION
#============================================================
import os, glob
import logging
import numpy as np
from astropy import units as u
from astropy.io import fits
import astropy.io.ascii as ascii
import sys
import datetime
from astropy.nddata import CCDData
import ccdproc
import matplotlib.pyplot as plt
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from util import tool
from util import robust

logger = logging.getLogger(__name__)

def bn_median(masked_array, axis=None):
	"""
	Perform fast median on masked array

	Parameters

	masked_array : `numpy.ma.masked_array`
		Array of which to find the median.

	axis : int, optional
		Axis along which to perform the median. Default is to find the median of
		the flattened array.
	"""
	import numpy as np
	import bottleneck as bn
	data = masked_array.filled(fill_value=np.nan)
	med = bn.nanmedian(data, axis=axis)
	# construct a masked array result, setting the mask from any NaN entries
	return np.ma.array(med, mask=np.isnan(med))

def folder_check(datalist, path_raw, path_factory):
	'''
	CHECK NEW OBSERVATION DATA FOLDER
	'''
	newlist = []
	#	IS THERE NEW FILE?
	for data in glob.glob(path_raw+'/*'):
		if data not in datalist:
			logger.info('NEW DATA\t: {}'.format(data))
			newlist.append(data)
	#	MOVE NEW FILE TO FACTORY FOLDER
	for data in newlist:
		#	PREVENT OVERWRITE
		if path_factory+'/'+os.path.basename(data) in glob.glob(path_factory+'/*'):
			rmcom = 'rm -rf '+path_factory+'/'+os.path.basename(data)
			logger.info(rmcom) ; os.system(rmcom)
		cpcom = 'cp -r '+data+' '+path_factory
		logger.info(cpcom) ; os.system(cpcom)
	#	CHANGE TO WORKING PATH
	for i in range(len(newlist)):
		newlist[i] = path_factory+'/'+os.path.basename(newlist[i])
	return newlist
#------------------------------------------------------------
def changehdr(inim, where, what):
	from astropy.io import fits
	data, hdr = fits.getdata(inim, header = True)
	hdr[where] = what
	fits.writeto(inim, data, hdr, clobber=True)
#------------------------------------------------------------
def correcthdr_routine(path_data, hdrtbl, obs):
	'''
	1.	NGC337		-> NGC0337
		ngc0337		-> NGC0337
	'''
	from astropy.io import fits
	from astropy.time import Time
	comment = '-'*60+'\n' \
			+ 'CHANGE TO UNIVERSAL HEADER SERIES ...\n' \
			+ '-'*60+'\n'
	logger.info(comment)
	objfilterlist = []
	objexptimelist = []
	flatfilterlist = []
	darkexptimelist = []
	imlist = []
	for ims in ('{}/*.fits'.format(path_data), '{}/*.fit'.format(path_data), '{}/*.fts'.format(path_data)):
		imlist.extend(sorted(glob.glob(ims)))
	# for inim in glob.glob(path_data+'/*.fit*'):
	for inim in imlist:
		logger.info(inim)
		if os.path.getsize(inim) < 2880:
			logger.warning(f'WARNING: skipping empty/corrupt file: {inim}')
			continue
		# tool_tbd.puthdr(inim, hdrkey='OBS', hdrval=obs, hdrcomment='observation location')
		fits.setval(inim, 'OBS', value=obs, comment='observation location', ignore_missing_end=True)
		data, hdr   = fits.getdata(inim, header=True)
		#	CHECK IMAGETYP HDR
		if hdr['IMAGETYP'] == 'Light': hdr['IMAGETYP'] = 'object'
		if hdr['IMAGETYP'] == 'zero' : hdr['IMAGETYP'] = 'Bias'
		if hdr['IMAGETYP'] == 'Dark' : hdr['IMAGETYP'] = 'dark'
		if hdr['IMAGETYP'] == 'Flat Field' : hdr['IMAGETYP'] = 'FLAT'
		#	CHECK FILTER HDR
		if hdr['IMAGETYP'] in ['object', 'Bias', 'dark',]:
			if 'FILTER' in hdr.keys():
				if ((hdr['FILTER'] == 'U101') | (hdr['FILTER'] == 1)): hdr['FILTER'] = 'U'
				if ((hdr['FILTER'] == 'B102') | (hdr['FILTER'] == 2)): hdr['FILTER'] = 'B'
				if ((hdr['FILTER'] == 'V103') | (hdr['FILTER'] == 3)): hdr['FILTER'] = 'V'
				if ((hdr['FILTER'] == 'R104') | (hdr['FILTER'] == 4)): hdr['FILTER'] = 'R'
				if ((hdr['FILTER'] == 'I105') | (hdr['FILTER'] == 5)): hdr['FILTER'] = 'I'
				if (hdr['FILTER'] == '0') | (hdr['FILTER'] == 0):
					hdr['FILTER'] = 'R'
					logger.warning('PLEASE CHECK SOAO FILTER HDR (FILTER:0?)')
		#	CHECK DATE-OBS
		if 'DATE-OBS' not in hdr:
			if 'UTDATE' in hdr and 'UTSTART' in hdr:
				date = hdr['UTDATE']
				time = hdr['UTSTART']

			# UTDATE can be YYYYMMDD or YYYY-MM-DD
			if isinstance(date, str) and len(date) == 8 and date.isdigit():
				date = f"{date[0:4]}-{date[4:6]}-{date[6:8]}"

			# UTARTS cand be HHMMSS or HH:MM:SS
			if isinstance(time, str) and len(time) ==6 and time.isdigit():
				time = f"{time[0:2]}:{time[2:4]}:{time[4:6]}"

			hdr['DATE-OBS'] = f"{date}T{time}"

		if 'T' not in hdr['DATE-OBS']: hdr['DATE-OBS'] = hdr['DATE-OBS']+'T'+hdr['TIME-OBS']
		t = Time(hdr['DATE-OBS'], format='isot')
		hdr['JD'] = t.jd
		hdr['MJD'] = t.mjd
		#	CHECK OBJECT HDR
		if 'OBJECT' in hdr.keys() and hdr['OBJECT'] is not None:
			if 'ngc' in hdr['OBJECT']:
				while len(hdr['OBJECT'])<7:
					head = hdr['OBJECT'][0:3]
					tail = hdr['OBJECT'][3: ]
					tail = '0'+tail
					hdr['OBJECT'] = head+tail
			_obj = fits.getheader(inim)['object']
			#	GW event
			if (_obj[:2] == 'S2') | (_obj[:3] == 'MS2'):
				obj = _obj
				pass
			#	Other target
			else:
				obj = _obj.upper()
			hdr['OBJECT'] = obj
		else:
			# No OBJECT Header
			pass
		#	CHECK USER SETTING HDR
		for i in range(len(hdrtbl)):
			key = hdrtbl['key'][i]
			val = hdrtbl['val'][i]
			newval = hdrtbl['newval'][i]
			if key in hdr.keys():
				if hdr[key] == val:
					# logger.info(hdr[key], key, newval)
					hdr[key] = newval
		fits.writeto(inim, data, hdr, overwrite=True)
		if hdr['IMAGETYP'] == 'object':
			objfilterlist.append(hdr['FILTER'])
			objexptimelist.append(hdr['EXPTIME'])
		if hdr['IMAGETYP'].upper() == 'FLAT':
			flatfilterlist.append(hdr['FILTER'])
		if hdr['IMAGETYP'].upper() == 'DARK':
			darkexptimelist.append(hdr['EXPTIME'])
	#	OBJECT FILTER, FLAT FILTER
	objfilterlist = list(set(objfilterlist))#;		objfilterlist.sort()
	objexptimelist = list(set(objexptimelist))#;		objexptimelist.sort()
	flatfilterlist = list(set(flatfilterlist))#;		flatfilterlist.sort()
	darkexptimelist = list(set(darkexptimelist))#;	darkexptimelist.sort()
	return objfilterlist, objexptimelist, flatfilterlist, darkexptimelist, t
#------------------------------------------------------------
def isot_to_mjd(time):	  # 20181026 to 2018-10-26T00:00:00:000 to MJD form
	from astropy.time import Time
	yr  = time[0:4]		 # year
	mo  = time[4:6]		 # month
	da  = time[6:8]		 # day
	isot = yr+'-'+mo+'-'+da+'T00:00:00.000'		 #	ignore hour:min:sec
	t = Time(isot, format='isot', scale='utc')		#	transform to MJD
	return t.mjd
#------------------------------------------------------------
def call_images(path_data):
	import ccdproc
	from astropy.io import ascii
	images = ccdproc.ImageFileCollection(path_data, keywords='*')
	#	DATA SUMMARY
	ascii.write(images.summary, path_data+'/summary.txt', format='fixed_width_two_line', overwrite=True)
	return images
#------------------------------------------------------------
def getobsinfo(obs, obstbl):
	import numpy as np
	from astropy import units as u
	indx_obs = np.where(obs == obstbl['obs'])
	#	CCD INFORMATION
	gain = obstbl['gain'][indx_obs].item() * u.electron / u.adu
	rdnoise = obstbl['RDnoise'][indx_obs].item() * u.electron
	pixscale = obstbl['pixelscale'][indx_obs].item()
	fov = obstbl['fov'][indx_obs].item()
	# dark		= float(obstbl['dark'][indx_obs][0])
	obsinfo = dict(
					gain=gain,
					rdnoise=rdnoise,
					pixscale=pixscale,
					fov=fov,
					)
	return obsinfo
#------------------------------------------------------------
#	CCD PROCESSING
#------------------------------------------------------------
def master_zero(images, fig=False):
	import os
	from astropy.nddata import CCDData
	import matplotlib.pyplot as plt
	comment	 = '-'*60+'\n' \
				+ 'MAKING MASTER ZERO\n' \
				+ '-'*60+'\n'
	logger.info(comment)
	path_data = images.location
	data_list = []
	newmeta = None
	n = 0
	for hdu, fname in images.hdus(imagetyp='Bias', return_fname=True):
		if newmeta is None:
			newmeta = hdu.header.copy()
			newmeta['FILENAME'] = 'zero.fits'
		n += 1
		newmeta[f'COMB{n}'] = fname
		data_list.append(hdu.data.astype(np.float32))
	logger.info('{} ZERO IMAGES WILL BE COMBINED.'.format(len(data_list)))
	combined = bn_median(np.ma.array(data_list), axis=0)
	mzero = CCDData(data=combined.data.astype(np.float32), meta=newmeta, unit='adu')
	if '{}/zero.fits'.format(path_data) in glob.glob(path_data+'/*'):
		os.system('rm {}/zero.fits'.format(path_data))
	mzero.write('{}/zero.fits'.format(path_data))
	if fig != False:
		imstats = lambda dat: (dat.min(), dat.max(), dat.mean(), dat.std())
		zero_min, zero_max, zero_mean, zero_std = imstats(np.asarray(mzero))
		plt.figure(figsize=(15, 15))
		plt.imshow(mzero, vmax=zero_mean + 4*zero_std, vmin=zero_mean - 4*zero_std)
		plt.savefig(path_data+'/zero.png')
	return mzero
#------------------------------------------------------------
def master_zero_bad(images, fig=False):
	comment	 = '-'*60+'\n' \
				+ 'MAKING MASTER ZERO\n' \
				+ '-'*60+'\n'
	logger.info(comment)
	path_data = images.location
	for hdu, fname in images.hdus(imagetyp='bias', return_fname=True):
		meta = hdu.header
		break
	#	New Meta data based on the first image
	newmeta = meta.copy()
	imlist = images.filter(imagetyp='Bias Frame',).files
	for n, inim in enumerate(imlist):
		n += 1
		newmeta[f'COMB{n}'] = os.path.basename(inim)
	newmeta['FILENAME'] = 'zero.fits'
	logger.info(f'{len(imlist)} ZERO IMAGES WILL BE COMBINED.')

	combined_data = bn_median(np.ma.array([fits.getdata(inim) for inim in imlist]), axis=0)
	mzero = CCDData(data=combined_data.data, meta=newmeta, unit='adu')

	#	SAVE MASTER FRAME
	if os.path.exists(f"{path_data}/zero.fits"):
		os.system(f'rm {path_data}/zero.fits')
	mzero.write(f'{path_data}/zero.fits')
	return mzero
#------------------------------------------------------------
def master_dark(images, mzero, exptime, fig=False):
	comment	 = '-'*60+'\n' \
				+ 'MAKING MASTER DARK\n' \
				+ '-'*60+'\n'
	logger.info(comment)
	path_data = images.location
	mdark_name = 'dark-{}.fits'.format(int(exptime))
	zero_data = mzero.data.astype(np.float32)
	data_list = []
	newmeta = None
	n = 0
	for hdu, fname in images.hdus(imagetyp='dark', exptime=exptime, return_fname=True):
		if newmeta is None:
			newmeta = hdu.header.copy()
			newmeta['FILENAME'] = mdark_name
			newmeta['SUBBIAS'] = mzero.meta['FILENAME']
		n += 1
		newmeta[f'COMB{n}'] = fname
		data_list.append(hdu.data.astype(np.float32) - zero_data)
	combined = bn_median(np.ma.array(data_list), axis=0)
	mdark = CCDData(data=combined.data.astype(np.float32), meta=newmeta, unit='adu')
	if '{}/{}'.format(path_data, mdark_name) in glob.glob(path_data+'/*'):
		os.system('rm {}/{}'.format(path_data, mdark_name))
	mdark.write('{}/{}'.format(path_data, mdark_name))
	if fig != False:
		imstats = lambda dat: (dat.min(), dat.max(), dat.mean(), dat.std())
		zero_min, zero_max, zero_mean, zero_std = imstats(np.asarray(mzero))
		plt.figure(figsize=(15, 15))
		plt.imshow(mzero, vmax=zero_mean + 4*zero_std, vmin=zero_mean - 4*zero_std)
		plt.savefig(path_data+'/zero.png')
	return mdark
#------------------------------------------------------------
def master_flat(images, mzero, filte, mdark=False, fig=False):
	import os
	from astropy.nddata import CCDData
	import matplotlib.pyplot as plt
	comment	 = '-'*60+'\n' \
				+ 'MAKING MASTER FLAT\n' \
				+ '-'*60+'\n'
	logger.info(comment)
	path_data = images.location
	outname = 'n' + filte + '.fits'
	zero_data = mzero.data.astype(np.float32)
	dark_data = mdark.data.astype(np.float32) if mdark is not False else None
	if dark_data is not None:
		dark_exptime = float(mdark.meta.get('EXPTIME', 1.0))

	data_list = []
	newmeta = None
	n = 0
	for hdu, fname in images.hdus(imagetyp='flat', filter=filte, return_fname=True):
		if newmeta is None:
			newmeta = hdu.header.copy()
			newmeta['FILENAME'] = outname
			newmeta['SUBBIAS'] = mzero.meta['FILENAME']
		n += 1
		newmeta[f'COMB{n}'] = fname
		frame = hdu.data.astype(np.float32) - zero_data
		if dark_data is not None:
			exptime = float(hdu.header.get('EXPTIME', dark_exptime))
			frame -= dark_data * (exptime / dark_exptime)
			newmeta['SUBDARK'] = mdark.meta['FILENAME']
		data_list.append(frame)

	logger.info('{} {} FLAT IMAGES WILL BE COMBINED.'.format(len(data_list), filte))
	# Reproduce old ccdproc.Combiner order exactly:
	#   1) minmax_clipping on unscaled values  →  2) scale using clipped median  →  3) median combine
	stack = np.array(data_list, dtype=np.float32)  # (N, H, W)
	clipped = stack.copy()
	if stack.shape[0] > 2:
		rows = np.arange(stack.shape[1], dtype=np.intp)[:, None]
		cols = np.arange(stack.shape[2], dtype=np.intp)[None, :]
		clipped[np.argmin(stack, axis=0), rows, cols] = np.nan  # per-pixel min → NaN
		clipped[np.argmax(stack, axis=0), rows, cols] = np.nan  # per-pixel max → NaN
	# Scale using median of each frame's clipped pixels (matches old scaling_func)
	medians = np.nanmedian(clipped, axis=(1, 2))[:, None, None]
	clipped /= medians
	combined = bn_median(np.ma.array(clipped, mask=np.isnan(clipped)), axis=0)

	mflat = CCDData(data=combined.data.astype(np.float32), meta=newmeta, unit='adu')
	if '{}/{}'.format(path_data, outname) in glob.glob(path_data+'/*'):
		os.system('rm {}/{}'.format(path_data, outname))
	mflat.write(path_data + '/' + outname)
	if fig != False:
		imstats = lambda dat: (dat.min(), dat.max(), dat.mean(), dat.std())
		f_min, f_max, f_mean, f_std = imstats(np.asarray(mflat))
		plt.figure(figsize=(15, 15))
		plt.imshow(mflat, vmin=f_mean - 5*f_std, vmax=f_mean + 5*f_std)
		plt.savefig(path_data + '/' + outname[:-5] + '.png')

	return mflat
#------------------------------------------------------------
def calibration(images, mzero, mflat, filte, mdark=False, exptime=None):
	import ccdproc
	comment	 = '-'*60+'\n' \
				+ 'OBJECT CORRECTION\n' \
				+ '-'*60+'\n'
	logger.info(comment)
	path_data = images.location
	objlist = []
	filter_kwargs = dict(imagetyp='object', filter=filte, return_fname=True)
	if exptime is not None:
		filter_kwargs['exptime'] = exptime
	for hdu, fname in images.hdus(**filter_kwargs):
		meta = hdu.header
		objlist.append(meta['object'])
		obj = ccdproc.CCDData(data=hdu.data, meta=meta, unit="adu")
		if mdark == False:
			#	ZERO CORRECTION
			zobj = ccdproc.subtract_bias(obj, mzero)
			meta['SUBBIAS'] = mzero.meta['FILENAME']
		else:
			#	ZERO & DARK CORRECTION
			zobj = ccdproc.subtract_bias(obj, mzero)
			meta['SUBBIAS'] = mzero.meta['FILENAME']
			zobj = ccdproc.subtract_dark(	ccd=zobj, master=mdark,
											# dark_exposure=mdark.meta['EXPTIME'],
											# data_exposure=dzobj.meta['EXPTIME'],
											exposure_time='EXPTIME',
											exposure_unit=u.second,
											scale=True)
			meta['SUBDARK'] = mdark.meta['FILENAME']
		#	FLAT CORRECTION
		fzobj = ccdproc.flat_correct(zobj, mflat)
		meta['DIVFLAT'] = mflat.meta['FILENAME']
		fzobj.header = meta
		# Downcast to float32 before writing: ccdproc's subtract_bias/subtract_dark/
		# flat_correct promote the frame to float64 (8 B/px), doubling every calibrated
		# frame on disk (and its MissFITS .back) with no benefit — CCD ADUs fit float32
		# easily, and the SWarp coadd is float32 anyway (so a float64 input frame looked
		# LARGER than its own stack). Keeps stack inputs consistent with the coadd.
		fzobj.data = fzobj.data.astype(np.float32)
		# fzobj.write(path_data+'/fz'+fname)
		fzobj.write('{}/fz{}'.format(path_data, fname), overwrite=True)
#------------------------------------------------------------
def _run_solve_field(com):
	import subprocess
	result = subprocess.run(com, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
	if result.stdout:
		logger.info(result.stdout)
#------------------------------------------------------------
def astrometry(inim, pixscale, ra=None, dec=None, fov=1, cpulimit=180):
	'''
	ra : hh:mm:ss
	dec : dd:mm:ss
	fov [deg]
	'''
	import os

	# This runs inside an ncores-wide (default 4) worker pool: 4 threads per
	# worker oversubscribed the shared server 4x (up to 16 threads per run).
	# One thread per worker keeps the total at the promised ncores.
	os.environ["OMP_NUM_THREADS"] = "1"
	os.environ["OPENBLAS_NUM_THREADS"] = "1"
	os.environ["MKL_NUM_THREADS"] = "1"
	os.environ["NUMEXPR_NUM_THREADS"] = "1"

	upscl = str(pixscale + pixscale*0.10)
	loscl = str(pixscale - pixscale*0.10)
	outname = os.path.dirname(inim)+'/a'+os.path.basename(inim)
	import shutil as _shutil
	_solve_field = _shutil.which('solve-field') or '/usr/bin/solve-field'
	com	 =_solve_field+' '+inim \
			+' --scale-unit arcsecperpix --scale-low '+loscl+' --scale-high '+upscl \
			+' --no-plots --new-fits '+outname+' --overwrite --use-source-extractor'\
			+' --cpulimit  {}'.format(cpulimit) +' --config {}'.format(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'astrometry_gppy.config'))\
			#+' -D {}'.format(os.path.dirname(inim))
	#' --index-dir /data7/sophia2/util/astrometry/data/index-5200'
	if ((ra != None) & (dec != None) & (fov != None)):
		com	= com + ' --ra {} --dec {} --radius {}'.format(ra, dec, fov)
	else:
		pass
	logger.info(com)
	_run_solve_field(com)


def astrometry_RASA36(inim, pixscale, ra=None, dec=None, fov=1, cpulimit=180, config=None):
	'''
	ra : hh:mm:ss
	dec : dd:mm:ss
	fov [deg]
	config : path to astrometry.net config file (default: config/astrometry_gppy_rasa36.config under toYP_pipeline/)
	'''
	import os

	# Runs inside an ncores-wide (default 4) worker pool: 4 threads per worker
	# oversubscribed the shared server 4x. One thread per worker keeps the
	# total at the promised ncores.
	os.environ["OMP_NUM_THREADS"] = "1"
	os.environ["OPENBLAS_NUM_THREADS"] = "1"
	os.environ["MKL_NUM_THREADS"] = "1"
	os.environ["NUMEXPR_NUM_THREADS"] = "1"

	if config is None:
		config = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'astrometry_gppy_rasa36.config')

	upscl = str(pixscale + pixscale*0.10)
	loscl = str(pixscale - pixscale*0.10)
	outname = os.path.dirname(inim)+'/a'+os.path.basename(inim)
	import shutil as _shutil
	_solve_field = _shutil.which('solve-field') or '/usr/bin/solve-field'
	com	 =_solve_field+' '+inim \
			+' --scale-unit arcsecperpix --scale-low '+loscl+' --scale-high '+upscl \
			+' --no-plots --new-fits '+outname+' --overwrite --use-source-extractor'\
			+' --cpulimit  {}'.format(cpulimit) +' --config {}'.format(config)
	if ((ra != None) & (dec != None) & (fov != None)):
		com	= com + ' --ra {} --dec {} --radius {}'.format(ra, dec, fov)
	logger.info(com)
	_run_solve_field(com)
#------------------------------------------------------------
def fnamechange(inim, obs, com='mv'):
	import os
	from astropy.io import fits
	hdr = fits.getheader(inim)
	dateobs = hdr['DATE-OBS']
	datestr = dateobs[0:4]+dateobs[5:7]+dateobs[8:10]
	timestr = dateobs[11:13]+dateobs[14:16]+dateobs[17:19]
	_objname = hdr['OBJECT']
	#	GW event
	if (_objname[:2] == 'S2') | (_objname[:3] == 'MS2'):
		objname = _objname
		pass
	else:
		objname = _objname.upper()
	filname = hdr['FILTER']  # str(hdr['FILTER'])
	#exptime = str(int(hdr['EXPTIME']))
	exptime = str(int(float(hdr['EXPTIME'])))
	newname = 'Calib-{}-{}-{}-{}-{}-{}.fits'.format(obs, objname, datestr, timestr, filname, exptime)
	if com == 'cp':
		cpcom = 'cp {} {}/{}'.format(inim, os.path.dirname(inim), newname)
		logger.info(cpcom)
		os.system(cpcom)
	if com == 'mv':
		mvcom = 'mv {} {}/{}'.format(inim, os.path.dirname(inim), newname)
		logger.info(mvcom)
		os.system(mvcom)
#------------------------------------------------------------
# def movecalib(inim, path_gal):
# 	import os, glob
# 	img = os.path.basename(inim)
# 	part = img.split('-')
# 	obs = part[1]
# 	obj = part[2]
# 	filte = part[5]
# 	#	MAKE SAVE PATH
# 	path_goal = path_gal
# 	for folder in [obj, obs, filte]:
# 		path_goal = path_goal+'/'+folder
# 		mkcom = 'mkdir {}'.format(path_goal)
# 		logger.info(mkcom); os.system(mkcom)
# 	mvcom = 'mv {} {}'.format(inim, path_goal)
# 	logger.info(mvcom); os.system(mvcom)
#------------------------------------------------------------
def movecalib(inim, path_gal, additional=False):
	import os
	'''
	path_gal = '/data3/IMSNG/IMSNGgalaxies'
	inim = '/data3/paek/factory/doao/20201209-1m-IMSNG/Calib-DOAO-NGC0000-20201209-113240-B-60.fits'
	inim = '/data3/paek/factory/doao/20201209-1m-IMSNG/Calib-DOAO-ESO000-123-20201209-113240-B-60.fits'
	'''
	part = os.path.basename(inim).split('-')
	#	(obj name) = 'NGC0000'
	if (len(part)==7) & (os.path.basename(inim).count('-')==6):
		obs = part[1]
		obj = part[2]
		filte = part[5]
	#	(obj name) = 'ESO000-123' containing additional '-'
	elif (len(part)==8) & (os.path.basename(inim).count('-')==7):
		obs = part[1]
		obj = part[2]
		filte = obj = '-'.join([part[2], part[3]])
	else:
		pass

	#	Path
	path_obj = '{}/{}'.format(path_gal, obj)
	path_obs = '{}/{}/{}'.format(path_gal, obj, obs)
	path_filte = '{}/{}/{}/{}'.format(path_gal, obj, obs, filte)
	if additional != False:
		path_filte = '{}/{}'.format(path_filte, additional)

	#	Command
	mkcom0 = 'mkdir {}'.format(path_obj)
	mkcom1 = 'mkdir {}'.format(path_obs)
	mkcom2 = 'mkdir {}'.format(path_filte)
	mvcom = 'mv {} {}'.format(inim, path_filte)

	#	Create folder
	if os.path.exists(path_obj):
		if os.path.exists(path_obs):
			if os.path.exists(path_filte):
				pass
			else:
				logger.info(mkcom2)
				os.system(mkcom2)
		else:
			logger.info(mkcom1)
			os.system(mkcom1)
	else:
		logger.info(mkcom0)
		os.system(mkcom0)
		logger.info(mkcom1)
		os.system(mkcom1)
		logger.info(mkcom2)
		os.system(mkcom2)
	#	Move image
	# logger.info(mvcom)
	os.system(mvcom)
#------------------------------------------------------------
def defringe(inim, dfim, dfdat):
	'''
	inim : image to remove fringe
	dfim : master fringe image
	dfdat : master fringe data
	'''
	data, hdr = fits.getdata(inim, header=True)
	#	Master fringe
	dataf, hdrf	= fits.getdata(dfim, header=True)

	master_fri = fringe_cal(dfim, dfdat)
	infr_list = []
	fscale = np.median(fringe_cal(inim, dfdat)/master_fri)
	# logger.info(fscale)

	fri_scaled = dataf*fscale
	dfdata = data-fri_scaled
	outim = '{}/df{}'.format(os.path.dirname(inim), os.path.basename(inim))
	# fscale is a float64 scalar, so data-fri_scaled promotes back to float64; keep
	# defringed frames float32 to match the calibrated inputs and the coadd.
	fits.writeto(outim, dfdata.astype(np.float32), hdr, overwrite=True)
	# logger.info(np.median(data), np.median(fri_scaled), np.median(dfdata))

	return outim
#------------------------------------------------------------
def fringe_cal(inim, dfdat):
	data, hdr	= fits.getdata(inim, header=True)
	#subframe	= data[512:1537, 512:1537]
	#skymean, skymed, skysig		= bkgest_mask(inim)
	#subdata		= data-skymed

	dfr_list	= []
	dftbl = ascii.read(dfdat)

	xb1, xb2 = dftbl['xb']-5, dftbl['xb']+5
	yb1, yb2 = dftbl['yb']-5, dftbl['yb']+5

	xf1, xf2 = dftbl['xf']-5, dftbl['xf']+5
	yf1, yf2 = dftbl['yf']-5, dftbl['yf']+5

	for n in range(len(dftbl)):
		'''
		fringe_b= np.median(subdata[yb1[n]:yb2[n], xb1[n]:xb2[n]])
		fringe_f= np.median(subdata[yf1[n]:yf2[n], xf1[n]:xf2[n]])
		'''
		fringe_b= np.median(data[yb1[n]:yb2[n], xb1[n]:xb2[n]])
		fringe_f= np.median(data[yf1[n]:yf2[n], xf1[n]:xf2[n]])
		dfringe	= fringe_b-fringe_f
		dfr_list.append(dfringe)
	return np.array(dfr_list)
#------------------------------------------------------------
def cr_removal(inim, outim, gain, rdnoise):
	'''
	inim
	obs = 'LOAO'
	gain = 2.68
	rdnoise = 4.84
	'''
	import os
	from astroscrappy import detect_cosmics
	from astropy.io import fits
	import time

	os.environ["OMP_NUM_THREADS"] = "4"
	os.environ["OPENBLAS_NUM_THREADS"] = "4"
	os.environ["MKL_NUM_THREADS"] = "4"
	os.environ["NUMEXPR_NUM_THREADS"] = "2"

	data, hdr = fits.getdata(inim, header=True)

	param_cr = dict(
					indat=data,
					sigclip=4.5,
					sigfrac=0.3,
					objlim=5.0,
					gain=gain, readnoise=rdnoise,
					# pssl=0.0,
					niter=4,
					sepmed=True,
					cleantype='meanmask',
					fsmode='median',
					psfmodel='gauss',
					psffwhm=hdr['seeing'],
					#	Notouch
					inmask=None,
					satlevel=65536.0,
					psfsize=7, psfk=None, psfbeta=4.765,
					verbose=False
					)

	time_st = time.time()
	mcrdata, crdata = detect_cosmics(**param_cr)
	fits.writeto(outim, crdata, hdr, overwrite=True)
	# tool_tbd.puthdr(outim,  )
	time_delta = time.time() - time_st
	# fits.writeto('{}/mcr{}'.format(os.path.dirname(inim), os.path.basename(inim)), mcrdata, hdr, overwrite=True)
	#------------------------------------------------------------
	# outbl = Table()
	indx_cr = np.where(mcrdata == True)
	logger.info('Remove {} cosmic-ray for {} [{} sec]'.format(len(indx_cr[0]), inim, round(time_delta, 3)))
	fits.setval(outim, 'CR', value=len(indx_cr[0]), comment='# of removed cosmic ray by Astroscrappy', ignore_missing_end=True)
	# return outbl

#-----------------------------------------------------------------
def bgscale(im, outdir) :

	logger.info('Quad pattern Processing of '+str(im)+' begins...')
	head = fits.getheader(im)
	data = fits.getdata(im)
	## array slice arr[x:y]  = ( x <= index < y )
	## ex) range(10)[3:7] = [3, 4, 5, 6]
	## range(10) = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
	if head['NAXIS'] == 2 :
		logger.info('NAXIS : '+str(head['NAXIS']))
		aq = data[2048:4096,0:2048]
		bq = data[2048:4096,2048:4096]
		cq = data[0:2048,2048:4096]
		dq = data[0:2048,0:2048]
		logger.info('A '+str(np.mean(aq)))
		logger.info('B '+str(np.mean(bq)))
		logger.info('C '+str(np.mean(cq)))
		logger.info('D '+str(np.mean(dq)))

		av, ah=data[2048:4050,2047], data[2048,5:2045]
		bv, bh=data[2048:4050,2048], data[2048,2050:4090]
		cv, ch=data[40:2040,2048], data[2047,2050:4090]
		dv, dh=data[40:2040,2047], data[2047,5:2045]

	else :
		aq = data[0][2048:4096,0:2048]
		bq = data[0][2048:4096,2048:4096]
		cq = data[0][0:2048,2048:4096]
		dq = data[0][0:2048,0:2048]
		logger.info('A '+str(np.mean(aq)))
		logger.info('B '+str(np.mean(bq)))
		logger.info('C '+str(np.mean(cq)))
		logger.info('D '+str(np.mean(dq)))

		av, ah=data[0][2048:4050,2047], data[0][2048,5:2045]
		bv, bh=data[0][2048:4050,2048], data[0][2048,2050:4090]
		cv, ch=data[0][40:2040,2048], data[0][2047,2050:4090]
		dv, dh=data[0][40:2040,2047], data[0][2047,5:2045]

	logger.info('offset : A - B = ' + str(robust.mean(av-bv)))
	logger.info('offset : C - D = ' + str(robust.mean(cv-dv)))
	logger.info('A-B, C-D difference is calculated and apply that value to both pair...')

	aq=aq - (robust.mean(av)[0]-robust.mean(bv)[0])
	dq=dq + (robust.mean(cv)[0]-robust.mean(dv)[0])

	if head['NAXIS'] == 2 :
		downim   = data[0:2048,0:4096]  ## D + C quad
		upim = data[2048:4096,0:4096]   ## A + B quad
	else :
		downim   = data[0][0:2048,0:4096]  ## D + C quad
		upim = data[0][2048:4096,0:4096]   ## A + B quad

	upim[0:2048,0:2048] = aq
	upim[0:2048,2048:4096] = bq
	downim[0:2048,2048:4096] = cq
	downim[0:2048,0:2048] = dq

	uplow   = upim[0,10:4085]
	downtop = downim[2047,10:4085]

	logger.info('Upper Half bottom line mean value = ' + str(robust.mean(uplow)))
	logger.info('Lower Half top line mean value    = ' + str(robust.mean(downtop)))
	halfdiff= robust.mean(uplow,Cut=5.0)[0] - robust.mean(downtop,Cut=5.0)[0]
	logger.info('difference between upper & lower region ' + str(halfdiff))

	## Lower part will be reference (a little low val ~ 1-2 ADU)
	upim = upim - (robust.mean(uplow)[0] - robust.mean(downtop)[0])

	if head['NAXIS'] == 2 :
		data[2048:4096,0:4096] = upim
		data[0:2048,0:4096]	= downim
	else :
		data[0][2048:4096,0:4096] = upim
		data[0][0:2048,0:4096]	= downim

	nowdt = datetime.datetime.utcnow()
	nowdtstr = nowdt.isoformat()
	head['COMMENT'] = 'Background Scaled for Quad pattern Removal'
	head['COMMENT'] = str(nowdtstr)+' UTC'
	outname = 'bg' + os.path.basename(im)
	outpath = os.path.join(outdir, outname)

	fits.writeto(outpath, data, head, overwrite=True)
	logger.info('Back Ground Scaling for quadrant pattern is finished for '+str(im)+'\n')
