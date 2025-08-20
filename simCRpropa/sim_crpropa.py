from crpropa import *
import crpropa
import logging
import yaml
import numpy as np
import argparse
from os import path, environ
from fermiAnalysis.batchfarm import utils, lsf, sdf
from copy import deepcopy
from glob import glob
from astropy.table import Table
from astropy.io import fits
from astropy import units as u
from astropy import constants as c
import simCRpropa
import socket
from simCRpropa import collect
from collections import OrderedDict
import h5py
import os
from numpy.random import RandomState


@lsf.setLsf
def _submit_run_lsf(script, config, option, njobs, **kwargs):
    """Submit jobs to LSF (old) cluster using bsub"""
    kwargs.setdefault('span', f"span[ptile={kwargs['n']}]")
    option += " -b lsf"
    lsf.submit_lsf(script,
                   config,
                   option,
                   njobs, 
                   **kwargs)

@sdf.set_sdf
def _submit_run_sdf(script, config, option, njobs, **kwargs):
    """Submit jobs to SDF cluster using slurm"""
    kwargs['ntasks_per_node'] = kwargs['n']
    if kwargs['n'] > 1 and kwargs['mem'] is None:
        kwargs['mem'] = int(4000 * kwargs['n'])

    option += " -b sdf"
    
    sdf.submit_sdf(script,
                   config,
                   option,
                   njobs, 
                   **kwargs)


def init_rectangular_prism_bfield(field_zero_near_origin, vgrid, obsSize,
                                  b_void=0*gauss, b_ext=1e-13*gauss, 
                                  seed=0):
    """Fill grid with magnetic field values in the shape of a rectangular prism.
    PUT ALL UNITS IN SI.

    Parameters
    ----------
    field_zero_near_src : bool
        If True, |B|=0 in the rectangular prism near the source (from approximately the source to halfway along the LoS), 
        and is nonzero elsewhere.
        If False, |B| != 0 in the rectangular prism near the source (from approximately the source to halfway along the LoS), 
        and |B|=0 elsewhere.
    b_void : float
        Magnetic field in voids, in Gauss.
    b_ext : float
        Magnetic field outside/exterior to voids, in Gauss.
    seed : int
        Seed for random direction of magnetic field at each grid point.
    gridSpacingMpc : float or int?
        Side length of each individual grid cell, in Mpc
    num_pad_cell : int
    even_grid : bool
    h : float
    Om : float

    Returns
    -------
    """
    msg = f"Using random seed={seed}"
    logging.info(msg)
    print(msg)
    np.random.seed(seed)
    gridArray = vgrid.getGrid()
    nx = vgrid.getNx()
    ny = vgrid.getNy()
    nz = vgrid.getNz()
    msg = f"vgrid: nx={nx}, ny={ny}, nz={nz}"
    logging.info(msg)
    print(msg)

    # Fill grid
    for xi in range(0, nx):
        for yi in range(0, ny):
            for zi in range(0, nz):
                vect3d = vgrid.get(xi, yi, zi)
                x = np.random.uniform(-1,1)
                y = np.random.uniform(-1,1)
                z = np.random.uniform(-1,1)
                d = np.sqrt(x*x+y*y+z*z)

                if field_zero_near_origin:
                    # Check if grid x coordinate is in the first half along the LoS from observer to source (x is along LoS: src->obs)
                    # Pad edge of grid with B=0 so that the center of the grid (the source) does not have B=/=0 from interpolating the edge
                    # Add B = 0 outside source to avoid edge effects in which the center of the grid has B =/=0 (interpolated from case when B=/=0 at the edge of the grid?)
                    if xi < int(obsSize/2) or xi > obsSize:
                        vect3d.x = b_void * x/d
                        vect3d.y = b_void * y/d
                        vect3d.z = b_void * z/d
                    else:
                        vect3d.x = b_ext * x/d
                        vect3d.y = b_ext * y/d
                        vect3d.z = b_ext * z/d

                else:
                    # B is nonzero near source
                    # It should not matter that B=/=0 beyond the observer (near the grid edge) in this case, because if interpolated
                    # to the center, the desired behavior is B=/=0 there anyway
                    if xi <= int(obsSize/2): # or xi > obsSize:
                        vect3d.x = b_ext * x/d
                        vect3d.y = b_ext * y/d
                        vect3d.z = b_ext * z/d
                    else:
                        vect3d.x = b_void * x/d
                        vect3d.y = b_void * y/d
                        vect3d.z = b_void * z/d
                
    return None
  

def initPixelizedSphere(seed):
    # TODO
    msg = f"Using random seed={seed}"
    logging.info(msg)
    print(msg)
    np.random.seed(seed)
    
    return None


def initRandomField(vgrid, Bamplitude, seed=0):
    msg = f"Using random seed={seed}"
    logging.info(msg)
    print(msg)
    prng = RandomState(seed)
    gridArray = vgrid.getGrid()
    nx = vgrid.getNx()
    ny = vgrid.getNy()
    nz = vgrid.getNz()
    logging.info(f"vgrid: nx = {nx}, ny = {ny}, nz = {nz}")
    for xi in range(0,nx):
        for yi in range(0,ny):
            for zi in range(0,nz):
                vect3d = vgrid.get(xi,yi,zi)
                x = prng.uniform(-1,1)
                y = prng.uniform(-1,1)
                z = prng.uniform(-1,1)
                d = np.sqrt(x*x+y*y+z*z)

                vect3d.x = Bamplitude * x/d
                vect3d.y = Bamplitude * y/d
                vect3d.z = Bamplitude * z/d
    # Print last random number/vector to see if it is thread-safe
    logging.info(f"Last x-component random unit vector in grid: ({x})")
    logging.info(f"Last random unit vector in grid: ({x, y ,z})")

    return None

def build_histogram(combined, config, cascparent = 11., intparent = 22., obs = 22.):
    """
    Build a numpy histogram for the cascade 
    and intrinsic spectrum

    Parameters
    ----------
    combined: `~h5py.File`
        combined hdf5 histogram
    config: dict
        dict with configuration

    {options}
    cascparent: float
        parent particle ID for cascade
    intparent: float
        parent particle ID for intrinsic spectrum
    obs: float
        particle ID for observed spectrum

    Returns
    -------
    tuple with instrinsic spectrum, cascade spectrum, and 
    energy bins
    """
    Ebins = np.logspace(np.log10(config['Source']['Emin']),
                 np.log10(config['Source']['Emax']),
                                 config['Source']['Esteps'])
    Ecen = np.sqrt(Ebins[1:] * Ebins[:-1])

    # intrinsic spectrum
    intspec = np.zeros((Ecen.size,Ecen.size))
    # casc spectrum
    casc = np.zeros((Ecen.size,Ecen.size))

    for i in range(Ebins.size - 1):
        m = (combined[f'simEM/ID1/Ebin{i:03n}'][...] == np.abs(intparent)) \
                & (combined[f'simEM/ID/Ebin{i:03n}'][...] == np.abs(obs))
        h = np.histogram(combined[f'simEM/E/Ebin{i:03n}'][m], bins = Ebins)
        intspec[i,:] = h[0]
                                
        m = (combined[f'simEM/ID1/Ebin{i:03n}'][...] == np.abs(cascparent)) \
                & (combined[f'simEM/ID/Ebin{i:03n}'][...] == np.abs(obs))
        h = np.histogram(combined[f'simEM/E/Ebin{i:03n}'][m], bins = Ebins)
        casc[i,:] = h[0]
    return intspec, casc, Ebins

def build_histogram_obs(combined, config, obs = 22., Ebins = np.array([])):
    """
    Build a numpy histogram for the cascade spectrum 
    of some particle type. 

    Parameters
    ----------
    combined: `~h5py.File`
        combined hdf5 histogram
    config: dict
        dict with configuration

    {options}
    obs: float 
        particle ID for observed spectrum
    Ebins: `~numpy.ndarray`
        custom energy binning.
        If zero length, use binning from config file

    Returns
    -------
    `~numpy.ndarray` with cascade spectrum
    """
    if Ebins.size == 0:
        Ebins = np.logspace(np.log10(config['Source']['Emin']),
                 np.log10(config['Source']['Emax']),
                                 config['Source']['Esteps'])
    Ecen = np.sqrt(Ebins[1:] * Ebins[:-1])

    # casc spectrum
    casc = np.zeros((config['Source']['Esteps'] - 1,Ecen.size))

    for i,k in enumerate(combined['simEM/E'].keys()):
                                
        m = combined[f'simEM/ID/{k:s}'][...] == np.abs(obs)
        h = np.histogram(combined[f'simEM/E/{k:s}'][m], bins = Ebins)
        casc[i,:] = h[0]
    return casc, Ebins

defaults = """
FileIO:
    outdir: ./
"""

class SimCRPropa(object):
    def __init__(self, **kwargs):
        """
        Initialize the class
        """
        df = yaml.safe_load(defaults)
        for k,v in df.items():
            kwargs.setdefault(k,v)
            for kk,vv in v.items():
                kwargs[k].setdefault(kk,vv)
        self.config = deepcopy(kwargs)
        self.__dict__.update(self.config)

        self.emcasc = self.Simulation['emcasc']

        btype = self.Bfield['type']
        if btype != 'txt':
            for i, k in enumerate(['B', 'gridSpacing']):
                if isinstance(self.Bfield[btype][k], list):
                    x = deepcopy(self.Bfield[btype][k])
                    self.Bfield[btype][k] = x[0]
                elif isinstance(self.Bfield[btype][k], float):
                    x = [self.Bfield[btype][k]]
                else:
                    raise ValueError(f"{self.Bfield[btype][k]} type not understood: {type(self.Bfield[btype][k])}")
                if not i:
                    self._bList = x
                else:
                    self._gridSpacingList = x
        else:
            for i, k in enumerate(['txtFile', 'fnDescriptor', 'gridSpacing']):
                if isinstance(self.Bfield[btype][k], list):
                    x = deepcopy(self.Bfield[btype][k])
                    self.Bfield[btype][k] = x[0]
                elif isinstance(self.Bfield[btype][k], float):
                    x = [self.Bfield[btype][k]]
                elif isinstance(self.Bfield[btype][k], str):
                    x = [self.Bfield[btype][k]]
                else:
                    raise ValueError(f"{self.Bfield[btype][k]} type not understood: {type(self.Bfield[btype][k])}")
                # i = 0
                if not i:
                    self._bFileList = x
                elif i == 1:
                    self._gridSpacingList = x
                else:
                    self._bDescriptorList = x
            # For the purpose of evaluatingthe rest of `init`
            self._bList = self._bFileList

        if np.isscalar(self.Simulation['multiplicity']):
            self._multiplicity = list(np.full(len(self._bList),
                                              self.Simulation['multiplicity']))
        else:
            self._multiplicity = self.Simulation['multiplicity']

        if not len(self._multiplicity) == len(self._bList):
            raise ValueError("Bfield and multiplicity lists must have same length!")

        for i, k in enumerate(['th_jet', 'z']):
            if isinstance(self.Source[k], list):
                x = deepcopy(self.Source[k])
                self.Source[k] = x[0]
            elif isinstance(self.Source[k], float):
                x = [self.Source[k]]
            else:
                raise ValueError(f"{self.Source[k]} type not understood: {type(self.Source[k])}")
            if not i:
                self._th_jetList= x
            else:
                self._zList = x

        if 'IRB_Gilmore12' in self.Bfield['EBL']:
            self._EBL = IRB_Gilmore12 #Dominguez11, Finke10, Franceschini08
        elif 'Dominguez11' in self.Bfield['EBL']:
            self._EBL = IRB_Dominguez11
        elif 'Finke10' in self.Bfield['EBL']:
            self._EBL = IRB_Finke10
        elif 'Franceschini08' in self.Bfield['EBL']:
            self._EBL = IRB_Franceschini08
        elif 'IRB_Saldana21' in self.Bfield['EBL']:
            self._EBL = IRB_Saldana21
        else:
            raise ValueError("Unknown EBL model chosen")
        self._URB = URB_Protheroe96

        if self.Source['useSpectrum']:
            self.nbins = 1
            self.weights = [self.Simulation['Nbatch']]
        # do a bin-by-bin analysis
        else:
            if not type(self.Source['Emin']) == type(self.Source['Emax']) \
                    == type(self.Simulation['Nbatch']):
                raise TypeError("Emin, Emax, and Nbatch must be the same type")

            if type(self.Source['Emin']) == float:
                self.EeVbins = np.logspace(np.log10(self.Source['Emin']),
                    np.log10(self.Source['Emax']), self.Source['Esteps'])
                self.weights = self.Simulation['Nbatch'] * \
                            np.ones(self.EeVbins.size - 1, dtype = int) # weight with optical depth?
                self.EeV = np.sqrt(self.EeVbins[1:] * self.EeVbins[:-1])

            elif type(self.Source['Emin']) == list or type(self.Source['Emin']) == tuple \
                or type(self.Source['Emin']) == np.ndarray:

                self.Source["Emin"] = list(self.Source["Emin"])
                self.Source["Emax"] = list(self.Source["Emax"])
                self.Simulation["Nbatch"] = list(self.Simulation["Nbatch"])

                if not len(self.Source["Emin"]) == len(self.Source["Emax"]) == \
                    len(self.Simulation["Nbatch"]):
                    raise TypeError("Emin, Emax, Nbatch arrays must be of same size")

                self.EeVbins = np.vstack([self.Source["Emin"], self.Source["Emax"]])
                self.weights = np.array(self.Simulation["Nbatch"])
                self.EeV = np.sqrt(np.prod(self.EeVbins, axis=0))

            self.weights = self.weights.astype(int)
            self.nbins = self.EeV.size
            self.Source['Energy'] = self.EeV[0]
            logging.info(f"There will be {self.nbins} energy bins")
            if not self.nbins:
                raise ValueError("No energy bins requested, change Emin, Emax, or Esteps")

        # set min step length for simulation 
        # depending on min requested time resolution
        # takes precedence over minStepLength
        if 'minTresol' in self.Simulation.keys():
            if np.isscalar(self.Simulation['minTresol']):
                self._minStepLength = list(np.full(len(self._bList),
                                                  self.Simulation['minTresol']))
            else:
                self._minStepLength = self.Simulation['minTresol']

            if not len(self._minStepLength) == len(self._bList):
                raise ValueError("Bfield and minStepLength lists must have same length!")

            dt = [u.Quantity(msl) for msl in self._minStepLength]
            dt = np.array([t.value for t in dt]) * dt[0].unit
            self._minStepLength = (dt * c.c.to(f"pc / {dt[0].unit}")).value
            self.Simulation['minStepLength'] = self._minStepLength[0]
            logging.info(f"Set step length(s) to {self._minStepLength} pc " \
                         f"from requsted time resolution(s) {dt}")
        else:
            self._minStepLength = self.Simulation['minStepLength']
            logging.info(f"Set step length(s) to {self._minStepLength} pc")

        # set up cosmology
        logging.info(f"Setting up cosmology with h={self.Cosmology['h']} and Omega_matter={self.Cosmology['Om']}")
        setCosmologyParameters(self.Cosmology['h'], self.Cosmology['Om'])
        return

    def setOutput(self,jobid, idB=0, idL=0, it=0, iz=0):
        """Set output file and directory"""
        if self.Simulation.get('outputtype', 'ascii') == 'ascii':
            self.PhotonOutName = f'casc_{jobid:05d}.dat'
        elif self.Simulation.get('outputtype', 'ascii') == 'hdf5':
            self.PhotonOutName = f'casc_{jobid:05d}.hdf5'
        else:
            raise ValueError("unknown output type chosen")

        self.Source['th_jet'] = self._th_jetList[it]
        self.Source['z'] = self._zList[iz]
        self.D = redshift2ComovingDistance(self.Source['z']) # comoving source distance

        # append options to file path
        self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['basedir'],
                                f"z{self.Source['z']:.3f}"))
        if self.Source['source_morphology'] == 'cone':
            self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['outdir'],
                            f"th_jet{self.Source['th_jet']}/"))
        elif self.Source['source_morphology'] == 'iso':
            self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['outdir'],
                                                'iso/'))
        elif self.Source['source_morphology'] == 'dir':
            self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['outdir'],
                                                'dir/'))
        else:
            raise ValueError("Chosen source morphology not supported.")
        self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['outdir'],
                        f"th_obs{self.Observer['obsAngle']}/"))
        self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['outdir'],
                        f"spec{self.Source['useSpectrum']:d}/"))

        # This is read from a loop
        self.Bfield['B'] = self._bList[idB]
        self.Bfield['gridSpacing'] = self._gridSpacingList[idL]
        solver = self.Simulation['propagation']
        solver_dict = {'CK': 'cash_karp', 'BP': 'boris_push'}

        if self.Bfield['type'] == 'turbulence':
            self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['outdir'],
                f"{solver_dict[solver]}/Bturb{self.Bfield['B']:.2e}/q{self.Bfield['turbulence']['turbIndex']:.2f}/scale{self.Bfield['gridSpacing']:.2f}/maxStep{self.Simulation['maxStepLength']}"))
        elif self.Bfield['type'] == 'cell':
            self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['outdir'],
                f"{solver_dict[solver]}/Bcell{self.Bfield['B']:.2e}/scale{self.Bfield['gridSpacing']:.2f}/maxStep{self.Simulation['maxStepLength']}"))
        elif self.Bfield['type'] == 'geo':
            self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['outdir'], 
                                                          solver_dict[solver], 'Bgeo', f"{self.Bfield['geo']['descriptor']}_filled", 
                                                          f"bvoid{self.Bfield['b_void']}_bext{self.Bfield['geo']['B']}", 
                                                          f"maxStep{self.Simulation['maxStepLength']}"))
        elif self.Bfield['type'] == 'txt':
            self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['outdir'], 
                                                          solver_dict[solver], 'Btxt', self.Bfield['txt']['fnDescriptor'], f"maxStep{self.Simulation['maxStepLength']}"))
        else:
            raise ValueError(f"Bfield type must be either 'cell' or 'turbulence' or 'txt' not {self.Bfield['type']}")
        

        self.photonoutputfile = str(path.join(self.FileIO['outdir'], self.PhotonOutName))

        if self.Observer["obsElectrons"]:
            self.electronoutputfile = str(path.join(self.FileIO['outdir'], "electrons_positrons", f"e_{self.PhotonOutName}"))
            # Make directory if it doesn't exist
            os.makedirs(os.path.dirname(self.electronoutputfile), exist_ok=True)

        logging.info(f"outdir: {self.FileIO['outdir']:s}")
        logging.info(f"outfile for photons: {self.photonoutputfile}")
        
        return

    def _create_bfield(self):
        """Set up simulation volume and magnetic field.
        PUT ALL UNITS IN SI."""
        # Origin for magnetic field grid, in all cases
        logging.info(f"Setting up B field with type: {self.Bfield['type'] }")
        gridOrigin = Vector3d(0, 0, 0)

        if self.Bfield['type'] == 'cell':
            #>>> 1*Mpc
            #   3.085677581491367e+22
            #   3.085677581491367e+22 meters = 1 Mpc
            #   THEREFORE x*Mpc converts the x (Mpc) to meters
            #   I believe SI units are used within CRPropa
            # >>> 1/Mpc
            #   3.240779289444365e-23
            #   3.240779289444365e-23 Mpc = 1 meter
            #   THEREFORE x/Mpc converts x (meters) to Mpc
            # >>> Mpc
            #   3.085677581491367e+22           
            gridSpacingMpcImplicit = self.Bfield['cell']['gridSpacing']
            gridSpacingMeters = gridSpacingMpcImplicit * Mpc
            # Number of grid cells to reach the observer sphere from the source
            # redshift2ComovingDistance returns a number in meters; convert to Mpc by diving by Mpc
            gridSize = int(np.ceil(redshift2ComovingDistance(self.Source['z']) / Mpc / gridSpacingMpcImplicit ))
            # floating point 3D vector grid 
            vgrid = Grid3f(gridOrigin,
                           gridSize,
                           gridSpacingMeters)
            initRandomField(vgrid, self.Bfield['cell']['B'] * gauss, seed=self.Bfield['cell']['seed'])
            self.bField = MagneticFieldGrid(vgrid)
            self.__extentMeters = gridSize * gridSpacingMeters
            logging.info(f"Box spacing for cell-like B field: {gridSpacingMpcImplicit} Mpc with {gridSize}^3 cells")
            logging.info('B field initialized')

        if self.Bfield['type'] == 'geo':
            # Create grid
            gridSpacingMpcImplicit = self.Bfield['geo']['gridSpacing']
            gridSpacingMeters = gridSpacingMpcImplicit * Mpc
            # Number of cells to pad grid to avoid edge effects
            gridPad = 2
            obsSize = int(np.ceil(redshift2ComovingDistance(self.Source['z']) / Mpc / gridSpacingMpcImplicit ))
            gridSize = obsSize + gridPad
            if gridSize % 2 != 0:
                msg = f"The grid size {gridSize} does not have an even length to divide in half. Increasing the grid size by 1."
                logging.warning(msg)
                print(msg)
                gridSize += 1
            vgrid = Grid3f(gridOrigin,
                    gridSize,
                    gridSpacingMeters)
            # Fill grid with vector values
            if self.Bfield['geo']['descriptor'] == 'outer_rectangular_prism':
                init_rectangular_prism_bfield(field_zero_near_origin=True,
                                              vgrid=vgrid, obsSize=obsSize, b_void=self.Bfield['b_void']*gauss, b_ext=self.Bfield['geo']['B']*gauss, seed=self.Bfield['geo']['seed'])
            elif self.Bfield['geo']['descriptor'] == 'inner_rectangular_prism':
                init_rectangular_prism_bfield(field_zero_near_origin=False, 
                                              vgrid=vgrid, obsSize=obsSize, b_void=self.Bfield['b_void']*gauss, b_ext=self.Bfield['geo']['B']*gauss, seed=self.Bfield['geo']['seed'])
            else:
                import sys
                sys.exit(f"Unrecognized type {self.Bfield['geo']['descriptor']}")
            # Create CRPropa magnetic field
            # Base field (positive x, y, z coordinates)
            bField0 = MagneticFieldGrid(vgrid)
            self.__extentMeters = gridSize * gridSpacingMeters
            self.bField = PeriodicMagneticField(bField0, Vector3d(self.__extentMeters), gridOrigin, bool(self.Bfield['geo']['reflective']))
   
        if self.Bfield['type'] == 'txt':
            fnTxt = self.Bfield['txt']['txtFile']
            gridSpacingMpcImplicit = self.Bfield['txt']['gridSpacing']  # Implicit/implied units of Mpc
            gridSpacingMeters = gridSpacingMpcImplicit * Mpc
            # Get size of grid; the number of grid cells per side of the cube
            with open(fnTxt) as f:
                lines = f.readlines()
                num_lines = len(lines)
            gridSize = int(round(num_lines**(1.0/3.0)))
            #  origin, N, spacing; boxSize = length of each size of individual cells: boxSize
            gridprops = GridProperties(gridOrigin, gridSize, gridSpacingMeters)
            vgrid = Grid3f(gridprops)
            loadGridFromTxt(vgrid, fnTxt)
            # Base magnetic field grid. Boundary conditions are applied to this.
            bField0 = MagneticFieldGrid(vgrid)
            # __extent has units of Mpc. From `gridSpacing`.
            self.__extentMeters =  gridSize * gridSpacingMeters
            # args: field, extends, origin, reflective
            if self.Bfield['txt']['reflective']:
                self.bField = PeriodicMagneticField(bField0, Vector3d(self.__extentMeters), gridOrigin, True)
            else:
                # Periodic; reflective=False
                self.bField = PeriodicMagneticField(bField0, Vector3d(self.__extentMeters), gridOrigin, False)
            if self.__extentMeters < redshift2ComovingDistance(self.config['Source']['z'])/Mpc:
                logging.error(f"The grid extent {self.__extentMeters/Mpc} Mpc is less than the comoving distance to the source {redshift2ComovingDistance(self.config['Source']['z'])/Mpc}")
            logging.info(f"B field initialized with file {fnTxt}, {gridSize} cells each of length {gridSpacingMpcImplicit} Mpc with reflective={self.Bfield['txt']['reflective']}")
            logging.info(f"Base grid extent {self.__extentMeters/Mpc} Mpc for comoving distance {redshift2ComovingDistance(self.config['Source']['z'])/Mpc} Mpc.")

        # Not in use so not tested
        if self.Bfield['type'] == 'turbulence':
            gridSpacingMpcImplicit = self.Bfield['turbulence']['gridSpacing'] * Mpc
            gridSize = int(np.ceil(redshift2ComovingDistance(self.Source['z'])/ gridSpacingMpcImplicit / Mpc))
            # floating point 3D vector grid 
            turbSpectrum = SimpleTurbulenceSpectrum(self.Bfield['turbulence']['B'] * gauss,  # Brms
                                                    2. * gridSpacingMpcImplicit,  #lMin
                                                    self.Bfield['turbulence']['maxTurbScale'] * Mpc,  #lMax
                                                    self.Bfield['turbulence']['turbIndex'])  #sIndex)
            gridprops = GridProperties(gridOrigin,
                                       gridSize,
                                       gridSpacingMpcImplicit)
            self.bField = SimpleGridTurbulence(turbSpectrum, gridprops, self.Bfield['turbulence']['seed'])
            self.__extentMeters = gridSpacingMpcImplicit
            logging.info('B field initialized')
            logging.info(f'Lc = {self.bField.getCorrelationLength() / kpc} kpc')  # correlation length, input in kpc

        try:
            logging.info(f'<B^2> = {self.bField.getBrms() / nG} nG')   # RMS
            logging.info(f'<|B|> = {self.bField.getMeanFieldStrength() / nG} nG')  # mean
        except AttributeError:
            pass
        logging.info(f'B(10 Mpc, 0, 0)={self.bField.getField(Vector3d(10,0,0) * Mpc) / nG} nG')
        logging.info(f'vgrid extension: {self.__extentMeters/Mpc} Mpc')
        return


    def _create_electron_positron_observer(self):
        """Set up the observer for the simulation. Observe electrons and positrons."""
        obsPosition = Vector3d(self.Observer['obsPosX'],self.Observer['obsPosY'],self.Observer['obsPosZ'])
        self.electron_observer = Observer()
        # also possible: detect particles upon exiting a shpere: 
        # ObserverLargeSphere (Vector3d center=Vector3d(0.), double radius=0)
        # radius is of large sphere is equal to source distance
        self.electron_observer.add(ObserverSurface(Sphere(obsPosition, self.D)))
        self.electron_observer.add(ObserverPhotonVeto())

        # for CR secondaries testing
        self.electron_observer.add(ObserverNucleusVeto())
        #ObserverNucleusVeto
        #ObserverTimeEvolution

        logging.info(f'Saving electron output to {self.electronoutputfile}')
        if self.Simulation.get('outputtype', 'ascii') == 'ascii':
            self.electron_output = TextOutput(self.electronoutputfile,
                                     Output.Event3D)
        elif self.Simulation.get('outputtype', 'ascii') == 'hdf5':
            self.electron_output = HDF5Output(self.electronoutputfile,
                                     Output.Event3D)
        else:
            raise ValueError("unknown output type chosen")

        self.electron_output.enable(Output.CurrentIdColumn)
        self.electron_output.enable(Output.CurrentDirectionColumn)
        self.electron_output.enable(Output.CurrentEnergyColumn)
        self.electron_output.enable(Output.CurrentPositionColumn)
        self.electron_output.enable(Output.CreatedIdColumn)
        self.electron_output.enable(Output.SourceEnergyColumn)
        self.electron_output.enable(Output.TrajectoryLengthColumn)
        self.electron_output.enable(Output.SourceDirectionColumn)
        self.electron_output.enable(Output.SourcePositionColumn)
        self.electron_output.enable(Output.WeightColumn)

        self.electron_output.disable(Output.RedshiftColumn)
        self.electron_output.disable(Output.CreatedDirectionColumn)
        self.electron_output.disable(Output.CreatedEnergyColumn)
        self.electron_output.disable(Output.CreatedPositionColumn)
        self.electron_output.disable(Output.SourceIdColumn)
        # we need this column for the blazar jet, don't disable
        #self.output.disable(Output.SourcePositionColumn)


        self.electron_output.setEnergyScale(eV)
        self.electron_observer.onDetection(self.electron_output)

        logging.info('Electron observer and output initialized')
        return
    

    def _create_photon_observer(self):
        """Set up the observer for the simulation. Observe photons only"""
        obsPosition = Vector3d(self.Observer['obsPosX'],self.Observer['obsPosY'],self.Observer['obsPosZ'])
        self.photon_observer = Observer()
        # also possible: detect particles upon exiting a shpere: 
        # ObserverLargeSphere (Vector3d center=Vector3d(0.), double radius=0)
        # radius is of large sphere is equal to source distance
        self.photon_observer.add(ObserverSurface(Sphere(obsPosition, self.D)))
        # looses a lot of particles -- need periodic boxes
        #Detects particles in a given redshift window. 
        #self.observer.add(ObserverRedshiftWindow(-1. * self.Observer['zmin'], self.Observer['zmin']))
        self.photon_observer.add(ObserverElectronVeto())

        # for CR secondaries testing
        self.photon_observer.add(ObserverNucleusVeto())
        #ObserverNucleusVeto
        #ObserverTimeEvolution

        self.photon_observer.setDeactivateOnDetection(True)

        logging.info(f'Saving photon output to {self.photonoutputfile}')
        if self.Simulation.get('outputtype', 'ascii') == 'ascii':
            self.photon_output = TextOutput(self.photonoutputfile,
                                     Output.Event3D)
        elif self.Simulation.get('outputtype', 'ascii') == 'hdf5':
            self.photon_output = HDF5Output(self.photonoutputfile,
                                     Output.Event3D)
        else:
            raise ValueError("unknown output type chosen")

        self.photon_output.enable(Output.CurrentIdColumn)
        self.photon_output.enable(Output.CurrentDirectionColumn)
        self.photon_output.enable(Output.CurrentEnergyColumn)
        self.photon_output.enable(Output.CurrentPositionColumn)
        self.photon_output.enable(Output.CreatedIdColumn)
        self.photon_output.enable(Output.SourceEnergyColumn)
        self.photon_output.enable(Output.TrajectoryLengthColumn)
        self.photon_output.enable(Output.SourceDirectionColumn)
        self.photon_output.enable(Output.SourcePositionColumn)
        self.photon_output.enable(Output.WeightColumn)

        if self.Simulation['CandidateTagColumn']:
            self.photon_output.enable(Output.CandidateTagColumn)

        self.photon_output.disable(Output.RedshiftColumn)
        self.photon_output.disable(Output.CreatedDirectionColumn)
        self.photon_output.disable(Output.CreatedEnergyColumn)
        self.photon_output.disable(Output.CreatedPositionColumn)
        self.photon_output.disable(Output.SourceIdColumn)
        # we need this column for the blazar jet, don't disable
        #self.output.disable(Output.SourcePositionColumn)


        self.photon_output.setEnergyScale(eV)
        self.photon_observer.onDetection(self.photon_output)

        logging.info('Photon observer and output initialized')
        return


    def _create_source(self):
        """Set up the source for the simulation"""
        self.source = Source()
        self.source.add(SourceRedshift(self.Source['z']))
        if self.Observer['obsLargeSphere']:
            obsPosition = Vector3d(self.Observer['obsPosX'],self.Observer['obsPosY'],self.Observer['obsPosZ'])
            # obs position same as source position for LargeSphere Observer
            self.source.add(SourcePosition(obsPosition))
            # emission cone towards positive x-axis
            if self.Source['source_morphology'] == 'cone':
                self.source.add(SourceEmissionCone(
                    Vector3d(np.cos(np.radians(self.Observer['obsAngle'])), 
                             np.sin(np.radians(self.Observer['obsAngle'])), 0), 
                             # Convert deg to rad bc:
                             # ss << "half-opening angle = " << aperture << " rad\n";
                             np.radians(self.Source['th_jet'])))
            elif self.Source['source_morphology'] == 'iso':
                self.source.add(SourceIsotropicEmission())
            elif self.Source['source_morphology'] == 'dir':
                self.source.add(SourceDirection(
                                    Vector3d(np.cos(np.radians(self.Observer['obsAngle'])), 
                                        np.sin(np.radians(self.Observer['obsAngle'])), 0)
                                    ))
            else:
                raise ValueError("Chosen source morphology not supported.")
        else:
            raise ValueError("Observer small sphere not available in CRPropa v3.2")
        # SourceParticleType takes int for particle ID. 
        # for a nucleus with A,Z you can use nucleusId(int a, int z) function
        # other IDs are given in http://pdg.lbl.gov/2016/reviews/rpp2016-rev-monte-carlo-numbering.pdf
        # e- : 11, e+ -11 ; antiparticles have negative sign
        # nu_e : 12
        # mu : 13
        # nu_mu : 14
        # nu_tau : 16
        # proton: 2212
        if self.Source['useSpectrum']:
            # for a power law use SourcePowerLawSpectrum (double Emin, double Emax, double index)
            logging.info(f"Using power spectrum E^{self.Source['index']}")
            self.source.add(SourcePowerLawSpectrum(self.Source['Emin'] * eV, 
                                                   self.Source['Emax'] * eV, 
                                                   self.Source['index']))
        else:
        # mono-energetic particle:
            self.source.add(SourceEnergy(self.Source['Energy'] * eV))
        self.source.add(SourceParticleType(self.Source['Composition']))
        logging.info('source initialized')
        return

    def _setup_emcascade(self):
        """Setup simulation module for electromagnetic cascade"""
        self.m = ModuleList()

        if self.Simulation['propagation'] == 'CK':
            #PropagationCK (ref_ptr< MagneticField > field=NULL, double tolerance=1e-4, double minStep=(0.1 *kpc), double maxStep=(1 *Gpc))
            logging.info("Using CK propagation module")
            self.m.add(PropagationCK(self.bField, self.Simulation['tol'],
                                     self.Simulation['minStepLength'] * pc,
                                     self.Simulation['maxStepLength'] * Mpc))

        elif self.Simulation['propagation'] == 'BP':
            # PropagationBP(ref_ptr<Ma.gneticField> field, double tolerance, double minStep, double maxStep)
            logging.info("Using BP propagation module")
            self.m.add(PropagationBP(self.bField, self.Simulation['tol'],
                                     self.Simulation['minStepLength'] * pc,
                                     self.Simulation['maxStepLength'] * Mpc))
        else:
            raise ValueError("unknown propagation module chosen")

        # Track all particles
        thinning = self.Simulation['thinning']
        # Updates redshift and applies adiabatic energy loss according to the traveled distance. 
        #m.add(Redshift())
        # Updates redshift and applies adiabatic energy loss according to the traveled distance. 
        # Extends to negative redshift values to allow for symmetric time windows around z=0
        if self.Simulation['include_z_evol']:
            self.m.add(FutureRedshift())

        # Interactions involving CMB
        if self.Simulation['include_CMB']:
            self.m.add(EMInverseComptonScattering(CMB(), True, thinning))
            # EMPairProduction:  electron-pair production of cosmic ray photons 
            # with background photons: gamma + gamma_b -> e+ + e- (Breit-Wheeler process).
            self.m.add(EMPairProduction(CMB(), True, thinning))
            if self.Simulation['include_higher_order_pp']:
                self.m.add(EMDoublePairProduction(CMB(), True, thinning))
                self.m.add(EMTripletPairProduction(CMB(), True, thinning))

        if self.Simulation['include_EBL']:
            self.m.add(EMInverseComptonScattering(self._EBL(), True, thinning))
            try:
                # CRpropa version with 
                # possibility to deactivate small angle approximation
                self.m.add(EMPairProduction(self._EBL(), True, thinning, self.Simulation['forward_approx']))
                logging.info(f"Using forward approx: {self.Simulation.get('forward_approx', True)} (if this is false, simulation will be slower!)")
            except:
                self.m.add(EMPairProduction(self._EBL(), True, thinning))

            if self.Simulation['include_higher_order_pp']:
                self.m.add(EMDoublePairProduction(self._EBL(), True, thinning))
                self.m.add(EMTripletPairProduction(self._EBL(), True, thinning))

        # Synchrotron radiation: 
        #SynchrotronRadiation (ref_ptr< MagneticField > field, bool havePhotons=false, double limit=0.1) or 
        #SynchrotronRadiation (double Brms=0, bool havePhotons=false, double limit=0.1) ; 
        #Large number of particles can cause memory problems!
        if self.Simulation['include_sync']:
            self.m.add(SynchrotronRadiation(self.bField, True, thinning))
        logging.info('modules initialized')
        return

    # def _setup_crcascade(self):
    #     """
    #     Setup simulation module for cascade initiated by cosmic rays
        
    #     kwargs 
    #     ------
    #     """
    #     if self.emcasc:
    #         photons = True
    #         electrons = True
    #         neutrinos = False
    #     else:
    #         photons = False
    #         electrons = False
    #         neutrinos = True
    #     antinucleons = False if self.Source['Composition'] == 2212 else True
    #     limit = self.Simulation.get('thinning', 0.1)
    #     logging.info("Set limit (= fraction of the mean free path, to which the propagation step will be limited)" +\
    #                  f" for PhotoPionProduction to {limit}")
    #     if limit < 0.5:
    #         logging.warning("for high energies, set limit to >= 0.5 to avoid memory problems")

    #     self.m = ModuleList()
    #     #PropagationCK (ref_ptr< MagneticField > field=NULL, double tolerance=1e-4,
    #     #double minStep=(0.1 *kpc), double maxStep=(1 *Gpc))
    #     #self.m.add(PropagationCK(self.bField, 1e-2, 100 * kpc, 10 * Mpc))
    #     #self.m.add(PropagationCK(self.bField, 1e-9, 1 * pc, 10 * Mpc))
    #     if self.Source['Energy'] >= 1e18 and self.emcasc:
    #         logging.info("Energy is greater than 1 EeV, limiting " \
    #                     f"sensitivity due to memory. E = {self.Source['Energy']}")
    #         #self.m.add(PropagationCK(self.bField, 1e-6, 1 * kpc, 10 * Mpc))
    #         tol = np.max([1e-4, self.Simulation['tol']])
    #     else:
    #         tol = self.Simulation['tol']
    #         # this takes about a factor of five longer:
    #         #self.m.add(PropagationCK(self.bField, 1e-9, 1 * pc, 10 * Mpc))
    #         # than this:
    #         #self.m.add(PropagationCK(self.bField, 1e-6, 1 * kpc, 10 * Mpc))

    #     if self.Simulation.get('propagation', 'CK') == 'CK':
    #         #PropagationCK (ref_ptr< MagneticField > field=NULL, double tolerance=1e-4, double minStep=(0.1 *kpc), double maxStep=(1 *Gpc))
    #         logging.info("Using CK propagation module")
    #         self.m.add(PropagationCK(self.bField, tol,
    #                                  self.Simulation['minStepLength'] * pc,
    #                                  self.Simulation['maxStepLength'] * Mpc))

    #     elif self.Simulation.get('propagation', 'CK') == 'BP':
    #         # PropagationBP(ref_ptr<Ma.gneticField> field, double tolerance, double minStep, double maxStep)
    #         logging.info("Using BP propagation module")
    #         self.m.add(PropagationBP(self.bField, tol,
    #                                  self.Simulation['minStepLength'] * pc,
    #                                  self.Simulation['maxStepLength'] * Mpc))
    #     else:
    #         raise ValueError("unknown propagation module chosen")

    #     thinning = self.Simulation.get('thinning', 0.)
    #     logging.info(f"Using thinning {thinning}")
    #     if thinning <= 0.1:
    #         logging.warning("for high energies, you might want to choose higher thinning values (close to 1.)")
    #     # Updates redshift and applies adiabatic energy loss according to the traveled distance. 
    #     #m.add(Redshift())
    #     # Updates redshift and applies adiabatic energy loss according to the traveled distance. 
    #     # Extends to negative redshift values to allow for symmetric time windows around z=0
    #     self.m.add(FutureRedshift())
    #     if self.emcasc:
    #         #self.m.add(EMInverseComptonScattering(CMB, photons, limit)) # not activated in example notebook
    #         #self.m.add(EMInverseComptonScattering(self._EBL(), photons, limit)) # not activated in example notebook
    #         # EMPairProduction:  electron-pair production of cosmic ray photons 
    #         #with background photons: gamma + gamma_b -> e+ + e- (Breit-Wheeler process).
    #         # EMPairProduction(PhotonField photonField = CMB, bool haveElectrons = false,double limit = 0.1 ), 
    #         #if haveElectrons = true, electron positron pair is created
    #         # EMInverComptonScattering(PhotonField photonField = CMB,bool havePhotons = false,double limit = 0.1 ), 
    #         #if havePhotons = True, photons are created
    #         # also availableL EMDoublePairProduction, EMTripletPairProduction
    #         #self.m.add(EMPairProduction(self._EBL(), electrons, limit)) # not activated in example notebook

    #         #self.m.add(EMPairProduction(CMB(), electrons, limit)) # not activated in example notebook

    #         self.m.add(EMInverseComptonScattering(CMB(), photons, thinning))
    #         self.m.add(EMInverseComptonScattering(self._URB(), photons, thinning))
    #         self.m.add(EMInverseComptonScattering(self._EBL(), photons, thinning))

    #         self.m.add(EMPairProduction(CMB(), electrons, thinning))
    #         self.m.add(EMPairProduction(self._URB(), electrons, thinning))
    #         self.m.add(EMPairProduction(self._EBL(), electrons, thinning))
    #         self.m.add(EMDoublePairProduction(CMB(), electrons, thinning))

    #         self.m.add(EMDoublePairProduction(self._URB(), electrons, thinning))
    #         self.m.add(EMDoublePairProduction(self._EBL(), electrons, thinning))

    #         self.m.add(EMTripletPairProduction(CMB(), electrons, thinning))
    #         self.m.add(EMTripletPairProduction(self._URB(), electrons, thinning))
    #         self.m.add(EMTripletPairProduction(self._EBL(), electrons, thinning))

    #     # for photo-pion production: 
    #     # PhotoPionProduction (PhotonField photonField=CMB, bool photons=false, bool neutrinos=false, 
    #     # bool electrons=false, bool antiNucleons=false, double limit=0.1, bool haveRedshiftDependence=false)
    #     self.m.add(PhotoPionProduction(CMB(), photons, neutrinos, electrons, antinucleons, limit, True))
    #     self.m.add(PhotoPionProduction(self._EBL(), photons, neutrinos, electrons, antinucleons, limit, True))

    #     # ElectronPairProduction (PhotonField photonField=CMB, bool haveElectrons=false, double limit=0.1)
    #     # Electron-pair production of charged nuclei with background photons. 
    #     self.m.add(ElectronPairProduction(CMB(), electrons, limit))
    #     self.m.add(ElectronPairProduction(self._EBL(), electrons, limit))
    #     if not self.Source['Composition'] == 2212: # protons don't decay or diseintegrate
    #         # for nuclear decay:
    #         #NuclearDecay (bool electrons=false, bool photons=false, bool neutrinos=false, double limit=0.1)
    #         self.m.add(NuclearDecay(electrons, photons, neutrinos))
    #         # for photo disentigration:
    #         #PhotoDisintegration (PhotonField photonField=CMB, bool havePhotons=false, double limit=0.1)
    #         self.m.add(PhotoDisintegration(CMB(), photons))
    #         self.m.add(PhotoDisintegration(self._EBL(), photons))
    #     # Synchrotron radiation: 
    #     #SynchrotronRadiation (ref_ptr< MagneticField > field, bool havePhotons=false, double limit=0.1) or 
    #     #SynchrotronRadiation (double Brms=0, bool havePhotons=false, double limit=0.1) ; 
    #     #Large number of particles can cause memory problems!
    #     #self.m.add(SynchrotronRadiation(self.bField, photons)) # not in example notebook
    #     logging.info('modules initialized')
    #     return

    def _setup_break(self):
        """Setup breaking conditions"""
        # add breaking conditions
        self.m.add(MinimumEnergy(self.BreakConditions['Emin'] * eV))
        self.m.add(self.photon_observer)
        if self.Observer["obsElectrons"]:
            self.m.add(self.electron_observer)
        # stop tracing particle once its propagation is longer than Dmax
        # or 1.5 * comoving distance of distance > 100. Mpc. 
        # this would anyway correspond to a very long time delay of > 50. Mpc / c
        #if self.D / Mpc > 100.:
            #dmax = np.min([self.BreakConditions['Dmax'] * 1000.,self.D * 1.5 / Mpc])
        #else: 
        dmax = self.BreakConditions['Dmax'] * 1000.
        self.m.add(MaximumTrajectoryLength(dmax * Mpc)) # Dmax is COMOVING
        # deactivate particle below a certain redshift
        if self.Observer['zmin'] is not None:
            self.m.add(MinimumRedshift(-1. * self.Observer['zmin']))

        # apply cut on rigidity for EM cascades
        rigidity = self.BreakConditions['minRigidity']
        # calc min rigidity of electron that produces average energy 
        # larger than MinimumEnergy
        # gamma factor of electron is given by gamma^2 = MinimumEnergy / mean CMB energy * 3 / 4
        # where mean CMB energy is 634 micro eV 
        # and where IC scattering in Thomson regime is assumed.
        # and rigidity is R = p c / q = mc^2 * sqrt(gamma^2 - 1) / q
        # divide min energy by 10 to be conservative
        min_rigidity = np.sqrt( 3. / 4. * self.BreakConditions['Emin'] / 634.e-6 / 10. - 1.)

        # this below is the prefactor m c^2 / q in Volt
        min_rigidity *= crpropa.mass_electron * crpropa.c_squared / crpropa.eV * crpropa.volt
        logging.info(f"The minimum electron / positron rigidity should be <~ {min_rigidity / 1e9} GV")

        if rigidity > 0.:

            if rigidity > min_rigidity / 1e9:
                raise ValueError(f"chosen minimal rigidity {rigidity} GV too large for minimum chosen photon energy {self.BreakConditions['Emin'] * eV} eV")

            self.m.add(MinimumRigidity(rigidity * crpropa.giga * crpropa.volt))
            logging.info(f"Set minimum rigidity to {rigidity} GV")

        else:
            logging.info("No cut on Rigidity set")

        # periodic boundaries
        #self.extent is the size of the B field grid        
        #sim.add(PeriodicBox(Vector3d(-self.__extent), Vector3d(2 * self.__extent)))
        logging.info('breaking conditions initialized')
        return

    def setup(self):
        """Setup the simulation"""
        self._create_bfield()
        if self.Source['Composition'] == 22 or \
            self.Source['Composition'] == 11 or \
            self.Source['Composition'] == -11:
            self._setup_emcascade()
        else:
            raise ValueError("CR cascaded commented out. Uncomment.")
            #self._setup_crcascade()
        self._create_source()
        self._create_photon_observer()
        if self.Observer["obsElectrons"]:
            self._create_electron_positron_observer()
            logging.info("Setup electron observer")
        self._setup_break()
        return 

    def run(self,  overwrite=False, force_combine=False, overwrite_combine=False,
        **kwargs):
        """Submit simulation jobs"""
        logging.info("Submitting job")
        option = ""   # extra options passed to run crpropa sim script

        script = path.join(path.abspath(path.dirname(simCRpropa.__file__)), 'scripts/run_crpropa_em_cascade.py')
        print (script)

        if not path.isfile(script):
            raise IOError(f"Script {script} not found!")
        
        # FIXME this trace is never reached
        import pdb; pdb.set_trace()
        
        logging.info("Looping over:")
        logging.info(self._bList, self._gridSpacingList)
        for ib, b in enumerate(self._bList):
            for il, l in enumerate(self._gridSpacingList):
                for it, t in enumerate(self._th_jetList):
                    for iz, z in enumerate(self._zList):
                        njobs = int(self._multiplicity[ib])
                        self.Simulation['multiplicity'] = int(self._multiplicity[ib])
                        self.Simulation['minStepLength'] = self._minStepLength[ib]
                        self.Simulation.pop('minTresol', None)  # delete resolution, as step length is set
                        self.Bfield['gridSpacing'] = l
                        if self.Bfield['type'] != 'txt':
                            self.Bfield['B'] = b
                            
                        else:
                            # `b` is a filename in the case of type=txt
                            # TODO retrieve filename and pass it somewhere?
                            self.Bfield['B'] = 'txt'
                        self.Source['th_jet'] = t
                        self.Source['z'] = z
                        self.D = redshift2ComovingDistance(self.Source['z']) # comoving source distance
                        self.setOutput(0, idB=ib, idL=il, it=it, iz=iz)

                        photonoutfile = path.join(self.FileIO['outdir'], self.PhotonOutName.split('_')[0] + '*.hdf5')
                        missing = utils.missing_files(photonoutfile,njobs, split = '.hdf5')
                        self.config['Simulation']['n_cpu'] = kwargs['n']

                        if len(missing) < njobs:
                            logging.debug(f'here {njobs}')
                            njobs = missing
                            logging.info(f'there are {len(missing)} files missing in {photonoutfile}')

                        if len(missing) and not force_combine:
                            self.config['configname'] = 'r'
                            kwargs['logdir'] = path.join(self.FileIO['outdir'],'log/')
                            kwargs['tmpdir'] = path.join(self.FileIO['outdir'],'tmp/')
                            kwargs['jname'] = f"b{np.log10(b):.2f}l{np.log10(l):.2f}th{t:.2f}z{z:.3f}{self.Simulation.get('name', '')}"
                            kwargs['log'] = path.join(kwargs['logdir'], kwargs['jname'] + ".out")
                            kwargs['err'] = path.join(kwargs['logdir'], kwargs['jname'] + ".err")

                            # submit job to either to lsdf or sdf
                            if 'sdf' in socket.gethostname():
                                _submit_run_sdf(script,
                                                self.config,
                                                option,
                                                njobs, 
                                                **kwargs)
                            else:
                                _submit_run_lsf(script,
                                                self.config,
                                                option,
                                                njobs, 
                                                **kwargs)
                        else:
                            if len(missing) and force_combine:
                                logging.info("There are files missing but combining anyways.")
                            else:
                                logging.info("All files present.")

                            ffdat = glob(path.join(path.dirname(photonoutfile),
                                               path.basename(photonoutfile).split('.hdf5')[0] + '.dat'))
                            if len(ffdat):
                                logging.info("Deleting *.dat files.")
                                for f in ffdat:
                                    utils.rm(f)

                            collect.combine_output(photonoutfile, overwrite=overwrite_combine)
        return

@lsf.setLsf
def main(**kwargs):
    usage = "usage: %(prog)s"
    description = "Run the analysis"
    parser = argparse.ArgumentParser(usage=usage,description=description)
    parser.add_argument('--conf', required=True)
    parser.add_argument('--dry', default=0, action="store_true")
    parser.add_argument('--time', default='09:59',help='Max time for lsf cluster job')
    parser.add_argument('--n', default=8,help='number of reserved cores', type=int)
    parser.add_argument('--span', default='span[ptile=8]',help='spanning of jobs on lsf cluster')
    parser.add_argument('--concurrent', default=0,help='number of max simultaneous jobs', type=int)
    parser.add_argument('--sleep', default=2, help='seconds to sleep between job submissions', type=int)
    parser.add_argument('--overwrite', help='overwrite existing combined files', action="store_true")
    parser.add_argument('--overwrite_combine', help='overwrite existing combined files', action="store_true")
    parser.add_argument('--force_combine', help='force the combination of files', action="store_true")
    parser.add_argument('--resubmit-running-jobs', action="store_false", default=True, help='Resubmit jobs even if they are running')
    parser.add_argument('--mem', help='mimimum requested memory in MB for SDF cluster', type=int)
    parser.add_argument('--loglevel', help='logging level', default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args()

    kwargs['dry'] = args.dry
    kwargs['time'] = args.time
    kwargs['concurrent'] = args.concurrent
    kwargs['sleep'] = args.sleep
    kwargs['n'] = args.n
    kwargs['span'] = args.span
    kwargs['mem'] = args.mem
    kwargs['no_resubmit_running_jobs'] = args.resubmit_running_jobs
    
    utils.init_logging(args.loglevel, color=True)

    with open(args.conf) as f:
        config = yaml.safe_load(f)

    sim = SimCRPropa(**config)
    sim.run(overwrite=bool(args.overwrite),
        force_combine=bool(args.force_combine),
        overwrite_combine=bool(args.overwrite_combine),
        **kwargs)
    return sim

if __name__ == '__main__':
    sim = main()
