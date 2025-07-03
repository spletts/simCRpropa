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


def initRandomField(vgrid, Bamplitude, seed=0):
    np.random.seed(seed)
    gridArray = vgrid.getGrid()
    nx = vgrid.getNx()
    ny = vgrid.getNy()
    nz = vgrid.getNz()
    logging.info(f"vgrid: nx = {nx}, ny = {ny}, nz = {nz}")
    for xi in range(0,nx):
        for yi in range(0,ny):
            for zi in range(0,nz):
                vect3d = vgrid.get(xi,yi,zi)

                x = np.random.uniform(-1,1)
                y = np.random.uniform(-1,1)
                z = np.random.uniform(-1,1)
                d = np.sqrt(x*x+y*y+z*z)
                #while d > 1:  
                #    x = np.random.uniform(-1,1)
                #    y = np.random.uniform(-1,1)
                #    z = np.random.uniform(-1,1)
                #    d = np.sqrt(x*x+y*y+z*z)

                vect3d.x = Bamplitude * x/d
                vect3d.y = Bamplitude * y/d
                vect3d.z = Bamplitude * z/d
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
        m = (combined['simEM/ID1/Ebin{0:03n}'.format(i)][...] == np.abs(intparent)) \
                & (combined['simEM/ID/Ebin{0:03n}'.format(i)][...] == np.abs(obs))
        h = np.histogram(combined['simEM/E/Ebin{0:03n}'.format(i)][m], bins = Ebins)
        intspec[i,:] = h[0]
                                
        m = (combined['simEM/ID1/Ebin{0:03n}'.format(i)][...] == np.abs(cascparent)) \
                & (combined['simEM/ID/Ebin{0:03n}'.format(i)][...] == np.abs(obs))
        h = np.histogram(combined['simEM/E/Ebin{0:03n}'.format(i)][m], bins = Ebins)
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
                                
        m = combined['simEM/ID/{0:s}'.format(k)][...] == np.abs(obs)
        h = np.histogram(combined['simEM/E/{0:s}'.format(k)][m], bins = Ebins)
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

        for i, k in enumerate(['B', 'maxTurbScale']):
            if isinstance(self.Bfield[k], list):
                x = deepcopy(self.Bfield[k])
                self.Bfield[k] = x[0]
            elif isinstance(self.Bfield[k], float):
                x = [self.Bfield[k]]
            else:
                raise ValueError(f"{self.Bfield[k]} type not understood: {type(self.Bfield[k])}")
            if not i:
                self._bList = x
            else:
                self._turbScaleList = x

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

            # increase number of weights for small scale observer
            if self.Observer['obsSmallSphere']:
                # weight for Bfield
                if self.Bfield['B'] > 1e-18:
                    self.weights *= (1. + 0.1 * (np.log10(self.Bfield['B']) + 18.)**2.)
                # weight for jet opening angle 
                if self.Source['th_jet'] > 1.:
                    self.weights *= (1. + 0.1 *self.Source['th_jet']**2.)
                if self.Observer['obsAngle'] > 0.:
                    self.weights *= (1. + 0.1 * (self.Observer['obsAngle'] + 1.))

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
            self.OutName = 'casc_{0:05d}.dat'.format(jobid)
        elif self.Simulation.get('outputtype', 'ascii') == 'hdf5':
            self.OutName = 'casc_{0:05d}.hdf5'.format(jobid)
        else:
            raise ValueError("unknown output type chosen")

        self.Source['th_jet'] = self._th_jetList[it]
        self.Source['z'] = self._zList[iz]
        self.D = redshift2ComovingDistance(self.Source['z']) # comoving source distance

        # append options to file path
        self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['basedir'],
                                'z{0[z]:.3f}'.format(self.Source)))
        if self.Source.get('source_morphology', 'cone') == 'cone':
            self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['outdir'],
                            'th_jet{0[th_jet]}/'.format(self.Source)))
        elif self.Source.get('source_morphology', 'cone') == 'iso':
            self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['outdir'],
                                                'iso/'))
        elif self.Source.get('source_morphology', 'cone') == 'dir':
            self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['outdir'],
                                                'dir/'))
        else:
            raise ValueError("Chosen source morphology not supported.")
        self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['outdir'],
                        'th_obs{0[obsAngle]}/'.format(self.Observer)))
        self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['outdir'],
                        'spec{0[useSpectrum]:d}/'.format(self.Source)))

        self.Bfield['B'] = self._bList[idB]
        self.Bfield['maxTurbScale'] = self._turbScaleList[idL]

        if self.Bfield['type'] == 'turbulence':
            self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['outdir'],
                'Bturb{0[B]:.2e}/q{0[turbIndex]:.2f}/scale{0[maxTurbScale]:.2f}/'.format(self.Bfield)))
        elif self.Bfield['type'] =='cell':
            self.FileIO['outdir'] = utils.mkdir(path.join(self.FileIO['outdir'],
                'Bcell{0[B]:.2e}/scale{0[maxTurbScale]:.2f}/'.format(self.Bfield)))
        else:
            raise ValueError("Bfield type must be either 'cell' or 'turbulence' not {0[type]}".format(self.Bfield))

        self.outputfile = str(path.join(self.FileIO['outdir'], self.OutName))
        logging.info("outdir: {0[outdir]:s}".format(self.FileIO))
        logging.info("outfile: {0:s}".format(self.outputfile))
        return

    def _create_bfield(self):
        """Set up simulation volume and magnetic field"""
        boxOrigin = Vector3d(0, 0, 0)
        boxSpacing = self.Bfield['boxSize'] * Mpc / self.Bfield['NBgrid']
        logging.info(f'Box spacing for B field: {boxSpacing / Mpc} Mpc')

        if self.Bfield['type'] == 'turbulence':
            print(2. * boxSpacing / Mpc, self.Bfield['maxTurbScale'])
            turbSpectrum = SimpleTurbulenceSpectrum(self.Bfield['B'] * gauss,  # Brms
                                                    2. * boxSpacing,  #lMin
                                                    self.Bfield['maxTurbScale'] * Mpc,  #lMax
                                                    self.Bfield['turbIndex'])  #sIndex)

            gridprops = GridProperties(boxOrigin,
                                       self.Bfield['NBgrid'],
                                       boxSpacing)

            self.bField = SimpleGridTurbulence(turbSpectrum, gridprops, self.Bfield['seed'])


            #vgrid = Grid3f(boxOrigin,
            #               self.Bfield['NBgrid'],
            #               boxSpacing)
            #initTurbulence(vgrid, self.Bfield['B'] * gauss, 
            #                2 * boxSpacing, 
            #                self.Bfield['maxTurbScale'] * Mpc, 
            #                self.Bfield['turbIndex'],
            #                self.Bfield['seed'])
            #bField0 = MagneticFieldGrid(vgrid)
            #self.bField = PeriodicMagneticField(bField0,
            #        Vector3d(self.Bfield['periodicity']* Mpc), Vector3d(0), False)

            self.__extent = self.Bfield['boxSize'] * Mpc
            logging.info('B field initialized')
            logging.info(f'Lc = {self.bField.getCorrelationLength() / kpc} kpc')  # correlation length, input in kpc

        if self.Bfield['type'] == 'cell':
            logging.info(f"Box spacing for cell-like B field: {self.Bfield['maxTurbScale']} Mpc")
            gridSpacing = self.Bfield['maxTurbScale'] * Mpc
            if self.Bfield['NBgrid'] == 0:
                gridSize = int(np.ceil(redshift2ComovingDistance(self.Source['z'])/\
                                                self.Bfield['maxTurbScale'] / Mpc))
            else:
                gridSize = self.Bfield['NBgrid']

            # init 
            # floating point 3D vector grid 
            vgrid = Grid3f(boxOrigin,
                           gridSize,
                           gridSpacing)

            initRandomField(vgrid, self.Bfield['B'] * gauss, seed=self.Bfield['seed'])
            self.bField = MagneticFieldGrid(vgrid)
            self.__extent = int(np.ceil(redshift2ComovingDistance(self.Source['z'])/\
                                    self.Bfield['maxTurbScale'] / Mpc)) \
                                    * self.Bfield['maxTurbScale'] * Mpc
            logging.info('B field initialized')


        logging.info(f'vgrid extension: {self.__extent / Mpc} Mpc')
        try:
            logging.info(f'<B^2> = {self.bField.getBrms() / nG} nG')   # RMS
            logging.info(f'<|B|> = {self.bField.getMeanFieldStrength() / nG} nG')  # mean
        except AttributeError:
            pass
        logging.info(f'B(10 Mpc, 0, 0)={self.bField.getField(Vector3d(10,0,0) * Mpc) / nG} nG')
        return

    def _create_observer(self):
        """Set up the observer for the simulation"""
        obsPosition = Vector3d(self.Observer['obsPosX'],self.Observer['obsPosY'],self.Observer['obsPosZ'])
        self.observer = Observer()
        #ObserverSmallSphere (Vector3d center=Vector3d(0.), double radius=0)
        #Detects particles upon entering a sphere
        if self.Observer['obsSmallSphere']:
            self.observer.add(ObserverSmallSphere(obsPosition, self.Observer['obsSize'] * Mpc))
        else:
            # also possible: detect particles upon exiting a shpere: 
            # ObserverLargeSphere (Vector3d center=Vector3d(0.), double radius=0)
            # radius is of large sphere is equal to source distance
            #self.observer.add(ObserverLargeSphere(obsPosition, self.D))
            self.observer.add(ObserverSurface(Sphere(obsPosition, self.D)))
        # looses a lot of particles -- need periodic boxes
        #Detects particles in a given redshift window. 
        #self.observer.add(ObserverRedshiftWindow(-1. * self.Observer['zmin'], self.Observer['zmin']))
        self.observer.add(ObserverElectronVeto())

        # for CR secondaries testing
        self.observer.add(ObserverNucleusVeto())
        #ObserverNucleusVeto
        #ObserverTimeEvolution

        logging.info(f'Saving output to {self.outputfile}')
        if self.Simulation.get('outputtype', 'ascii') == 'ascii':
            self.output = TextOutput(self.outputfile,
                                     Output.Event3D)
        elif self.Simulation.get('outputtype', 'ascii') == 'hdf5':
            self.output = HDF5Output(self.outputfile,
                                     Output.Event3D)
        else:
            raise ValueError("unknown output type chosen")

        self.output.enable(Output.CurrentIdColumn)
        self.output.enable(Output.CurrentDirectionColumn)
        self.output.enable(Output.CurrentEnergyColumn)
        self.output.enable(Output.CurrentPositionColumn)
        self.output.enable(Output.CreatedIdColumn)
        self.output.enable(Output.SourceEnergyColumn)
        self.output.enable(Output.TrajectoryLengthColumn)
        self.output.enable(Output.SourceDirectionColumn)
        self.output.enable(Output.SourcePositionColumn)
        self.output.enable(Output.WeightColumn)

        self.output.disable(Output.RedshiftColumn)
        self.output.disable(Output.CreatedDirectionColumn)
        self.output.disable(Output.CreatedEnergyColumn)
        self.output.disable(Output.CreatedPositionColumn)
        self.output.disable(Output.SourceIdColumn)
        # we need this column for the blazar jet, don't disable
        #self.output.disable(Output.SourcePositionColumn)


        self.output.setEnergyScale(eV)
        self.observer.onDetection(self.output)

        logging.info('Observer and output initialized')
        return

    def _create_source(self):
        """Set up the source for the simulation"""
        self.source = Source()
        self.source.add(SourceRedshift(self.Source['z']))
        if self.Observer['obsSmallSphere']:
            self.source.add(SourcePosition(Vector3d(self.D, 0, 0)))
            # emission to negativ x-axis
            if self.Source.get('source_morphology', 'cone') == 'cone':
                self.source.add(SourceEmissionCone(
                                    Vector3d(np.cos(np.pi - np.radians(self.Observer['obsAngle'])), 
                                        np.sin(np.pi - np.radians(self.Observer['obsAngle'])), 0), 
                                    np.radians(self.Source['th_jet'])))
            elif self.Source.get('source_morphology', 'cone') == 'dir':
                self.source.add(SourceDirection(
                                    Vector3d(np.cos(np.pi - np.radians(self.Observer['obsAngle'])), 
                                        np.sin(np.pi - np.radians(self.Observer['obsAngle'])), 0)
                                    ))
            elif self.Source.get('source_morphology', 'cone') == 'iso':
                self.source.add(SourceIsotropicEmission())
            else:
                raise ValueError("Chosen source morphology not supported.")
        else:
            obsPosition = Vector3d(self.Observer['obsPosX'],self.Observer['obsPosY'],self.Observer['obsPosZ'])
            # obs position same as source position for LargeSphere Observer
            self.source.add(SourcePosition(obsPosition))
            # emission cone towards positiv x-axis
            if self.Source.get('source_morphology', 'cone') == 'cone':
                self.source.add(SourceEmissionCone(
                    Vector3d(np.cos(np.radians(self.Observer['obsAngle'])), 
                             np.sin(np.radians(self.Observer['obsAngle'])), 0), 
                             np.radians(self.Source['th_jet'])))
            elif self.Source.get('source_morphology', 'cone') == 'iso':
                self.source.add(SourceIsotropicEmission())
            elif self.Source.get('source_morphology', 'cone') == 'dir':
                self.source.add(SourceDirection(
                                    Vector3d(np.cos(np.pi - np.radians(self.Observer['obsAngle'])), 
                                        np.sin(np.pi - np.radians(self.Observer['obsAngle'])), 0)
                                    ))
            else:
                raise ValueError("Chosen source morphology not supported.")
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


        if self.Simulation.get('propagation', 'CK') == 'CK':
            #PropagationCK (ref_ptr< MagneticField > field=NULL, double tolerance=1e-4, double minStep=(0.1 *kpc), double maxStep=(1 *Gpc))
            logging.info("Using CK propagation module")
            self.m.add(PropagationCK(self.bField, self.Simulation['tol'],
                                     self.Simulation['minStepLength'] * pc,
                                     self.Simulation['maxStepLength'] * Mpc))

        elif self.Simulation.get('propagation', 'CK') == 'BP':
            # PropagationBP(ref_ptr<Ma.gneticField> field, double tolerance, double minStep, double maxStep)
            logging.info("Using BP propagation module")
            self.m.add(PropagationBP(self.bField, self.Simulation['tol'],
                                     self.Simulation['minStepLength'] * pc,
                                     self.Simulation['maxStepLength'] * Mpc))
        else:
            raise ValueError("unknown propagation module chosen")

        thinning = self.Simulation.get('thinning', 0.)
        # Updates redshift and applies adiabatic energy loss according to the traveled distance. 
        #m.add(Redshift())
        # Updates redshift and applies adiabatic energy loss according to the traveled distance. 
        # Extends to negative redshift values to allow for symmetric time windows around z=0
        if self.Simulation.get('include_z_evol', True):
            self.m.add(FutureRedshift())

        self.m.add(EMInverseComptonScattering(CMB(), True, thinning))
        if self.Simulation.get('include_CMB', True):
            # this is a bit counter intuitive here, but I just want to 
            # make a comparison to all the other codes by excluding the EBL here
            self.m.add(EMInverseComptonScattering(self._EBL(), True, thinning))
        # EMPairProduction:  electron-pair production of cosmic ray photons 
        #with background photons: gamma + gamma_b -> e+ + e- (Breit-Wheeler process).
        # EMPairProduction(PhotonField photonField = CMB, bool haveElectrons = false,double limit = 0.1 ), 
        #if haveElectrons = true, electron positron pair is created
        # EMInverComptonScattering(PhotonField photonField = CMB,bool havePhotons = false,double limit = 0.1 ), 
        #if havePhotons = True, photons are created
        # also availableL EMDoublePairProduction, EMTripletPairProduction
        try:
            # CRpropa version with 
            # possibility to deactivate small angle approximation
            self.m.add(EMPairProduction(self._EBL(), True, thinning, self.Simulation.get('forward_approx', True)))
            logging.info(f"Using forward approx: {self.Simulation.get('forward_approx', True)} (if this is false, simulation will be slower!)")
        except:
            self.m.add(EMPairProduction(self._EBL(), True, thinning))

        if self.Simulation.get('include_higher_order_pp', False):
            self.m.add(EMDoublePairProduction(self._EBL(), True, thinning))
            self.m.add(EMTripletPairProduction(self._EBL(), True, thinning))

        if self.Simulation.get('include_CMB', True):
            self.m.add(EMPairProduction(CMB(), True))
            if self.Simulation.get('include_higher_order_pp', False):
                self.m.add(EMDoublePairProduction(CMB(), True, thinning))
                self.m.add(EMTripletPairProduction(CMB(), True, thinning))

        # for photo-pion production: 
        #PhotoPionProduction (PhotonField photonField=CMB, bool photons=false, bool neutrinos=false, 
        # bool antiNucleons=false, double limit=0.1, bool haveRedshiftDependence=false)
        # for photo disentigration:
        #PhotoDisintegration (PhotonField photonField=CMB, bool havePhotons=false, double limit=0.1)
        # for nuclear decay:
        #NuclearDecay (bool electrons=false, bool photons=false, bool neutrinos=false, double limit=0.1)
        # Synchrotron radiation: 
        #SynchrotronRadiation (ref_ptr< MagneticField > field, bool havePhotons=false, double limit=0.1) or 
        #SynchrotronRadiation (double Brms=0, bool havePhotons=false, double limit=0.1) ; 
        #Large number of particles can cause memory problems!
        if self.Simulation.get('include_sync', True):
            self.m.add(SynchrotronRadiation(self.bField, True, thinning))
        logging.info('modules initialized')
        return

    def _setup_crcascade(self):
        """
        Setup simulation module for cascade initiated by cosmic rays
        
        kwargs 
        ------
        """
        if self.emcasc:
            photons = True
            electrons = True
            neutrinos = False
        else:
            photons = False
            electrons = False
            neutrinos = True
        antinucleons = False if self.Source['Composition'] == 2212 else True
        limit = self.Simulation.get('thinning', 0.1)
        logging.info("Set limit (= fraction of the mean free path, to which the propagation step will be limited)" +\
                     f" for PhotoPionProduction to {limit}")
        if limit < 0.5:
            logging.warning("for high energies, set limit to >= 0.5 to avoid memory problems")

        self.m = ModuleList()
        #PropagationCK (ref_ptr< MagneticField > field=NULL, double tolerance=1e-4,
        #double minStep=(0.1 *kpc), double maxStep=(1 *Gpc))
        #self.m.add(PropagationCK(self.bField, 1e-2, 100 * kpc, 10 * Mpc))
        #self.m.add(PropagationCK(self.bField, 1e-9, 1 * pc, 10 * Mpc))
        if self.Source['Energy'] >= 1e18 and self.emcasc:
            logging.info("Energy is greater than 1 EeV, limiting " \
                        f"sensitivity due to memory. E = {self.Source['Energy']}")
            #self.m.add(PropagationCK(self.bField, 1e-6, 1 * kpc, 10 * Mpc))
            tol = np.max([1e-4, self.Simulation['tol']])
        else:
            tol = self.Simulation['tol']
            # this takes about a factor of five longer:
            #self.m.add(PropagationCK(self.bField, 1e-9, 1 * pc, 10 * Mpc))
            # than this:
            #self.m.add(PropagationCK(self.bField, 1e-6, 1 * kpc, 10 * Mpc))

        if self.Simulation.get('propagation', 'CK') == 'CK':
            #PropagationCK (ref_ptr< MagneticField > field=NULL, double tolerance=1e-4, double minStep=(0.1 *kpc), double maxStep=(1 *Gpc))
            logging.info("Using CK propagation module")
            self.m.add(PropagationCK(self.bField, tol,
                                     self.Simulation['minStepLength'] * pc,
                                     self.Simulation['maxStepLength'] * Mpc))

        elif self.Simulation.get('propagation', 'CK') == 'BP':
            # PropagationBP(ref_ptr<Ma.gneticField> field, double tolerance, double minStep, double maxStep)
            logging.info("Using BP propagation module")
            self.m.add(PropagationBP(self.bField, tol,
                                     self.Simulation['minStepLength'] * pc,
                                     self.Simulation['maxStepLength'] * Mpc))
        else:
            raise ValueError("unknown propagation module chosen")

        thinning = self.Simulation.get('thinning', 0.)
        logging.info(f"Using thinning {thinning}")
        if thinning <= 0.1:
            logging.warning("for high energies, you might want to choose higher thinning values (close to 1.)")
        # Updates redshift and applies adiabatic energy loss according to the traveled distance. 
        #m.add(Redshift())
        # Updates redshift and applies adiabatic energy loss according to the traveled distance. 
        # Extends to negative redshift values to allow for symmetric time windows around z=0
        self.m.add(FutureRedshift())
        if self.emcasc:
            #self.m.add(EMInverseComptonScattering(CMB, photons, limit)) # not activated in example notebook
            #self.m.add(EMInverseComptonScattering(self._EBL(), photons, limit)) # not activated in example notebook
            # EMPairProduction:  electron-pair production of cosmic ray photons 
            #with background photons: gamma + gamma_b -> e+ + e- (Breit-Wheeler process).
            # EMPairProduction(PhotonField photonField = CMB, bool haveElectrons = false,double limit = 0.1 ), 
            #if haveElectrons = true, electron positron pair is created
            # EMInverComptonScattering(PhotonField photonField = CMB,bool havePhotons = false,double limit = 0.1 ), 
            #if havePhotons = True, photons are created
            # also availableL EMDoublePairProduction, EMTripletPairProduction
            #self.m.add(EMPairProduction(self._EBL(), electrons, limit)) # not activated in example notebook

            #self.m.add(EMPairProduction(CMB(), electrons, limit)) # not activated in example notebook

            self.m.add(EMInverseComptonScattering(CMB(), photons, thinning))
            self.m.add(EMInverseComptonScattering(self._URB(), photons, thinning))
            self.m.add(EMInverseComptonScattering(self._EBL(), photons, thinning))

            self.m.add(EMPairProduction(CMB(), electrons, thinning))
            self.m.add(EMPairProduction(self._URB(), electrons, thinning))
            self.m.add(EMPairProduction(self._EBL(), electrons, thinning))
            self.m.add(EMDoublePairProduction(CMB(), electrons, thinning))

            self.m.add(EMDoublePairProduction(self._URB(), electrons, thinning))
            self.m.add(EMDoublePairProduction(self._EBL(), electrons, thinning))

            self.m.add(EMTripletPairProduction(CMB(), electrons, thinning))
            self.m.add(EMTripletPairProduction(self._URB(), electrons, thinning))
            self.m.add(EMTripletPairProduction(self._EBL(), electrons, thinning))

        # for photo-pion production: 
        # PhotoPionProduction (PhotonField photonField=CMB, bool photons=false, bool neutrinos=false, 
        # bool electrons=false, bool antiNucleons=false, double limit=0.1, bool haveRedshiftDependence=false)
        self.m.add(PhotoPionProduction(CMB(), photons, neutrinos, electrons, antinucleons, limit, True))
        self.m.add(PhotoPionProduction(self._EBL(), photons, neutrinos, electrons, antinucleons, limit, True))

        # ElectronPairProduction (PhotonField photonField=CMB, bool haveElectrons=false, double limit=0.1)
        # Electron-pair production of charged nuclei with background photons. 
        self.m.add(ElectronPairProduction(CMB(), electrons, limit))
        self.m.add(ElectronPairProduction(self._EBL(), electrons, limit))
        if not self.Source['Composition'] == 2212: # protons don't decay or diseintegrate
            # for nuclear decay:
            #NuclearDecay (bool electrons=false, bool photons=false, bool neutrinos=false, double limit=0.1)
            self.m.add(NuclearDecay(electrons, photons, neutrinos))
            # for photo disentigration:
            #PhotoDisintegration (PhotonField photonField=CMB, bool havePhotons=false, double limit=0.1)
            self.m.add(PhotoDisintegration(CMB(), photons))
            self.m.add(PhotoDisintegration(self._EBL(), photons))
        # Synchrotron radiation: 
        #SynchrotronRadiation (ref_ptr< MagneticField > field, bool havePhotons=false, double limit=0.1) or 
        #SynchrotronRadiation (double Brms=0, bool havePhotons=false, double limit=0.1) ; 
        #Large number of particles can cause memory problems!
        #self.m.add(SynchrotronRadiation(self.bField, photons)) # not in example notebook
        logging.info('modules initialized')
        return

    def _setup_break(self):
        """Setup breaking conditions"""
        # add breaking conditions
        self.m.add(MinimumEnergy(self.BreakConditions['Emin'] * eV))
        self.m.add(self.observer)
        # stop tracing particle once it's propagation is longer than Dmax
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
        rigidity = self.BreakConditions.get('minRigidity', 50.)
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
            self._setup_crcascade()
        self._create_source()
        self._create_observer()
        self._setup_break()
        return 

    def run(self,  overwrite=False, force_combine=False, overwrite_combine=False,
        **kwargs):
        """Submit simulation jobs"""
        option = ""   # extra options passed to run crpropa sim script

        script = path.join(path.abspath(path.dirname(simCRpropa.__file__)), 'scripts/run_crpropa_em_cascade.py')
        print (script)

        if not path.isfile(script):
            raise IOError(f"Script {script} not found!")

        for ib, b in enumerate(self._bList):
            for il, l in enumerate(self._turbScaleList):
                for it, t in enumerate(self._th_jetList):
                    for iz, z in enumerate(self._zList):
                        njobs = int(self._multiplicity[ib])
                        self.Simulation['multiplicity'] = int(self._multiplicity[ib])
                        self.Simulation['minStepLength'] = self._minStepLength[ib]
                        self.Simulation.pop('minTresol', None)  # delete resolution, as step length is set
                        self.Bfield['B'] = b
                        self.Bfield['maxTurbScale'] = l
                        self.Source['th_jet'] = t
                        self.Source['z'] = z
                        self.D = redshift2ComovingDistance(self.Source['z']) # comoving source distance
                        self.setOutput(0, idB=ib, idL=il, it=it, iz=iz)

                        outfile = path.join(self.FileIO['outdir'], self.OutName.split('_')[0] + '*.hdf5')
                        missing = utils.missing_files(outfile,njobs, split = '.hdf5')
                        self.config['Simulation']['n_cpu'] = kwargs['n']

                        if len(missing) < njobs:
                            logging.debug(f'here {njobs}')
                            njobs = missing
                            logging.info(f'there are {len(missing)} files missing in {outfile}')

                        if len(missing) and not force_combine:
                            self.config['configname'] = 'r'
                            kwargs['logdir'] = path.join(self.FileIO['outdir'],'log/')
                            kwargs['tmpdir'] = path.join(self.FileIO['outdir'],'tmp/')
                            kwargs['jname'] = 'b{0:.2f}l{1:.2f}th{2:.2f}z{3:.3f}{4:s}'.format(
                                np.log10(b), np.log10(l), t, z, self.Simulation.get('name', ''))
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

                            ffdat = glob(path.join(path.dirname(outfile),
                                               path.basename(outfile).split('.hdf5')[0] + '.dat'))
                            if len(ffdat):
                                logging.info("Deleting *.dat files.")
                                for f in ffdat:
                                    utils.rm(f)

                            collect.combine_output(outfile, overwrite=overwrite_combine)
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
