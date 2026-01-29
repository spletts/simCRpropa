# simCRpropa

Some papers that have used simCRpropa which may explain some of the defaults in [`default_config.yaml`](default_config.yaml):
* [https://iopscience.iop.org/article/10.3847/2041-8213/acd777](https://iopscience.iop.org/article/10.3847/2041-8213/acd777)
* [https://arxiv.org/abs/2511.06116](https://arxiv.org/abs/2511.06116)

## Getting started

### Installation 

Clone the following repositories:
* [https://github.com/me-manu/fermiAnalysis](https://github.com/me-manu/fermiAnalysis)
* [https://github.com/spletts/simCRpropa](https://github.com/spletts/simCRpropa)

The Hummingbird team installed CRPropa, so you do not need to install it.

### Run a sample simulation

#### Write slurm script
To submit a `simCRpropa` job using slurm (which is UCSC Hummingbird's job manager), first edit [`sample_slurm.sh`](sample_slurm.sh) as follows:
* Add your email between the angle brackets: `--mail-user=<>` 
* In `PYTHONPATH`, update `/hb/home/mspletts/simCRPropaVoids/simCRpropa` accordingly, to your absolute path to simCRpropa
* In `PYTHONPATH`, update `/hb/home/mspletts/fermiAnalysis/`
* Update `LSCRATCH` to some absolute(?) path that will be used as a scratch directory
* You may want to update `simCRpropa/scripts/run_crpropa_em_cascade.py` to the absolute path (if not now, then in the future, so you can run your slurm scripts from any directory)
* You may want to update `--output` to an absolute path, but it is not necessary
* Note that `--array=0-1` will submit two identical jobs. To submit more or less jobs, adjust `--array` as necessary. I usually submit 500 jobs (`array=0-499` in the slurm script) and each injects 1000 particles (this is handled in the config file; see below).

Note that the Hummingbird team installed CRPropa, and the environment can be loaded via `module load crpropa`, but this is part of the slurm script.

#### Write config file
Edit the [`default_config.yaml`](default_config.yaml):
* Update the paths under `FileIO` to absolute paths in your own home directory

#### Submit job
`sample_slurm.sh` accepts one argument: the config file. 

Submit the job via
```
$ sbatch sample_slurm.sh default_config.yaml
```
(adjust the paths to these files if necessary)

You can monitor job status via `squeue -u your-username-here`. 
You will also get an emails about the job status if you kept `--mail-type=ALL` in the slurm script.

The job should take < 5 minutes to run.

#### Examine output
Once done, the slurm output/log file (set via `--output` in the slurm script) will print the full filepaths of the output files; additionally, the output will be within the config file's `outdir`. 

The slurm output file has something like:
```
INFO: copied /hb/home/mspletts/simCRPropaVoids/simCRpropa/output/_scratch/233.q3x3Fr/casc_00001.hdf5 to /hb/home/mspletts/simCRPropaVoids/simCRpropa/output/test/z0.050/th_jet0.1/th_obs0.0/inj_powlaw-1.5_expcut1.0TeV/boris_push/Bgeo/inner_rectangular_prism_filled/bvoid0.0_bext1e-13/scale50.00/maxStep0.001
```
In the directory ` /hb/home/mspletts/simCRPropaVoids/simCRpropa/output/test/z0.050/th_jet0.1/th_obs0.0/inj_powlaw-1.5_expcut1.0TeV/boris_push/Bgeo/inner_rectangular_prism_filled/bvoid0.0_bext1e-13/scale50.00/maxStep0.001` there should be two files: `casc_00000.hdf5` and `casc_00001.hdf5` (named with the job ID from `--array` in the slurm script). 
*These files are the final CRPropa output*.
You can open them with VSCode (it opens H5 files).
Alternately, you can view text file (this file is eventually converted to the hdf5 above). 
You can find this filename in the slurm output/log file as something like:
```
Saving photon output to /hb/home/mspletts/simCRPropaVoids/simCRpropa/output/_scratch/233.q3x3Fr/casc_00001.dat
```
*The text file describes each header*. 
Info is also at [https://crpropa.github.io/CRPropa3/buildingblocks/Output.html#output](https://crpropa.github.io/CRPropa3/buildingblocks/Output.html#output).


## General usage

See comments in `default_config.yaml` for simulation options.  

TODO: