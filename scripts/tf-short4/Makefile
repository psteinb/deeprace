
SLURM_FILES?=$(wildcard *slurm)
FIDUCIAL_TARGETS=$(SLURM_FILES:%.slurm=%.log)

all : submit

submit : $(FIDUCIAL_TARGETS)

%.log : %.slurm
	@echo $<
	@sbatch --array=1-10 -A p_gpuhack18_4 $<

show_found :
	echo $(SLURM_FILES)
	echo $(FIDUCIAL_TARGETS)
