# Running the behavioral experiment on MIT Engaging (SLURM + GPU)

The compute is the same `code/03_behavioral.py` / `code/05_human_comparison.py`
pipeline you ran locally — only the *machine* changes. On a CUDA GPU the code's
`device_map="auto"` automatically uses the GPU, so 7B / Llama-8B become feasible.

## 0. Copy the project to the cluster (rsync, NOT git clone)
The stimuli PDFs and the built master CSV are gitignored, so `git clone` would be
missing the data. Copy the whole folder instead:

```bash
# from your Mac (kerb = your MIT Kerberos ID)
# Correct hostname per ORCD docs: orcd-login.mit.edu
rsync -av --exclude '.venv' --exclude 'outputs/' \
  "/Users/Aimee/Desktop/Summer/ToM Project/" \
  <kerb>@orcd-login.mit.edu:~/tom_project/
```
(Exclude `.venv` — it's a macOS build; you'll make a fresh Linux one on the cluster.)
Note: rsync will prompt for your MIT Kerberos password + a Duo push. If you want to
skip 2FA each time, log into https://orcd-ood.mit.edu first (that caches a short-lived
SSH credential), or set up SSH key forwarding per the ORCD docs.

## 1. One-time setup on a LOGIN node (has internet)
```bash
ssh <kerb>@orcd-login.mit.edu    # correct Engaging login hostname
cd ~/tom_project
# (optional, for gated Llama) export HF_TOKEN=hf_xxx   or:  huggingface-cli login
bash engaging/setup_engaging.sh
```
This builds `.venv`, installs deps, rebuilds the master CSV if needed, and
**pre-downloads all models** into `HF_HOME` so the GPU job runs offline.

## 2. Submit the GPU job
First edit the two `EDIT:` lines in `engaging/run_behavioral.sbatch`
(`--partition`, maybe `--account`). Find your options with `sinfo -s`.
```bash
sbatch engaging/run_behavioral.sbatch
squeue -u $USER          # watch it
tail -f outputs/logs/tom_behavioral_*.out
```

## 3. Pull results back to your Mac
```bash
rsync -av <kerb>@orcd-login.mit.edu:~/tom_project/outputs/ \
  "/Users/Aimee/Desktop/Summer/ToM Project/outputs/"
```

## Notes
- **Module names vary.** If `module load` lines fail, run `module avail` and edit
  `setup_engaging.sh` + `run_behavioral.sbatch` to match (python/conda + cuda).
- **Quotas:** keep `HF_HOME` on scratch (`/pool001/$USER` or `/nobackup1/$USER`),
  not your small home directory.
- **Human ground truth:** fill `dataset/human_reference/human_reference.csv` from the
  papers in `dataset/human_reference/README.md` before step 2 if you want the
  model-vs-human (adult/child) comparison; otherwise `05` reports model profiles only.
