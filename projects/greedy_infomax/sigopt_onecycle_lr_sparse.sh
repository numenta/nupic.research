# Run loop 20 times.
for i in {1..40}; do
    # Run 1 trial
    echo "Running trial $i"
    python nta/nupic.research/projects/greedy_infomax/run.py -e one_cycle_lr_dimensionality_sigopt -j 0 -x 0 -g 8 --local-mode --single_instance  --wandb
    # Pause for 30 seconds before the next.
    sleep 20
done