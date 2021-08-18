# Run loop 20 times.
for i in {1..8}; do
    # Run 1 trial
    echo "Running trial $i"
    python run.py -e sigopt_dense_base -j 0 -x 0 -g 8 --local-mode --single_instance --wandb
    # Pause for 30 seconds before the next.
    sleep 30
done