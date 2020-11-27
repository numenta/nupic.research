# OML vs ANML vs Ours

# Meta-Training

## In The ANML Paper
```
for e in epochs (20000 total)
  sample 1 task

  # inner loop
  sample 20 images (if more then 1 task, done evenly across all of them)
  for i in images
    do one grad step; done w/ single image over fast params

  # outer loop
  sample 20 new images from the one task (these will be the same, since there are only 20 per class)
  sample 64 new images across all tasks (replay set)
  do one grad step; done w/ combined 128 images over fast and slow params
```


## In The OML Paper

```
for e in epoch (not sure how many)
  sample 3 tasks
  
  # inner loop
  sample 5 images from each task
  for each task:
    for i in images
      do one grad step; done w/ single image over fast params

  # outer loop
  sample 5 new images for each task (15 total)
  sample 15 replay images
  do one grad step; done w/ the combined 30 images over fast and slow params
```


## In Our Previous Implementation
This is prior to [PR #389](https://github.com/numenta/nupic.research/pull/389/commits/6e7bf3bb7728419da95b0766c1034b62a033d6c8)

Note: Our implementation after that PR supports both OML and ANML meta-training.

```
for e in epochs (only 1000)
  sample 10 tasks

  # inner loop
  for t in tasks
    sample 5 images
    do one grad step; done w/ all 5 images over fast params
  
  # outer loop
  sample 64 new images across the 10 tasks
  sample 64 new images across all tasks (remeber set)
  do one grad step; done w/ combined 128 images over fast and slow params
```

# Meta-Testing

## OML and ANML
```
for r in num_runs (anwhere from 15 to 50)
  sample t tasks (either 10, 50, or 100, ect)
  retrieve 15 images per task; hold-out 5 per task for testing

  # meta-test training: uses 15 × t images
  for each img; going sequentially, one task at a time
    do one grad step; done w/ just the one image

  # meta-test testing: using 5 × t images
  evaluate model on held out images
```

## In Our Previous Implementation
This is prior to [PR #389](https://github.com/numenta/nupic.research/pull/389/commits/6e7bf3bb7728419da95b0766c1034b62a033d6c8)

Note: Our implementation after that PR supports both OML and ANML meta-testing.

```
for r in num_runs (we used 15)
  sample t tasks (either 10, 50, or 100, ect)
  retrieve 15 images per task; hold-out 5 each for testing

  # meta-test training: uses 15 × t images
  for each img; sampled iid over across all the t tasks
    do one grad step; done w/ just the one image

  # meta-test testing: using 5 × t images
  evaluate model on held out images
```