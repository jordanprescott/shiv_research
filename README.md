run 3 introduces the indicator of scene overlap and changes the preprocessing to use random scenes

run 4 introduces the use of triplets in training. 4 now has validation loss as well as access to the larger datset, and is performing very well in the pose estimation. need to look into potentially not worrying about knowing about overlapping frames.

run 5 combines triplets, overlap indicator, and skips in the processing of the xyz TUM dataset to indicate no overlap for training data

run 6 switched to pairs again, but now with images and depth maps as the inputs

NEED TO:
 - add more data to the training and testing
 - if there is a scene skip it does not try to estimate the pose, uses pairs of depth maps and images
 - want to check that, when adding in sp+sg that the skipping is useful. If no correspondences are found with sp+sg, then can use that as indicator
 - TOMORROW: add sp+sg into run6 now that accuracy is acceptable