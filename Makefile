red:=$(shell tput bold ; tput setaf 1)
green:=$(shell tput bold ; tput setaf 2)
yellow:=$(shell tput bold ; tput setaf 3)
blue:=$(shell tput bold ; tput setaf 4)
magenta:=$(shell tput bold ; tput setaf 5)
cyan:=$(shell tput bold ; tput setaf 6)
reset:=$(shell tput sgr0)


data/TOY:
	python gen_toy.py --dest $@ -n 10 10 -wh 256 256 -r 50

data/TOY2:
	rm -rf $@_tmp $@
	python gen_two_circles.py --dest $@_tmp -n 1000 100 -r 25 -wh 256 256
	mv $@_tmp $@


# Extraction and slicing for Segthor
# data/segthor_train: data/segthor_train.zip
# 	$(info $(yellow)unzip $<$(reset))
# 	sha256sum -c data/segthor_train.sha256
# 	unzip -q $<

# Use with gaussian dataset + testset:
data/SEGTHOR: data/segthor_train
	$(info $(green)python $(CFLAGS) slice_segthor.py$(reset))
	rm -rf $@_tmp $@
	python $(CFLAGS) slice_segthor.py --source_dir $^ --dest_dir $@_tmp \
		--shape 256 256 --retain 19 --with_gaussian_dataset --test_mode
# ^^ retaining 19 because dataset size doubled from 35 to 75
	mv $@_tmp $@

# # Use without gaussian dataset but with testset:
# data/SEGTHOR: data/segthor_train
# 	$(info $(green)python $(CFLAGS) slice_segthor.py$(reset))
# 	rm -rf $@_tmp $@
# 	python $(CFLAGS) slice_segthor.py --source_dir $^ --dest_dir $@_tmp \
# 		--shape 256 256 --retain 8 --test_mode
# 	mv $@_tmp $@
