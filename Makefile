pre:
	pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
	pip install mmcv-full==1.4.7 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8.0/index.html
	mkdir -p thirdparty
	git clone -b v0.22.1 --depth=1 https://github.com/open-mmlab/mmsegmentation.git thirdparty/mmsegmentation
	cd thirdparty/mmsegmentation && pip install -v -e .
install:
	make pre
	pip install -v -e .
clean:
	rm -rf thirdparty
	rm -r dass.egg-info
