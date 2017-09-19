all:
	#pip3 install scikit-learn
	@echo "Downloading solutions..."
	rm -rf myoutput
	mkdir myoutput
	wget www.liv.ic.unicamp.br/~quispe/stabilization-results/affine_200_cosine_30_8_False_p2-1-0_joint.avi
	wget www.liv.ic.unicamp.br/~quispe/stabilization-results/affine_200_cosine_30_8_False_p2-1-0_keypoints.avi
	wget www.liv.ic.unicamp.br/~quispe/stabilization-results/affine_200_cosine_30_8_False_p2-1-1_joint.avi
	wget www.liv.ic.unicamp.br/~quispe/stabilization-results/affine_200_cosine_30_8_False_p2-1-1_keypoints.avi
	wget www.liv.ic.unicamp.br/~quispe/stabilization-results/affine_200_cosine_30_8_False_p2-1-2_joint.avi
	wget www.liv.ic.unicamp.br/~quispe/stabilization-results/affine_200_cosine_30_8_False_p2-1-2_keypoints.avi
	mv *.avi myoutput
	@echo "Running experiments"
	python3 src/video-stabilization.py
	
clean:
	rm -rf output/*
	rm -f *.zip
