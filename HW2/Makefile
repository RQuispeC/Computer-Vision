all: results input
	#pip3 install scikit-learn
	@echo "Downloading solutions..."
	@echo "Running experiments"
	python3 src/video-stabilization.py

results:
	rm -rf myoutput
	mkdir myoutput
	wget www.liv.ic.unicamp.br/~quispe/stabilization-results/affine_200_cosine_30_8_False_p2-1-0_joint.avi
	wget www.liv.ic.unicamp.br/~quispe/stabilization-results/affine_200_cosine_30_8_False_p2-1-0_keypoints.avi
	wget www.liv.ic.unicamp.br/~quispe/stabilization-results/affine_200_cosine_30_8_False_p2-1-1_joint.avi
	wget www.liv.ic.unicamp.br/~quispe/stabilization-results/affine_200_cosine_30_8_False_p2-1-1_keypoints.avi
	wget www.liv.ic.unicamp.br/~quispe/stabilization-results/affine_200_cosine_30_8_False_p2-1-2_joint.avi
	wget www.liv.ic.unicamp.br/~quispe/stabilization-results/affine_200_cosine_30_8_False_p2-1-2_keypoints.avi
	wget www.liv.ic.unicamp.br/~quispe/stabilization-results/projective_200_cosine_30_8_False_p2-1-0_joint.avi
	wget www.liv.ic.unicamp.br/~quispe/stabilization-results/projective_200_cosine_30_8_False_p2-1-0_keypoints.avi
	mv *.avi myoutput

input:
	wget www.liv.ic.unicamp.br/~quispe/stabilization-input/p2-1-0.avi
	wget www.liv.ic.unicamp.br/~quispe/stabilization-input/p2-1-2.avi
	mv *.avi input

clean:
	rm -rf output/*
	rm -f *.zip
