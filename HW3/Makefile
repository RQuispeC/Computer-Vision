all: data
	#pip3 install scikit-learn
	@echo "Downloading solutions..."
	@echo "Running experiments"
	python3 src/structure_from_motion.py
	@echo "Results .ply is in output/points.ply"

data:
	wget www.liv.ic.unicamp.br/~quispe/structure-from-motion/p3-1-1.mp4
	wget www.liv.ic.unicamp.br/~quispe/structure-from-motion/p3-1-2.mp4
	mv *.mp4 input

clean:
	rm -rf output/*
	rm -f *.zip
