all:
	@echo "Running convolution experiments"
	python3 src/convolution.py
	@echo "Running convolution experiments-openCV"
	python3 dbg/convOpenCV.py
	@echo "Running gaussian pyramid experiments"
	python3 src/gaussian_pyramid.py
	@echo "Running laplacian pyramid experiments"
	python3 src/laplacian_pyramid.py
	@echo "Running blending experiments"
	python3 src/blending.py
	@echo "Running fourier transform experiments"
	python3 src/fourier.py
	
clean:
	rm -rf output/*
	rm -f *.zip
