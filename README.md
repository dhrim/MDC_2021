# Python 기반 AI활용 데이터 분석가 양성 과정

- [Python 라이브러리를 활용한 데이터 분석 및 시각화](material/library_for_deep_learning.md)
- [딥러닝](material/deep_learning.md)
    - 딥러닝 기초 - DNN, CNN, RNN 등
    - 딥러닝을 이용한 영상 데이터 분석
    - 시계열 금융데이터 분석
    - 딥러닝을 이용한 추천 서비스 분석

<br>

# 실습한 자료들

[material/deep_learning/practice](material/deep_learning/practice/)

<br>

# 딥러닝 활용을 위한 지식 구조

```
Environment
    jupyter
	colab
	usage
		!, %, run
    GCP virtual machine
linux
	ENV
	command
		cd, pwd, ls
		mkdir, rm, cp
		head, more, tail, cat
	util
		apt
		git, wget
		grep, wc, tree
		tar, unrar, unzip
	gpu
		nvidia-smi

python
	env
		python
			interactive
			execute file
		pip
	syntax
        variable
        data
            tuple
            list
            dict
            set
        loop
        if
        comprehensive list
        function
        class
	module
		import

libray
    numpy
        load
        operation
        shape
        slicing
        reshape
        axis + sum, mean
    pandas
        load
        view
	    operation
        to numpy
    seaborn
        charts
    matplot
        plot
        scatter
        hist
        multi draw
        show image

Deep Learning
    DNN
        concept
            layer, node, weight, bias, activation
            cost function
            GD, BP
        data
            x, y
            train, validate, test
            shuffle
        learning curve : accuracy, loss
        tuning
            overfitting, underfitting
            dropout, batch normalization, regularization
            data augmentation
        Transfer Learning
    type
        supervised
        unsupervised
        reinforcement
    model
        CNN
            vanilla, named CNN
        RNN
        GAN
    task
        Classification
        Object Detection
        Generation
	Segmentation
	Pose Extraction
	Noise Removing
	Super Resolution
	Question answering
	Auto Captioning
    data type
    	attribute data
	image data
	natural language data
	time series data

TensorFlow/Keras
    basic frame
        data preparing
            x, y
            train, valid, test
            normalization
            ImageDataGenerator
        fit
        evaluate
        predict
    model
        activation function
        initializer
    tuning
        learning rate
        regularizer
        dropout
        batch normalization
    save/load
    compile
        optimizer
        loss
        metric
```

   
