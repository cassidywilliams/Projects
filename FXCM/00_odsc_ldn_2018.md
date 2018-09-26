Algorithmic Trading with ML & DL
================================

Dr. Yves J. Hilpisch | The Python Quants GmbH

Workshop at ODSC London on 20. September 2018

See http://odsc.com/london.

<img src="http://hilpisch.com/images/finaince_visual_shadow.png" width=300px>

Links
---------
* http://tpq.io
* http://pqp.io
* http://training.tpq.io
* http://certificate.tpq.io
* http://twitter.com/dyjh

Link to Slides
--------------
http://hilpisch.com/odsc_ldn_2018.pdf

Short Link to Gist
------------------
http://bit.ly/odsc_ldn_2018


Python
------

You should have installed either **Anaconda** or **Miniconda**.


If you do not have a Python installation/environment with the **core scientific packages**, do on the shell do:

    conda create -n odsc python=3.6 jupyter pandas nomkl scikit-learn matplotlib
    conda activate odsc

In any case do the following to install the `fxcmpy` package:

    pip install --upgrade pip
    pip install fxcmpy


FXCM
----

You should open a **demo account** under

https://www.fxcm.com/uk/forex-trading-demo/


The **documentation** for `fxcmpy` is found under

http://fxcmpy.tpq.io


You should generate a **token** in your account under `Token Management`.

Create your **working folder** where you put in all subsequently created files.

Then create a configuration file with name `fxcm.cfg` and the following content:

    [FXCM]
    log_level = error
    log_file = 'fxcm.log'
    access_token = YOUR_FXCM_API_TOKEN
