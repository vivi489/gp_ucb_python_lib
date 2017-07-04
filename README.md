# Hyper-Prameter Optimization Library using GP-UCB

<!---
[![Build Status](https://travis-ci.org/LittleWat/gp_ycb_python_lib.svg?branch=master)](https://travis-ci.org/LittleWat/gp_ycb_python_lib)
-->

## Install
```
python setup.py install
```


## Usage

1. Copy the sample directory (ex. samples/svm). 

2. Edit 5 or 6 files in your copied directory. ("HOGE" should be set to your model name.)

    1. param_dir/generate_HOGE_param_file.py (You don't necessarily use this script.)
    
        This file generate hyper-paramteres. 
        
    2. cmdline_HOGE.txt
    
        This file is a cmd-line script that calls your machine learning programs that contains hyper-parameter.
        
        If you need a config file that contains hyper-paramter to call your program, the config file sould be set "$param_file"
        
    3. parameter_HOGE.json (This file is optional, but useful in "env_HOGE.py")
    
        This file is used to kick your program. Also, you should set a paramter in order to get the result of your program (ex. "filename_result": "./libsvm/output/accuracy_$model_number.txt").
        
    4. parameter_gp.json
    
        This file is used to set paramers of Gaussian Process.
        
    5. myenv.py
    
        You have to implement "get_result" function.

        ```python
        from gphypo.env import Cmdline_Environment
        
        
        class MyEnvironment(Cmdline_Environment):
            def get_result(self):
                
                ### WRITE BELOW ###
                
                
                res = GET_YOUR_PROGRAMS_OUTPUT
                ################
                
                return res
        ```

     6. run.sh
        You have to call "lib/run_cmdenv.py".
        

## Demo
GP kernel was set below

```python
 my_kernel = C(1, constant_value_bounds="fixed") * RBF(2,
                                                      length_scale_bounds="fixed")  # works well, but not so sharp
if noise:
    my_kernel += WhiteKernel(1e-1)
```
 
### gaussian optimization
![sample](_static/gaussian_anim.gif)

### libsvm optimization
Train and test dataset is the same iris dataset. (not good)

![sample](_static/svm_anim.gif)

### lda optimization

![sample](_static/lda_anim.gif)


