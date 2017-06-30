# Hyper-Prameter Optimization Library using GP-UCB

<!---
[![Build Status](https://travis-ci.org/LittleWat/gp_ycb_python_lib.svg?branch=master)](https://travis-ci.org/LittleWat/gp_ycb_python_lib)
-->


## Usage

1. Copy the sample directory (ex. samples/svm). 

2. Edit 5 or 6 files.

    1. param_dir/generate_hoge_param_file.py (You don't necessarily use this script.)
    
        This file generate hyper-paramteres. 
        
    2. cmdline_hoge.txt
    
        This file is a cmd-line script that calls your machine learning programs that contains hyper-parameter.
        
        If you need a config file that contains hyper-paramter to call your program, the config file sould be set "$param_file"
        
    3. parameter_hoge.json ("Optional")
    
        This file is used to kick your program. Also, you should set a paramter in order to get the result of your program (ex. "filename_result": "./libsvm/output/accuracy_$model_number.txt").
        
    4. parameter_gp.json
    
        This file is used to set paramers of Gaussian Process.
        
    5. env_hoge.py
    
        You have to implement "get_result" function.

        ```
        from env import Cmdline_Environment
        
        
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
#### libsvm optimization

![sample](_static/svm_anim.gif)

