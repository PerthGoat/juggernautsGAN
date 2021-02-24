http://yann.lecun.com/exdb/mnist/

python 3.8.8
using
tensorflow 2.4.1
on
64 bit


NVIDIA CUDA 11.2
Symbolic link a file
New-Item -ItemType SymbolicLink -Path .\cusolver64_10.dll -Target .\cusolver64_11.dll
in 
CUDA\v11.2\bin

Install cudnn as well, version 8 (this one is hellish, nvidia developer lock-page), just copy it into the same directory