# Code-and-Data-for-IJDS

The sample data includes 10 '.m' files and a text file for explaning the 10 '.m' data.  

Please install tensor calculation toolbox before running the code.




Explanation for the Data: 

'randsequence' is new randomized sequence of 1-500.

'X_4D_complete_randseq' is a 4D high-dimensional tensor from the file 'Datageneration', where the first 3 mode is the 3 dimensions of a image stream,
the last dimension is the sample number. The 4D tensor is randomized sequence and has no missing entries.

'X_4D_complete_TC_randseq' is a 4D high-dimensional tensor which is applied 'TensorCompletion' to 'X_4D_complete_randseq'

'Miss10%', 'Miss50%', 'Miss90%' are the missing percentage towards 'X_4D_complete_randseq'.

'Ym_t_great23_randseq' is the TTF which has the same randomized sequence as the 4D tensors and the threshold is 23.
